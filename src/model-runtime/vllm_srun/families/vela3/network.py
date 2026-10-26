"""Vela 3.0 model inputs, how they batch, and the readout of each network.

Every input is one marker sequence (``vela2.encoder_layout.EncoderSequence``)
for one network: the decision backbone (``network`` None) or an adapter (the
engine's adapter branch of that name). An input belongs to a ``Call``: the
forward calls the released runtime makes for one request (all its decision
questions; one router group on one adapter; one span question), each batched
as that runtime batches it, by length, greedily under a token budget and a row
limit. Inputs of different calls never share a forward, so a request's answers
do not depend on what else runs with it.

Each network reads its own marker head (``heads.marker.MarkerHead``, readout
``both``): the backbone's from ``model.safetensors``, an adapter's from its
file, which holds only the readout tensors it trained apart from the
backbone's (the others are the backbone's own, as ``build_adapter`` shares
them). A head with no word projections answers no span question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...heads.marker import MarkerHead
from ...plugins.base import EncoderBatch, EngineModel
from ..vela2.encoder_layout import EncoderSequence, batch_indices, split_outputs
from ..vela2.members import INDEX_INPUTS, read_tensors
from .package import BACKBONE_FILE, Vela3Package

if TYPE_CHECKING:
    from numpy.typing import NDArray

HEAD_PREFIXES = ("norm.", "w_", "log_tau", "cls_mlp.")
DECISION_BUDGET = 65536


@dataclass(frozen=True, eq=False)
class Call:
    """One forward call of the released runtime: its network and batching (token budget, row limit)."""

    network: str | None
    budget: int
    rows: int | None = None


@dataclass(eq=False)
class Vela3Item:
    """One model input of a request (a scheduler work item)."""

    sequence: EncoderSequence
    call: Call
    ids: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ids = self.sequence.ids


Output = tuple[list["NDArray[np.float32]"], "NDArray[np.float32] | None"]


def batches(items: list[Vela3Item]) -> list[list[int]]:
    """The released runtime's forwards over ``items``: per call, by length under its budget and row limit."""
    calls: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        calls.setdefault(id(item.call), []).append(index)
    groups: list[list[int]] = []
    for members in calls.values():
        call = items[members[0]].call
        current: list[int] = []
        for index in sorted(members, key=lambda i: len(items[i].ids)):
            width = max(len(items[i].ids) for i in [*current, index])
            if current and (
                (call.rows is not None and len(current) + 1 > call.rows)
                or width * (len(current) + 1) > call.budget
            ):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
    return groups


def _head(hidden: int, projection: int, tensors: dict[str, torch.Tensor]) -> MarkerHead:
    spans = "w_t.weight" in tensors
    head = MarkerHead(hidden, projection, "both", spans=spans)
    head.load_state_dict({k: v.float() for k, v in tensors.items()}, strict=True)
    return head.float().eval()


class Networks:
    """The marker heads of the backbone and of every adapter, and the forward over a batch."""

    def __init__(self, package: Vela3Package, engine_model: EngineModel):
        config = package.config
        hidden = int(config["hidden_size"])
        projection = int(config["decision"]["proj_dim"])
        device = engine_model.device
        base = read_tensors([package.root / BACKBONE_FILE], HEAD_PREFIXES)
        self.heads: dict[str | None, MarkerHead] = {
            None: _head(hidden, projection, base).to(device)
        }
        for name, adapter in package.adapters.items():
            own = read_tensors([package.root / adapter.file], HEAD_PREFIXES)
            merged = {**base, **own}
            if "w_t.weight" not in own:
                merged = {
                    k: v for k, v in merged.items() if not k.startswith(("w_t", "w_e"))
                }
                merged.pop("log_tau_span", None)
            self.heads[name] = _head(hidden, projection, merged).to(device)
        self.engine_model = engine_model
        self.pad = int(config["pad_token_id"])

    def parameters(self) -> int:
        """Readout parameters the family adds, each shared tensor once."""
        seen: dict[int, int] = {}
        for head in self.heads.values():
            for parameter in head.parameters():
                seen[id(parameter)] = parameter.numel()
        return sum(seen.values())

    def run(self, items: list[Vela3Item]) -> list[Output]:
        """Each input's (option logits per question, word x label block), in the released forwards."""
        results: list[Any] = [None] * len(items)
        for group in batches(items):
            batch = [items[index].sequence for index in group]
            network = items[group[0]].call.network
            for index, value in zip(group, self._forward(batch, network), strict=True):
                results[index] = value
        return results

    def _forward(
        self, batch: list[EncoderSequence], network: str | None
    ) -> list[Output]:
        width = max(len(sequence.ids) for sequence in batch)
        ids = torch.full((len(batch), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, sequence in enumerate(batch):
            ids[row, : len(sequence.ids)] = torch.tensor(sequence.ids, dtype=torch.long)
            mask[row, : len(sequence.ids)] = 1
        output = self.engine_model.encode(EncoderBatch(ids, mask, branch=network))
        hidden = output.hidden[max(output.hidden)]
        indices = batch_indices(batch)
        with torch.inference_mode():
            option_logits, span_logits = self.heads[network](
                hidden,
                *(
                    torch.from_numpy(indices[name]).to(hidden.device)
                    for name in INDEX_INPUTS
                ),
            )
        return split_outputs(
            batch,
            option_logits.float().cpu().numpy(),
            span_logits.float().cpu().numpy(),
        )


def adapter_files(package: Vela3Package) -> dict[str, tuple[Path, int, float]]:
    """Every adapter's file, LoRA rank and alpha, by adapter (engine branch) name."""
    return {
        name: (package.root / adapter.file, adapter.rank, adapter.alpha)
        for name, adapter in package.adapters.items()
    }
