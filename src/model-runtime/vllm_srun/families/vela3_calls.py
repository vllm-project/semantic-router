"""Vela 3.0 model inputs, how they batch into the released forwards, and the marker heads that read them.

Every input is one marker sequence (``vela2.encoder_layout.EncoderSequence``)
for one network: the backbone (``network`` None) or an adapter (the engine's
adapter branch of that name). An input belongs to a ``Call``: one forward call
of the released runtime (all of a request's decision questions; one router
group on one adapter; one span question; a pico's one-pass sequence), each
batched as that runtime batches it, by length, greedily under a token budget
and a row limit. In ``MarkerNetworks.run`` inputs of different calls never
share a forward, so a request's answers do not depend on what else runs with
it. A batch that merges concurrent requests (``merged``) lets equal calls of
different requests share forwards, which only padding and GEMM shapes tell
apart.

Each network reads its own marker head (``heads.marker.MarkerHead``, readout
``both``). A head with no word projections answers no span question. The
Vela 3.0 families (``vela3``, ``vela3_pico``) share this module rather than one
importing the other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..heads.marker import MarkerHead
from ..plugins.base import EncoderBatch, EngineModel
from ..scheduler.planner import length_class
from .vela2.encoder_layout import EncoderSequence, batch_indices, split_outputs
from .vela2.members import INDEX_INPUTS

if TYPE_CHECKING:
    from numpy.typing import NDArray

HEAD_PREFIXES = ("norm.", "w_", "log_tau", "cls_mlp.")
SPAN_TENSORS = ("w_t.", "w_e.", "log_tau_span")
DECISION_BUDGET = 65536


@dataclass(frozen=True)
class Call:
    """One forward call of the released runtime: its network and batching (token budget, row limit).

    ``turn`` counts the request's earlier calls of the same network and limits,
    so equal calls come from different requests, never from one.
    """

    network: str | None
    budget: int
    rows: int | None = None
    turn: int = 0


@dataclass(eq=False)
class Vela3Item:
    """One model input of a request (a scheduler work item)."""

    sequence: EncoderSequence
    call: Call
    ids: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ids = self.sequence.ids


Output = tuple[list["NDArray[np.float32]"], "NDArray[np.float32] | None"]


def batches(items: list[Vela3Item], merged: bool = False) -> list[list[int]]:
    """The released runtime's forwards over ``items``: per call, by length under its budget and row limit.

    ``merged`` batches equal calls together, so concurrent requests share forwards,
    each holding rows of one length class so a long request does not pad short ones.
    """
    calls: dict[object, list[int]] = {}
    for index, item in enumerate(items):
        calls.setdefault(item.call if merged else id(item.call), []).append(index)
    groups: list[list[int]] = []
    for members in calls.values():
        call = items[members[0]].call
        current: list[int] = []
        for index in sorted(members, key=lambda i: len(items[i].ids)):
            width = max(len(items[i].ids) for i in [*current, index])
            if current and (
                (call.rows is not None and len(current) + 1 > call.rows)
                or width * (len(current) + 1) > call.budget
                or (
                    merged
                    and length_class(width) != length_class(len(items[current[0]].ids))
                )
            ):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
    return groups


def marker_head(
    hidden: int, projection: int, tensors: dict[str, torch.Tensor], spans: bool
) -> MarkerHead:
    """A marker head from its tensors; without ``spans``, any word projections among them are left out."""
    if not spans:
        tensors = {k: v for k, v in tensors.items() if not k.startswith(SPAN_TENSORS)}
    head = MarkerHead(hidden, projection, "both", spans=spans)
    head.load_state_dict({k: v.float() for k, v in tensors.items()}, strict=True)
    return head.float().eval()


class MarkerNetworks:
    """Marker heads by network name, and the forward over a batch of one call's inputs."""

    def __init__(
        self, heads: dict[str | None, MarkerHead], engine_model: EngineModel, pad: int
    ):
        self.heads = heads
        self.engine_model = engine_model
        self.pad = pad

    def parameters(self) -> int:
        """Readout parameters the family adds, each shared tensor once."""
        seen: dict[int, int] = {}
        for head in self.heads.values():
            for parameter in head.parameters():
                seen[id(parameter)] = parameter.numel()
        return sum(seen.values())

    def run(self, items: list[Vela3Item], merged: bool = False) -> list[Output]:
        """Each input's (option logits per question, word x label block), in the released forwards."""
        results: list[Any] = [None] * len(items)
        for group in batches(items, merged):
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
