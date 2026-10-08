"""The two Vela 2.0 members: how their rows become model inputs, run, and reduce to raw outputs.

The 0.3B member pads its marker sequences into one encoder batch; the engine
returns the last hidden states (native) or the published graph's outputs
(ONNX graph with the readout baked in), and the schema readout runs in the
family. The decoder member (0.8B, 4B, 9B) runs each tree once on the engine's
shared-context forward (the parts computed once, every block continuing from
them) and reads the blocks with the candidate head and the span heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...heads.candidate import CandidateHead
from ...heads.marker import MarkerHead
from ...heads.span import SpanHead
from ...plugins.base import EncoderBatch, EncoderOutput, EngineModel, TreeBatch
from .calibration import BROAD_HEAD, ROUTER_HEAD
from .decoder_layout import Block, DecoderLayout, DecoderTree, RowTrees
from .dispatch import Dispatcher
from .encoder_layout import EncoderLayout, EncoderSequence, batch_indices, split_outputs
from .layout import Row, SchemaTooLongError, Tokens
from .package import Vela2Package
from .raw import RawRow

if TYPE_CHECKING:
    from numpy.typing import NDArray

GRAPH_OUTPUTS = ("opt_logits", "span_logits")
INDEX_INPUTS = ("q_index", "opt_index", "unit_index", "ent_index")
# The 0.3B engine's batching: padded tokens per forward by device class, and rows per forward.
ENCODER_BUDGET = {"cpu": 16_384, "gpu": 32_768}
# The decoders' engine batching: tokens per forward (whole trees), and rows per forward.
DECODER_BUDGET = 24_576
MAX_ROWS = 32


def read_tensors(paths: list[Path], keep: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """The tensors whose names start with one of ``keep``, from safetensors files."""
    from safetensors import safe_open

    out = {}
    for path in paths:
        with safe_open(str(path), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open has no __iter__
                if name.startswith(keep):
                    out[name] = handle.get_tensor(name)
    return out


def load_prefixed(
    module: torch.nn.Module, tensors: dict[str, torch.Tensor], prefix: str
) -> None:
    """Load the ``prefix.*`` tensors into ``module`` strictly, as FP32."""
    state = {
        name[len(prefix) :]: value.float()
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    module.load_state_dict(state, strict=True)


def _unpack(tokens: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Packed ``[N, H]`` hidden states as right-padded ``[rows, width, H]`` rows (zeros past each length)."""
    width = max(lengths)
    valid = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    rows = tokens.new_zeros(len(lengths), width, tokens.shape[-1])
    rows[valid.to(tokens.device)] = tokens
    return rows


def _pad(rows: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return ids, mask


class EncoderMember:
    """Vela 2.0 0.3B: marker sequences, schema readout (in the family or baked into a graph).

    An engine that runs the published graph returns its ``opt_logits`` and
    ``span_logits``; one that returns hidden states gets the readout here.
    Which one the loaded engine does is probed once at load.
    """

    def __init__(self, package: Vela2Package, engine_model: EngineModel):
        config = package.config
        self.layout = EncoderLayout(config)
        self.engine_model = engine_model
        readout = MarkerHead(
            config["encoder_config"]["hidden_size"],
            int(config["proj_dim"]),
            config["readout"],
        )
        tensors = read_tensors(
            list(package.weights), ("norm.", "w_", "log_tau", "cls_mlp.")
        )
        readout.load_state_dict({k: v.float() for k, v in tensors.items()}, strict=True)
        self.readout = readout.float().eval().to(engine_model.device)
        probe, _ = self._encode(
            [EncoderSequence(ids=[self.layout.bos, self.layout.eos], questions=[])]
        )
        self.graph = bool(probe.outputs)

    def parameters(self) -> int:
        """Readout parameters the family adds (none when the graph holds the readout)."""
        return 0 if self.graph else sum(p.numel() for p in self.readout.parameters())

    def plan(
        self, rows: list[Row], tokens: Tokens, in_windows: bool = True
    ) -> tuple[list[EncoderSequence], list[list[int] | None]]:
        """Sequences of every row and, per row, the indices of its sequences (None when it cannot fit).

        Without ``in_windows`` a row is one sequence, its parts cut to fit.
        """
        items: list[EncoderSequence] = []
        groups: list[list[int] | None] = []
        for row in rows:
            try:
                sequences = self.layout.sequences(row, tokens, in_windows)
            except SchemaTooLongError:
                groups.append(None)
                continue
            groups.append(list(range(len(items), len(items) + len(sequences))))
            items.extend(sequences)
        return items, groups

    def _encode(
        self, items: list[EncoderSequence], packed: bool = False, reduced: bool = False
    ) -> tuple[EncoderOutput, dict[str, NDArray[np.int64]]]:
        """One batch through the engine: padded, asking for the graph outputs (graph engines return
        them), or packed back to back (hidden states only); ``reduced`` asks for the engine's
        reduced copy, which runs where the engine loaded one."""
        indices = batch_indices(items)
        if packed:
            ids = torch.tensor(
                [i for item in items for i in item.ids], dtype=torch.long
            )
            lengths = [len(item.ids) for item in items]
            return (
                self.engine_model.encode(
                    EncoderBatch(ids, None, lengths=lengths, reduced=reduced)
                ),
                indices,
            )
        input_ids, attention_mask = _pad([item.ids for item in items], self.layout.pad)
        return (
            self.engine_model.encode(
                EncoderBatch(
                    input_ids,
                    attention_mask,
                    graph_inputs={
                        name: torch.from_numpy(value) for name, value in indices.items()
                    },
                    outputs=GRAPH_OUTPUTS,
                    reduced=reduced,
                )
            ),
            indices,
        )

    def batches(self, items: list[EncoderSequence]) -> list[list[int]]:
        """The packages' batching: by length, greedily under the token budget and 32 rows."""
        device = "cpu" if self.engine_model.device.type == "cpu" else "gpu"
        budget = ENCODER_BUDGET[device]
        groups: list[list[int]] = []
        current: list[int] = []
        for index in sorted(range(len(items)), key=lambda i: len(items[i].ids)):
            width = max(len(items[i].ids) for i in [*current, index])
            if current and (
                len(current) + 1 > MAX_ROWS or width * (len(current) + 1) > budget
            ):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
        return groups

    def run(
        self, items: list[EncoderSequence], packed: bool = False, reduced: bool = False
    ) -> list[Any]:
        """Each sequence's (option logits, word x label block), batched and padded as the packages do.

        ``packed`` (approximate batches, hidden-state engines) packs each batch's
        sequences back to back, so no projection runs on padding, and the
        backbone's packed layout attends locally in query blocks on long rows.
        ``reduced`` (approximate batches) runs them on the engine's reduced copy
        where it loaded one; the readout stays FP32.
        """
        packed = packed and not self.graph
        results: list[Any] = [None] * len(items)
        for group in self.batches(items):
            batch = [items[index] for index in group]
            output, indices = self._encode(batch, packed, reduced)
            if self.graph:
                option_logits, span_logits = (
                    output.outputs[name] for name in GRAPH_OUTPUTS
                )
            else:
                hidden = output.hidden[max(output.hidden)]
                device = hidden.device
                if packed:
                    hidden = _unpack(hidden, [len(item.ids) for item in batch])
                with torch.inference_mode():
                    option_logits, span_logits = self.readout(
                        hidden,
                        *(
                            torch.from_numpy(indices[name]).to(device)
                            for name in INDEX_INPUTS
                        ),
                    )
            split = split_outputs(
                batch,
                option_logits.float().cpu().numpy(),
                span_logits.float().cpu().numpy(),
            )
            for index, value in zip(group, split, strict=True):
                results[index] = value
        return results

    def combine(
        self,
        rows: list[Row],
        groups: list[list[int] | None],
        items: list[EncoderSequence],
        results: list[Any],
    ) -> list[RawRow | None]:
        out: list[RawRow | None] = []
        for row, group in zip(rows, groups, strict=True):
            if group is None:
                out.append(None)
                continue
            out.append(
                self.layout.combine(
                    row, [items[i] for i in group], [results[i] for i in group]
                )
            )
        return out


class DecoderMember:
    """Vela 2.0 0.8B / 4B / 9B: trees through the engine's tree forward, candidate head and span heads.

    Exact: the packages' trees, batched as they batch them (rows' trees and
    window trees apart, by token count, under 24,576 tokens and 32 rows) and
    laid out as their engine lays them out (``layout: rows``). Packed (the
    shared-context path): trees with the same parts merge into one, so a
    request's questions share one parts pass, and each runs back to back.
    """

    def __init__(self, package: Vela2Package, engine_model: EngineModel):
        config = package.config
        hidden = config["backbone_config"]["hidden_size"]
        span = config.get("span_head") or {}
        tensors = read_tensors(list(package.weights), ("head.", "set_bias", "span2."))
        self.head = CandidateHead(hidden, int(config["head_dim"]))
        load_prefixed(self.head, tensors, "head.")
        self.set_bias = tensors["set_bias"].float().to(engine_model.device)
        self.spans: dict[str, SpanHead] = {
            ROUTER_HEAD: SpanHead(hidden, span.get("d", 256), span.get("slots", 64))
        }
        load_prefixed(self.spans[ROUTER_HEAD], tensors, "span2.")
        if package.broad_head is not None:
            broad = read_tensors([package.broad_head], ("span_broad.",))
            self.spans[BROAD_HEAD] = SpanHead(
                hidden,
                broad["span_broad.K.weight"].shape[0],
                broad["span_broad.slot.weight"].shape[0],
            )
            load_prefixed(self.spans[BROAD_HEAD], broad, "span_broad.")
        device = engine_model.device
        self.head = self.head.float().eval().to(device)
        self.spans = {
            name: head.float().eval().to(device) for name, head in self.spans.items()
        }
        self.engine_model = engine_model
        self.layout = DecoderLayout(
            config, Dispatcher(package.calibration, BROAD_HEAD in self.spans)
        )

    @property
    def broad_head(self) -> bool:
        return BROAD_HEAD in self.spans

    def parameters(self) -> int:
        modules = [self.head, *self.spans.values()]
        return (
            sum(p.numel() for m in modules for p in m.parameters())
            + self.set_bias.numel()
        )

    def plan(
        self, rows: list[Row], tokens: Tokens, in_windows: bool = True
    ) -> tuple[list[DecoderTree], list[RowTrees | None]]:
        """The rows' trees; a span target over the repeat limit is read in windows either way."""
        return self.layout.trees(rows, tokens)

    def batches(self, items: list[DecoderTree]) -> list[list[int]]:
        """The packages' batching of trees: rows' trees and window trees apart, by token count."""
        groups: list[list[int]] = []
        for window in (False, True):
            current: list[int] = []
            members = [i for i, tree in enumerate(items) if tree.window == window]
            for index in sorted(members, key=lambda i: len(items[i].ids)):
                width = max(len(items[i].ids) for i in [*current, index])
                if current and (
                    len(current) + 1 > MAX_ROWS
                    or width * (len(current) + 1) > DECODER_BUDGET
                ):
                    groups.append(current)
                    current = []
                current.append(index)
            if current:
                groups.append(current)
        return groups

    def run(
        self, items: list[DecoderTree], packed: bool = False
    ) -> list[list[NDArray[np.float32] | None]]:
        """Per tree, each block's readout in block order (None for a block nobody reads)."""
        results: list[list[NDArray[np.float32] | None]] = [
            [None] * len(tree.blocks) for tree in items
        ]
        if packed:
            shared: dict[tuple[int, ...], list[tuple[int, int]]] = {}
            for t, tree in enumerate(items):
                for b, block in enumerate(tree.blocks):
                    if block.read:
                        shared.setdefault(tuple(tree.prefix), []).append((t, b))
            forwards = [
                ([items[members[0][0]].prefix], members, [0] * len(members))
                for members in shared.values()
            ]
        else:
            forwards = []
            for group in self.batches(items):
                members = [(t, b) for t in group for b in range(len(items[t].blocks))]
                rows = {t: row for row, t in enumerate(group)}
                prefixes = [items[t].prefix for t in group]
                forwards.append((prefixes, members, [rows[t] for t, _ in members]))
        for prefixes, members, owners in forwards:
            blocks = [items[t].blocks[b] for t, b in members]
            hidden = self.engine_model.tree(
                TreeBatch(
                    prefixes=prefixes,
                    blocks=[block.ids for block in blocks],
                    owners=owners,
                    layout="packed" if packed else "rows",
                )
            ).hidden
            for (t, b), value in zip(members, self._read(blocks, hidden), strict=True):
                results[t][b] = value
        return results

    def combine(
        self, plans: list[RowTrees | None], items: list[DecoderTree], results: list[Any]
    ) -> list[RawRow | None]:
        outputs = {
            id(block): value
            for tree, values in zip(items, results, strict=True)
            for block, value in zip(tree.blocks, values, strict=True)
        }
        return [
            None if plan is None else self.layout.combine(plan, outputs)
            for plan in plans
        ]

    def _read(
        self, blocks: list[Block], hidden: torch.Tensor
    ) -> list[NDArray[np.float32] | None]:
        """Readouts of one forward's blocks: the candidate head over every question block at once."""
        device = hidden.device
        results: list[NDArray[np.float32] | None] = [None] * len(blocks)
        questions = [index for index, block in enumerate(blocks) if not block.is_span]
        with torch.inference_mode():
            if questions:
                width = max(len(blocks[index].ends) for index in questions)
                ends = torch.zeros((len(questions), width), dtype=torch.long)
                for row, index in enumerate(questions):
                    ends[row, : len(blocks[index].ends)] = torch.tensor(
                        blocks[index].ends
                    )
                rows = torch.tensor(questions, device=device)
                queries = torch.tensor(
                    [blocks[index].query for index in questions], device=device
                )
                scores = self.head(
                    hidden[rows[:, None], ends.to(device)], hidden[rows, queries]
                ).float()
                for row, index in enumerate(questions):
                    block = blocks[index]
                    values = scores[row, : len(block.ends)]
                    if block.question.type == "set":
                        values = values + self.set_bias
                    results[index] = values.cpu().numpy()
            for index, block in enumerate(blocks):
                if block.is_span and block.read:
                    results[index] = self._span(block, hidden[index])
        return results

    def _span(self, block: Block, rows: torch.Tensor) -> NDArray[np.float32]:
        """Word x label logits of one span block from its hidden rows (labels: mean over each label block)."""
        labels = torch.stack(
            [
                rows[start : end + 1].mean(0)
                for start, end in zip(block.starts, block.ends, strict=True)
            ]
        )
        words = rows[torch.as_tensor(block.words, device=rows.device)]
        logits: NDArray[np.float32] = (
            self.spans[block.head](words, labels).float().cpu().numpy()
        )
        return logits
