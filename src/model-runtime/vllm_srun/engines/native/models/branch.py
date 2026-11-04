"""State-once branching: each request's state block encoded once, its question blocks branched from it.

A backbone with a state-once layout (``attention_mode`` ``prefix_bidir``,
``prefix_causal`` or ``causal``) gives no state token a view of a
question, so a state's hidden states are the same for every question of its
request. ``Branches`` packs a batch of such requests into one row: every state,
then every question block grouped by its state. Every layer then computes:

- Full attention. A state attends within itself, both ways for
  ``prefix_bidir`` and causally otherwise. A block attends to its state's keys
  followed by its own (``block_keys`` gathers them), both ways, or causally
  aligned at the end for ``causal``: one softmax over the keys the row state +
  block gives it. A variable-length kernel runs the states, and then the
  blocks, of the whole batch in one call each.
- Gated DeltaNet. A state runs from a zero state; a block runs from its state's
  final recurrent state (the snapshot), with its convolution window seeded by
  the state's last inputs (``gapped``).
- Projections, norms and MLPs once over the packed row.

Positions continue after the state (with media, after ``rope_index``'s last
state position), so each block's hidden states are those of the row state +
block under the same mask, computed alone; only the floating-point order
differs (the delta rule's chunks restart at the block, and GEMMs see other
shapes).
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from .tree import varlen

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ....accel.kernels import KernelSet
    from .qwen3_5 import GatedDeltaNet

CAUSAL = "causal"
PREFIX_BIDIR = "prefix_bidir"
PREFIX_CAUSAL = "prefix_causal"
LAYOUTS = (CAUSAL, PREFIX_BIDIR, PREFIX_CAUSAL)


def _flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """PyTorch's variable-length flash attention: ``[T, H, D]`` queries, keys with ``H / groups`` heads.

    Its causal mask is aligned at the end of each key segment, so a block's
    last query sees every key of its segment.
    """
    out: torch.Tensor = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, False, scale=scale
    )[0]
    return out


# The variable-length kernel, or None where segments run one at a time (``reference_attention``).
VARLEN: Callable[..., torch.Tensor] | None = (
    _flash if hasattr(torch.ops.aten, "_flash_attention_forward") else None
)


def _cumulative(lengths: Sequence[int]) -> list[int]:
    return [0, *itertools.accumulate(lengths)]


class Branches:
    """A batch of state-once requests packed in one row: every state, then every block grouped by its state.

    ``states`` and ``blocks`` hold token counts and ``owners`` each block's
    state, in state order (state 0's blocks first), and every state owns a
    block. The host lists describe the layout; the device tensors that index
    it are built once per batch, so no layer reads the device back.
    """

    is_branches = True

    def __init__(
        self,
        states: Sequence[int],
        blocks: Sequence[int],
        owners: Sequence[int],
        layout: str,
        device: torch.device,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"no state-once layout {layout!r}")
        if not states or not blocks or min(states) < 1 or min(blocks) < 1:
            raise ValueError("branches need states and blocks of at least one token")
        if len(owners) != len(blocks) or list(owners) != sorted(owners):
            raise ValueError("every block needs its state, blocks in state order")
        if set(owners) != set(range(len(states))):
            raise ValueError("every state needs at least one block")
        self.states = list(states)
        self.blocks = list(blocks)
        self.owners = list(owners)
        self.layout = layout
        self.state_bidirectional = layout == PREFIX_BIDIR
        self.block_bidirectional = layout != CAUSAL
        self.state_tokens = sum(self.states)
        self.attention_split = self.state_tokens
        self.continues = True
        self.total = self.state_tokens + sum(self.blocks)
        self.max_state = max(self.states)
        self.max_block = max(self.blocks)
        # Host boundaries: of the states, of the blocks (from the first block) and of
        # each block's keys (its state's, then its own; ``block_keys``).
        self.cu_states_host = _cumulative(self.states)
        self.cu_blocks_host = _cumulative(self.blocks)
        keys = [
            self.states[o] + n for o, n in zip(self.owners, self.blocks, strict=True)
        ]
        self.cu_keys_host = _cumulative(keys)
        self.max_keys = max(keys)
        self.device = device
        # int32 boundaries for attention, int64 (and host copies) for the gated-delta kernel.
        self.cu_states = torch.tensor(self.cu_states_host, dtype=torch.int32).to(device)
        self.cu_blocks = torch.tensor(self.cu_blocks_host, dtype=torch.int32).to(device)
        self.cu_keys = torch.tensor(self.cu_keys_host, dtype=torch.int32).to(device)
        index: list[int] = []
        for owner, start, length in zip(
            self.owners, self.cu_blocks_host[:-1], self.blocks, strict=True
        ):
            index += range(self.cu_states_host[owner], self.cu_states_host[owner + 1])
            index += range(
                self.state_tokens + start, self.state_tokens + start + length
            )
        self.block_keys = torch.tensor(index, dtype=torch.long).to(device)
        self.rule_states = (
            self.cu_states.long(),
            torch.tensor(self.cu_states_host, dtype=torch.long),
        )
        self.rule_blocks = (
            self.cu_blocks.long(),
            torch.tensor(self.cu_blocks_host, dtype=torch.long),
        )
        self.owner = torch.tensor(self.owners, dtype=torch.long).to(device)
        self.last = torch.tensor(
            [self.state_tokens + end - 1 for end in self.cu_blocks_host[1:]],
            dtype=torch.long,
        ).to(device)
        self._gaps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _gap_index(self, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Indices into ``[zero row, packed rows]`` laying out every segment after ``window - 1`` lead rows.

        A state's lead rows are zeros (the history a row's first token sees);
        a block's are its state's last ``window - 1`` rows, zeros where the
        state is shorter. Returns the gapped layout's source rows and where
        the packed rows sit in it.
        """
        cached = self._gaps.get(window)
        if cached is not None:
            return cached
        lead = window - 1
        source: list[int] = []
        real: list[int] = []
        for start, length in zip(self.cu_states_host, self.states, strict=False):
            source += [0] * lead
            real += range(len(source), len(source) + length)
            source += range(start + 1, start + length + 1)
        for owner, start, length in zip(
            self.owners, self.cu_blocks_host, self.blocks, strict=False
        ):
            first, end = self.cu_states_host[owner], self.cu_states_host[owner + 1]
            source += [t + 1 if t >= first else 0 for t in range(end - lead, end)]
            real += range(len(source), len(source) + length)
            begin = self.state_tokens + start
            source += range(begin + 1, begin + length + 1)
        cached = (
            torch.tensor(source, dtype=torch.long).to(self.device),
            torch.tensor(real, dtype=torch.long).to(self.device),
        )
        self._gaps[window] = cached
        return cached

    def gapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """``[1, T, C]`` -> ``[1, T + segments * (window - 1), C]`` with each segment's lead rows in front."""
        source, _ = self._gap_index(window)
        padded = torch.cat([x.new_zeros((1, 1, *x.shape[2:])), x], dim=1)
        return padded[:, source]

    def ungapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """The packed rows of a gapped ``[1, T', ...]``: ``[1, T, ...]``."""
        _, real = self._gap_index(window)
        return x[:, real]


# FLA's chunk length: a region's chunk index has at most rows / CHUNK + segments entries.
CHUNK = 64


# A region of a static layout for a segmented convolution: its first row, its
# segments' offsets and FLA chunk index, and each segment's lead source rows.
Region = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]


def bucket_rows(tokens: int) -> int:
    """The rows of a region that holds ``tokens``: rounded up to 1 / 64 of their power of two, 64 at least."""
    step = max(CHUNK, (1 << max(0, (tokens - 1).bit_length())) // 64)
    return max(step, -(-tokens // step) * step)


def bucket_segments(segments: int) -> int:
    """A region's segment count: its own, a trash segment and a filler at least, as a power of two."""
    return 1 << (segments + 1).bit_length()


def bucket_chunks(count: int) -> int:
    """A region's chunk-index entries for ``count`` chunks: a multiple of 4 (the rest run a zero-length filler)."""
    return max(4, -(-count // 4) * 4)


def bucket_span(length: int) -> int:
    """A bound on a layout's segment lengths (what varlen attention sizes its grid by): a power of two, 64 at least."""
    return max(CHUNK, 1 << max(0, (length - 1).bit_length()))


@dataclass(frozen=True)
class BranchShape:
    """What one captured graph runs: a branched forward padded to its bucket (``StaticBranches``).

    The first region holds every state, then a trash segment up to
    ``state_rows`` rows, then zero-length fillers up to ``state_segments``; the
    blocks region likewise, each block continuing its state's recurrence.
    ``joined`` (every state with one block) lays each state and its block out
    as one row instead: one recurrence per row and no blocks region.
    Attention runs the states alone, then every block over gathered keys,
    ``queries`` segments of ``keys`` keys; ``joined`` and ``merged`` (states
    and blocks under one mask) run every state and block in that one call.
    ``spans`` bound the segment lengths of those calls. ``gaps`` keeps the gap
    layout a convolution without segments reads.
    Every index the forward reads has a size fixed by the shape, so a batch of
    the shape only rewrites their values.
    """

    state_rows: int
    block_rows: int
    state_segments: int
    block_segments: int
    queries: int
    keys: int
    spans: tuple[int, int, int]
    chunks: tuple[int, int]
    window: int
    joined: bool = False
    gaps: bool = True
    merged: bool = False

    @property
    def rows(self) -> int:
        return self.state_rows + self.block_rows

    @property
    def readouts(self) -> int:
        return self.state_segments if self.joined else self.block_segments

    @property
    def gapped_rows(self) -> int:
        return self.rows + (self.state_segments + self.block_segments) * (
            self.window - 1
        )

    def int32_sizes(self) -> dict[str, int]:
        return {
            "cu_states": self.state_segments + 1,
            "cu_blocks": self.queries + 1,
            "cu_keys": self.queries + 1,
        }

    def int64_sizes(self) -> dict[str, int]:
        states, blocks = 2 * self.chunks[0], 2 * self.chunks[1]
        both = self.state_segments + self.block_segments if self.block_rows else 0
        return {
            "ids": self.rows,
            "positions": self.rows,
            "rule_states": self.state_segments + 1,
            "rule_blocks": self.block_segments + 1,
            "chunks_states": states,
            "chunks_blocks": blocks,
            "rule_all": both + 1 if both else 0,
            "chunks_all": states + blocks if both else 0,
            "block_keys": self.keys,
            "owner": self.block_segments,
            "last": self.readouts,
            "leads": both * (self.window - 1),
            "gap_source": self.gapped_rows if self.gaps else 0,
            "gap_real": self.rows if self.gaps else 0,
        }


def _chunk_index(lengths: Sequence[int], entries: int, filler: int) -> list[int]:
    """FLA's chunk index of segments of ``lengths`` (each chunk's segment and its index in it, flattened),
    padded with chunks of the zero-length segment ``filler`` up to ``entries`` chunks.
    """
    index: list[int] = []
    for segment, length in enumerate(lengths):
        for chunk in range(-(-length // CHUNK)):
            index += (segment, chunk)
    return index + [filler, 0] * (entries - len(index) // 2)


def _padded(lengths: list[int], rows: int, segments: int) -> list[int]:
    """A region's segment lengths: ``lengths``, a trash segment up to ``rows``, fillers up to ``segments``."""
    return [*lengths, rows - sum(lengths)] + [0] * (segments - len(lengths) - 1)


def static_layout(
    states: Sequence[Sequence[int]],
    blocks: Sequence[Sequence[int]],
    owners: Sequence[int],
    window: int,
    *,
    joined: bool = False,
    gaps: bool = True,
    exact: bool = False,
    merged: bool = False,
) -> tuple[BranchShape, dict[str, Any]]:
    """The bucketed layout of a branched batch of token ``states`` and ``blocks``: its shape and every
    index's values (host lists; the token ids, positions and keys as int64 arrays). ``exact`` (an eager
    forward, no graph to share) pads nothing: empty trash segments and one filler.

    A trash segment runs alone (its queries see only its keys; its recurrence
    starts from zero) and nothing reads it, and the zero-length fillers run
    nothing; each real state and block computes what ``Branches`` computes.
    ``joined`` needs every state to own exactly one block; ``merged`` (states
    and blocks attend under one mask) puts the states' segments, each over its
    own keys, ahead of the blocks' in their attention call. ``last`` lists the
    blocks' last rows in the order of ``blocks``.
    """
    n_states, n_blocks = len(states), len(blocks)
    state_lengths, block_lengths = [len(t) for t in states], [len(t) for t in blocks]
    rows_of = (lambda tokens: tokens) if exact else bucket_rows
    segments_of = (lambda count: count + 2) if exact else bucket_segments
    span_of = (lambda length: max(1, length)) if exact else bucket_span
    chunks_of = (lambda count: count) if exact else bucket_chunks
    if joined:
        if list(owners) != list(range(n_states)):
            raise ValueError(
                "a joined layout needs one block per state, in state order"
            )
        first = [s + b for s, b in zip(state_lengths, block_lengths, strict=True)]
        second: list[int] = []
        state_rows, block_rows = rows_of(sum(first)), 0
        segments = (segments_of(n_states), 0)
    else:
        first, second = state_lengths, block_lengths
        state_rows, block_rows = rows_of(sum(first)), rows_of(sum(second))
        segments = (segments_of(n_states), segments_of(n_blocks))
    first_padded = _padded(first, state_rows, segments[0])
    second_padded = _padded(second, block_rows, segments[1]) if block_rows else []
    chunk_counts = [
        sum(-(-length // CHUNK) for length in padded)
        for padded in (first_padded, second_padded)
    ]
    cu_first, cu_second = _cumulative(first_padded), _cumulative(second_padded)
    # The token ids, positions and attention keys (row indices), the large indices, as arrays.
    ids = np.zeros(state_rows + block_rows, dtype=np.int64)
    positions = np.zeros(state_rows + block_rows, dtype=np.int64)
    keys: list[npt.NDArray[np.int64]] = []
    key_lengths: list[int] = []
    query_lengths: list[int] = []
    last: list[int] = []
    if joined:
        for start, state, block in zip(cu_first, states, blocks, strict=False):
            middle, end = start + len(state), start + len(state) + len(block)
            ids[start:middle], ids[middle:end] = state, block
            positions[start:end] = np.arange(end - start)
            keys += [np.arange(start, middle), np.arange(start, end)]
            key_lengths += (len(state), end - start)
            query_lengths += (len(state), len(block))
            last.append(end - 1)
        trash = (cu_first[n_states], state_rows)
        queries = segments_of(2 * n_states)
    else:
        for start, state in zip(cu_first, states, strict=False):
            ids[start : start + len(state)] = state
            positions[start : start + len(state)] = np.arange(len(state))
        if merged:
            keys.append(np.arange(state_rows))
            key_lengths += first_padded
            query_lengths += first_padded
        for owner, start, block in zip(owners, cu_second, blocks, strict=False):
            begin = state_rows + start
            ids[begin : begin + len(block)] = block
            length = state_lengths[owner]
            positions[begin : begin + len(block)] = np.arange(
                length, length + len(block)
            )
            keys += [
                np.arange(cu_first[owner], cu_first[owner + 1]),
                np.arange(begin, begin + len(block)),
            ]
            key_lengths.append(length + len(block))
            query_lengths.append(len(block))
            last.append(begin + len(block) - 1)
        trash = (state_rows + cu_second[n_blocks], state_rows + block_rows)
        queries = segments[0] + segments[1] if merged else segments[1]
    keys.append(np.arange(*trash))
    key_lengths.append(trash[1] - trash[0])
    query_lengths.append(trash[1] - trash[0])
    key_lengths += [0] * (queries - len(key_lengths))
    query_lengths += [0] * (queries - len(query_lengths))
    key_rows = sum(len(part) for part in keys)
    shape = BranchShape(
        state_rows,
        block_rows,
        *segments,
        queries,
        rows_of(key_rows),
        (
            0 if joined or merged else span_of(max(first_padded)),
            span_of(max(query_lengths)),
            span_of(max(key_lengths)),
        ),
        (chunks_of(chunk_counts[0]), chunks_of(chunk_counts[1]) if block_rows else 0),
        window,
        joined,
        gaps,
        merged and not joined,
    )
    keys.append(np.zeros(shape.keys - key_rows, dtype=np.int64))
    lead = window - 1
    leads: list[list[int]] = [[-1] * lead] * segments[0] if block_rows else []
    for index in range(segments[1]):
        if index < n_blocks:
            begin, end = cu_first[owners[index]], cu_first[owners[index] + 1]
            leads.append([t if t >= begin else -1 for t in range(end - lead, end)])
        else:
            leads.append([-1] * lead)
    source: list[int] = []
    real: list[int] = []
    if gaps:
        # The gap layout: every segment after its lead rows, indices into [zero row, packed rows].
        for start, length in zip(cu_first, first_padded, strict=False):
            source += [0] * lead
            real += range(len(source), len(source) + length)
            source += range(start + 1, start + length + 1)
        for start, length, row_leads in zip(
            cu_second, second_padded, leads[segments[0] :], strict=False
        ):
            source += [t + 1 for t in row_leads]
            real += range(len(source), len(source) + length)
            source += range(state_rows + start + 1, state_rows + start + length + 1)
    trash_state = n_states  # the trash state's final state starts the blocks region's trash and fillers
    chunks_states = _chunk_index(first_padded, shape.chunks[0], n_states + 1)
    chunks_blocks = _chunk_index(second_padded, shape.chunks[1], n_blocks + 1)
    values = {
        "cu_states": cu_first,
        "cu_blocks": _cumulative(query_lengths),
        "cu_keys": _cumulative(key_lengths),
        "ids": ids,
        "positions": positions,
        "rule_states": cu_first,
        "rule_blocks": cu_second,
        "chunks_states": chunks_states,
        "chunks_blocks": chunks_blocks,
        # Both regions as one sequence of segments, the blocks' numbered after the states'.
        "rule_all": (
            cu_first + [state_rows + end for end in cu_second[1:]] if block_rows else []
        ),
        "chunks_all": (
            chunks_states
            + [
                v + segments[0] if i % 2 == 0 else v
                for i, v in enumerate(chunks_blocks)
            ]
            if block_rows
            else []
        ),
        "block_keys": np.concatenate(keys),
        "owner": (
            [*owners, *[trash_state] * (segments[1] - n_blocks)] if block_rows else []
        ),
        "last": last + [0] * (shape.readouts - len(last)),
        "leads": [t for row in leads for t in row],
        "gap_source": source,
        "gap_real": real,
    }
    return shape, values


def pack_layout(
    shape: BranchShape, values: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The layout's indices as one int32 and one int64 host tensor, in ``BranchShape``'s order."""
    flat32 = np.concatenate(
        [np.asarray(values[name], dtype=np.int32) for name in shape.int32_sizes()]
    )
    flat64 = np.concatenate(
        [np.asarray(values[name], dtype=np.int64) for name in shape.int64_sizes()]
    )
    return torch.from_numpy(flat32), torch.from_numpy(flat64)


class StaticBranches:
    """A ``Branches`` layout of a ``BranchShape``: views of two flat index tensors, sizes fixed by the shape.

    The graph of a shape reads these views, so a batch of the shape runs by
    rewriting the tensors' values. ``host`` (the layout's host lists) serves
    the per-segment reference path; graphs run without it.
    """

    is_branches = True

    def __init__(
        self,
        shape: BranchShape,
        layout: str,
        ints32: torch.Tensor,
        ints64: torch.Tensor,
        host: dict[str, list[int]] | None = None,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"no state-once layout {layout!r}")
        self.shape = shape
        self.layout = layout
        self.state_bidirectional = layout == PREFIX_BIDIR
        self.block_bidirectional = layout != CAUSAL
        one_call = shape.joined or shape.merged
        if one_call and self.state_bidirectional != self.block_bidirectional:
            raise ValueError(
                "a joined or merged layout runs states and blocks under one mask"
            )
        self.state_tokens = shape.state_rows
        self.attention_split = 0 if one_call else shape.state_rows
        self.continues = shape.block_rows > 0
        self.total = shape.rows
        self.max_state, self.max_block, self.max_keys = shape.spans
        views: dict[str, torch.Tensor] = {}
        for flat, sizes in (
            (ints32, shape.int32_sizes()),
            (ints64, shape.int64_sizes()),
        ):
            offset = 0
            for name, size in sizes.items():
                views[name] = flat[offset : offset + size]
                offset += size
        self.cu_states, self.cu_blocks, self.cu_keys = (
            views["cu_states"],
            views["cu_blocks"],
            views["cu_keys"],
        )
        self.ids, self.positions = views["ids"], views["positions"]
        self.rule_states = (views["rule_states"], views["chunks_states"].view(-1, 2))
        self.rule_blocks = (views["rule_blocks"], views["chunks_blocks"].view(-1, 2))
        self.rule_all = (views["rule_all"], views["chunks_all"].view(-1, 2))
        self.block_keys, self.owner, self.last = (
            views["block_keys"],
            views["owner"],
            views["last"],
        )
        self.gap_source, self.gap_real = views["gap_source"], views["gap_real"]
        self.leads = views["leads"].view(-1, shape.window - 1)
        self.host = host
        self.readout: torch.Tensor | None = None
        self.cu_states_host = host["cu_states"] if host else None
        self.cu_blocks_host = host["cu_blocks"] if host else None
        self.cu_keys_host = host["cu_keys"] if host else None
        self.owners = host["owner"] if host else None

    def reading(self, rows: torch.Tensor) -> StaticBranches:
        """This layout for a last layer whose output only ``rows`` are read: its residual stream, output
        projections and MLP run on those rows (``[1, len(rows), hidden]``)."""
        view = copy.copy(self)
        view.readout = rows
        return view

    def gapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """``Branches.gapped`` over the shape's segments (fillers and trash included)."""
        if window != self.shape.window or not self.shape.gaps:
            raise ValueError("the layout has no gaps for this convolution window")
        padded = torch.cat([x.new_zeros((1, 1, *x.shape[2:])), x], dim=1)
        return padded[:, self.gap_source]

    def ungapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        return x[:, self.gap_real]

    def regions(self, window: int) -> tuple[Region, ...]:
        """The convolution's segments without the gap layout: from row 0, every segment's offsets and chunk
        index, and the rows a block's first taps read in its state (-1: zeros; none without blocks).
        """
        if window != self.shape.window:
            raise ValueError("the layout's leads serve another convolution window")
        if not self.continues:
            return ((0, *self.rule_states, None),)
        return ((0, *self.rule_all, self.leads),)


def is_static(branches: object) -> TypeIs[StaticBranches]:
    return isinstance(branches, StaticBranches)


def delta_kernel(
    kernels: KernelSet, branches: Branches | StaticBranches
) -> Callable[..., Any]:
    """The gated-delta kernel a layout runs: a static one takes FLA's chunk-indexed entry point where present."""
    if is_static(branches) and kernels.has("chunk_gated_delta_rule_static"):
        return kernels("chunk_gated_delta_rule_static")
    return kernels("chunk_gated_delta_rule")


def is_branches(mask: object) -> TypeIs[Branches | StaticBranches]:
    """Whether a layer's mask is a ``Branches`` layout, which runs the branched forward."""
    return bool(getattr(mask, "is_branches", False))


def _segments(bounds: Sequence[int]) -> list[tuple[int, int]]:
    return list(itertools.pairwise(bounds))


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q: Sequence[int],
    cu_k: Sequence[int],
    *,
    causal: bool,
    scale: float,
    groups: int,
) -> torch.Tensor:
    """Each segment alone in FP32; ``[Tq, H, D]`` outputs in the query dtype.

    Segment ``i`` of the queries attends to segment ``i`` of the keys
    (``[Tk, H / groups, D]``); a ``causal`` mask is aligned at the end of the
    key segment, as the kernel's is.
    """
    outputs = []
    for (qs, qe), (ks, ke) in zip(_segments(cu_q), _segments(cu_k), strict=True):
        q = query[qs:qe].transpose(0, 1).float()
        k = key[ks:ke].transpose(0, 1).float()
        v = value[ks:ke].transpose(0, 1).float()
        if groups > 1:
            k = k.repeat_interleave(groups, dim=0)
            v = v.repeat_interleave(groups, dim=0)
        scores = (q @ k.transpose(-1, -2)) * scale
        if causal:
            keep = torch.ones(
                qe - qs, ke - ks, dtype=torch.bool, device=scores.device
            ).tril((ke - ks) - (qe - qs))
            scores = scores.masked_fill(~keep, -float("inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
    return torch.cat(outputs).to(query.dtype)


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q: tuple[torch.Tensor, Sequence[int] | None, int],
    cu_k: tuple[torch.Tensor, Sequence[int] | None, int],
    *,
    causal: bool,
    scale: float,
    groups: int,
) -> torch.Tensor:
    """Variable-length attention: one kernel call on a GPU, else ``reference_attention``.

    ``cu_q`` / ``cu_k`` are (device boundaries, host boundaries, longest segment).
    """
    if (
        VARLEN is not None
        and query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
    ):
        return VARLEN(
            query, key, value, cu_q[0], cu_k[0], cu_q[2], cu_k[2], causal, scale
        )
    if cu_q[1] is None or cu_k[1] is None:
        raise ValueError("per-segment attention needs the layout's host boundaries")
    return reference_attention(
        query, key, value, cu_q[1], cu_k[1], causal=causal, scale=scale, groups=groups
    )


def branch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    branches: Branches | StaticBranches,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Attention over packed branches (``[1, H, T, D]`` inputs); returns ``[1, T, H, D]`` like ``common.attention``.

    Each block reads its state's keys and values followed by its own
    (``Branches.block_keys``), so every query takes one softmax over exactly
    the keys its row would give it.
    """
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    q, k, v = (x[0].transpose(0, 1).to(dtype).contiguous() for x in (query, key, value))
    b = branches
    split = b.attention_split
    outputs = []
    if split:
        states = (b.cu_states, b.cu_states_host, b.max_state)
        outputs.append(_attend(
            q[:split], k[:split], v[:split], states, states,
            causal=not b.state_bidirectional, scale=scaling, groups=groups,
        ))  # fmt: skip
    outputs.append(_attend(
        q[split:], k[b.block_keys], v[b.block_keys],
        (b.cu_blocks, b.cu_blocks_host, b.max_block),
        (b.cu_keys, b.cu_keys_host, b.max_keys),
        causal=not b.block_bidirectional, scale=scaling, groups=groups,
    ))  # fmt: skip
    return (torch.cat(outputs) if len(outputs) > 1 else outputs[0])[None]


def branch_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    branches: Branches | StaticBranches,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """The chunked gated delta rule over packed branches (``[1, T, ...]`` inputs).

    Every state runs from a zero state and keeps its final recurrent state;
    every block runs from its state's. A ``StaticBranches`` layout runs FLA
    with its own chunk index (``rule`` is ``chunk_gated_delta_rule_static``)
    where FLA serves the device, else per segment, zero-length ones skipped.
    """
    b = branches
    split = b.state_tokens
    if is_static(b):
        return _static_rule(rule, b, q, k, v, g, beta)

    def run(
        start: int,
        end: int,
        initial: torch.Tensor | None,
        final: bool,
        cu: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        extra: dict[str, Any] = {}
        if cu is not None:
            extra = {"cu_seqlens": cu[0], "cu_seqlens_cpu": cu[1]}
        return rule(
            q[:, start:end], k[:, start:end], v[:, start:end], g=g[:, start:end], beta=beta[:, start:end],
            initial_state=initial, output_final_state=final, use_qk_l2norm_in_kernel=True, **extra,
        )  # fmt: skip

    if varlen(rule):
        out_states, finals = run(0, split, None, True, b.rule_states)
        assert finals is not None
        out_blocks, _ = run(
            split, b.total, finals[b.owner].contiguous(), False, b.rule_blocks
        )
        return torch.cat([out_states, out_blocks], dim=1)
    outputs: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    for start, end in _segments(b.cu_states_host):
        out, final = run(start, end, None, True)
        assert final is not None
        outputs.append(out)
        snapshots.append(final)
    for owner, (start, end) in zip(b.owners, _segments(b.cu_blocks_host), strict=True):
        out, _ = run(split + start, split + end, snapshots[owner], False)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def _static_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    b: StaticBranches,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    split = b.state_tokens
    regions = getattr(rule, "regions", None)
    if b.continues and regions is not None:
        out: torch.Tensor = regions(
            q, k, v, g, beta, split, b.rule_all, b.rule_states, b.rule_blocks, b.owner
        )
        return out
    if getattr(rule, "chunk_indexed", False):
        out_states, finals = rule(
            q[:, :split], k[:, :split], v[:, :split], g[:, :split], beta[:, :split],
            None, b.continues, *b.rule_states,
        )  # fmt: skip
        if not b.continues:
            return out_states
        assert finals is not None
        out_blocks, _ = rule(
            q[:, split:], k[:, split:], v[:, split:], g[:, split:], beta[:, split:],
            finals[b.owner].contiguous(), False, *b.rule_blocks,
        )  # fmt: skip
        return torch.cat([out_states, out_blocks], dim=1)
    assert b.host is not None, "the per-segment path needs the layout's host lists"
    outputs: list[torch.Tensor] = []
    snapshots: list[torch.Tensor | None] = []

    def run(
        start: int, end: int, initial: torch.Tensor | None, final: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if end == start:
            return v.new_zeros((1, 0, *v.shape[2:])), None
        return rule(
            q[:, start:end], k[:, start:end], v[:, start:end], g=g[:, start:end], beta=beta[:, start:end],
            initial_state=initial, output_final_state=final, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip

    for start, end in _segments(b.host["rule_states"]):
        out, final_state = run(start, end, None, b.continues)
        outputs.append(out)
        snapshots.append(final_state)
    for owner, (start, end) in zip(
        b.host["owner"], _segments(b.host["rule_blocks"]), strict=True
    ):
        out, _ = run(split + start, split + end, snapshots[owner], False)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def branch_gated_delta(
    m: GatedDeltaNet,
    hidden_states: torch.Tensor,
    branches: Branches | StaticBranches,
    kernels: KernelSet,
) -> torch.Tensor:
    """``GatedDeltaNet.forward`` over packed branches (``[1, T, hidden]``), eager ops.

    The convolution reads the gapped rows (``Branches.gapped``): its input
    projection has no bias, so a state's zero lead rows stay zero through it.
    """
    _, length, _ = hidden_states.shape
    window = m.conv_kernel_size
    mixed = m.in_proj_qkv(hidden_states)
    z = m.in_proj_z(hidden_states).reshape(1, length, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)
    convolved = kernels("causal_conv1d")(
        branches.gapped(mixed, window).transpose(1, 2),
        m.conv1d.weight.squeeze(1),
        m.conv1d.bias,
        activation=m.activation,
    )
    mixed = branches.ungapped(convolved.transpose(1, 2), window)
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(1, length, -1, m.head_k_dim)
    key = key.reshape(1, length, -1, m.head_k_dim)
    value = value.reshape(1, length, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    core = branch_rule(
        delta_kernel(kernels, branches),
        branches,
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g,
        beta,
    )
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    out: torch.Tensor = m.out_proj(core.reshape(1, length, -1))
    return out


def state_once_mask(
    attention_mask: torch.Tensor, states: Sequence[int], layout: str
) -> torch.Tensor:
    """The full-attention mask ``[B, 1, L, L]`` of left-padded rows under a state-once layout.

    Row ``b`` holds its tokens at the end and the first ``states[b]`` of them
    are its state block. A padding position attends to itself only, so no
    query is left without a key; the gated-delta layers zero padding before it
    reaches a token.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"no state-once layout {layout!r}")
    device = attention_mask.device
    length = attention_mask.shape[1]
    real = attention_mask.bool()
    start = length - real.sum(1)
    end = start + torch.tensor(list(states), device=device)
    i = torch.arange(length, device=device)[None, :, None]
    j = torch.arange(length, device=device)[None, None, :]
    in_state = i < end[:, None, None]
    if layout == CAUSAL:
        allowed = j <= i
    elif layout == PREFIX_BIDIR:
        allowed = torch.where(in_state, j < end[:, None, None], True)
    else:
        allowed = torch.where(in_state, j <= i, True)
    allowed = (allowed & real[:, None, :] & real[:, :, None]) | (
        (i == j) & ~real[:, :, None]
    )
    return allowed[:, None]
