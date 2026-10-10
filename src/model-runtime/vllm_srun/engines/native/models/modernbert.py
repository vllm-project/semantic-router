"""ModernBERT encoder backbone (Vela 1.0 and 2.0 encoders, mmBERT, ModernBERT task models).

Every block reproduces the operation order of Transformers' ``ModernBertModel``
with SDPA attention, so the padded CPU path is bit-identical to it: pre-norm
layers alternating global and banded local attention (``|i - j| <=
local_attention // 2``), rotary positions per layer type (default or YaRN),
GeGLU MLPs and bias-free LayerNorms. Parameter names match the checkpoints'
backbone namespace (``model.`` inside task models).

Two layouts run the same layers (``Layout``):

- **padded** ``[B, T]`` rows with a key-padding mask, as Transformers runs them;
- **packed** ``[N]`` tokens of several sequences back to back: embeddings,
  norms, projections and MLPs run on the N real tokens only, and attention
  scatters each sequence into its own row of a ``[B, W]`` grid (one grid per
  group of rows of similar length) for one SDPA call per grid, then gathers
  the real rows back. Padded positions never reach the next layer, so no
  fully masked row can leak into real rows.

``encode`` returns the hidden states at any requested layer exits in one pass.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from ..encoder import pads_little

MODEL_TYPE = "modernbert"
FULL = "full_attention"
SLIDING = "sliding_attention"
BAND_BLOCK = 128
# Row width from which local layers attend in query blocks, by device type (Vela 307M,
# FP32): dense masked SDPA is faster below, on 16 EPYC cores and on MI325X.
BAND_FROM = {"cpu": 1024, "cuda": 2048}
# Query tokens one CPU SDPA call of local attention reads: the keys, values and
# mask it copies for them stay this small on rows of any length.
BAND_CALL_TOKENS = 4096
DEFAULT_THETA = {FULL: 160_000.0, SLIDING: 10_000.0}
# cos / sin per layer type for each grid width of a layout.
Rotary = dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
}


def layer_types(config: dict[str, Any]) -> list[str]:
    """Per layer ``full_attention`` or ``sliding_attention`` (every ``global_attn_every_n_layers``-th is global)."""
    if config.get("layer_types"):
        return list(config["layer_types"])
    every = config.get("global_attn_every_n_layers", 3)
    return [
        SLIDING if index % every else FULL
        for index in range(config["num_hidden_layers"])
    ]


def rope_parameters(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Rotary parameters per layer type, from either config generation.

    Older checkpoints carry ``global_rope_theta`` / ``local_rope_theta`` and one
    ``rope_scaling`` that applies to both layer types; newer ones carry
    ``rope_parameters`` keyed by layer type.
    """
    nested = config.get("rope_parameters") or {}
    params = {
        kind: dict(nested.get(kind) or {"rope_type": "default"})
        for kind in (FULL, SLIDING)
    }
    scaling = config.get("rope_scaling")
    for kind, legacy in ((FULL, "global_rope_theta"), (SLIDING, "local_rope_theta")):
        if scaling:
            params[kind].update(scaling)
        params[kind].setdefault("rope_type", params[kind].get("type", "default"))
        params[kind].setdefault("rope_theta", config.get(legacy, DEFAULT_THETA[kind]))
        if params[kind]["rope_type"] == "yarn":
            params[kind].setdefault(
                "original_max_position_embeddings", config["max_position_embeddings"]
            )
    return params


def _mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_frequencies(
    params: dict[str, Any], dim: int, max_positions: int
) -> tuple[torch.Tensor, float]:
    """YaRN inverse frequencies and attention factor, as Transformers computes them.

    The correction range is always truncated to whole dimensions: Transformers
    reads ``truncate`` from the per-type mapping's parent, where it never is.
    """
    base = params["rope_theta"]
    original = params["original_max_position_embeddings"]
    factor = params.get("factor") or max_positions / original
    attention_factor = params.get("attention_factor")
    if attention_factor is None:
        mscale, mscale_all_dim = params.get("mscale"), params.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_factor = float(
                _mscale(factor, mscale) / _mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = _mscale(factor)
    beta_fast = params.get("beta_fast") or 32
    beta_slow = params.get("beta_slow") or 1

    def correction(rotations: float) -> float:
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(beta_fast)), 0)
    high: float = min(math.ceil(correction(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    extrapolation_factor = 1 - ramp
    inv_freq = (
        interpolation * (1 - extrapolation_factor)
        + extrapolation * extrapolation_factor
    )
    return inv_freq, attention_factor


def rope_frequencies(
    params: dict[str, Any], dim: int, max_positions: int
) -> tuple[torch.Tensor, float]:
    """Inverse frequencies and the cos / sin scale of one layer type."""
    kind = params.get("rope_type", "default")
    if kind == "yarn":
        return yarn_frequencies(params, dim, max_positions)
    if kind != "default":
        raise ValueError(f"unsupported ModernBERT rope type {kind!r}")
    base = params["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq, 1.0


class ModernBertRotary(nn.Module):
    """cos / sin per layer type for positions ``0..T-1``; computed, never loaded."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = config["hidden_size"] // config["num_attention_heads"]
        params = rope_parameters(config)
        self.kinds = sorted(set(layer_types(config)))
        self.scaling: dict[str, float] = {}
        for kind in self.kinds:
            inv_freq, scaling = rope_frequencies(
                params[kind], dim, config["max_position_embeddings"]
            )
            self.register_buffer(f"{kind}_inv_freq", inv_freq, persistent=False)
            self.scaling[kind] = scaling

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, length: int
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{layer type: (cos, sin)}``, each ``[1, length, head_dim]`` in ``x.dtype``."""
        positions = torch.arange(length, device=x.device)[None, None, :].float()
        out = {}
        for kind in self.kinds:
            inv_freq = getattr(self, f"{kind}_inv_freq")
            inv_freq = inv_freq[None, :, None].to(dtype=torch.float, device=x.device)
            with torch.autocast(device_type=x.device.type, enabled=False):
                freqs = (inv_freq @ positions).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.scaling[kind]
                sin = emb.sin() * self.scaling[kind]
            out[kind] = (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))
        return out


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def attention_masks(
    key_valid: torch.Tensor | None,
    rows: int,
    width: int,
    window: int,
    device: torch.device | str,
    local: bool = True,
) -> dict[str, torch.Tensor | None]:
    """Boolean SDPA masks per layer type, None where Transformers passes none.

    Global layers mask padded keys; local layers also mask keys farther than
    ``window`` positions. Without padding, the global mask is dropped, and the
    local one too when the row is shorter than ``window``. ``local=False``
    skips the local mask (local layers attend in blocks, ``Band``).
    """
    padding = None if key_valid is None else key_valid[:, None, None, :]
    full = None if padding is None else padding.expand(rows, 1, width, width)
    if not local or (padding is None and width < window):
        return {FULL: full, SLIDING: None}
    position = torch.arange(width, device=device)
    band = ((position[:, None] - position[None, :]).abs() <= window)[None, None]
    sliding = band if padding is None else band & padding
    return {FULL: full, SLIDING: sliding.expand(rows, 1, width, width)}


@dataclass(frozen=True)
class Band:
    """Local attention in query blocks: block ``b`` reads keys ``[b * block - window, (b + 1) * block + window)``.

    ``mask`` (``[rows, 1, blocks, block, block + 2 * window]``) keeps each
    query's keys within ``window`` positions that are real tokens.
    """

    block: int
    window: int
    mask: torch.Tensor


def band(
    key_valid: torch.Tensor | None,
    rows: int,
    width: int,
    window: int,
    block: int,
    device: torch.device | str,
) -> Band:
    """The block mask of local attention over rows of ``width`` (built on the host)."""
    blocks = -(-width // block)
    span = block + 2 * window
    if key_valid is None:
        key_valid = torch.ones(rows, width, dtype=torch.bool)
    padded = F.pad(key_valid.cpu(), (window, blocks * block - width + window))
    keys = padded.unfold(1, span, block)
    query = torch.arange(block)[:, None]
    slot = torch.arange(span)[None, :]
    near = (slot >= query) & (slot <= query + 2 * window)
    mask = near[None, None, None] & keys[:, None, :, None, :]
    return Band(block, window, mask.to(device))


def banded_attention(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: Band,
    scale: float,
) -> torch.Tensor:
    """Local attention of ``[rows, heads, width, dim]`` inputs, O(width * (block + 2 window)).

    On the CPU, query blocks run ``BAND_CALL_TOKENS`` at a time: each SDPA call
    copies its blocks' keys, values and mask, so a call's copies stay that small
    on any row. Every block is its own SDPA problem, so the result is the same.
    """
    rows, heads, width, dim = query.shape
    count, block = blocks.mask.shape[2], blocks.block
    step = count if query.device.type != "cpu" else max(1, BAND_CALL_TOKENS // block)
    if step >= count:
        return _band_call(kernels, query, key, value, blocks, scale, 0, count)
    out = query.new_empty(rows, heads, width, dim)
    for first in range(0, count, step):
        last = min(count, first + step)
        out[:, :, first * block : last * block] = _band_call(
            kernels, query, key, value, blocks, scale, first, last
        )
    return out


def _band_call(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: Band,
    scale: float,
    first: int,
    last: int,
) -> torch.Tensor:
    """One SDPA call over query blocks ``[first, last)``: their rows of the output."""
    rows, heads, width, dim = query.shape
    span, block, window = blocks.mask.shape[-1], blocks.block, blocks.window
    start, stop, count = first * block, min(width, last * block), last - first
    q = F.pad(query[:, :, start:stop], (0, 0, 0, count * block - (stop - start)))
    # Block b reads keys [b * block - window, (b + 1) * block + window), zeros outside the row.
    low, high = start - window, last * block + window
    keys = (max(0, -low), max(0, high - width))
    k = F.pad(key[:, :, max(0, low) : min(width, high)], (0, 0, *keys))
    v = F.pad(value[:, :, max(0, low) : min(width, high)], (0, 0, *keys))
    # Heads and blocks fold into one batch axis: fused SDPA kernels take 4-D inputs only.
    folded = (rows, heads * count)
    mask = blocks.mask[:, :, first:last].expand(rows, heads, count, block, span)
    out: torch.Tensor = kernels("sdpa")(
        q.reshape(*folded, block, dim),
        k.unfold(2, span, block).transpose(-1, -2).reshape(*folded, span, dim),
        v.unfold(2, span, block).transpose(-1, -2).reshape(*folded, span, dim),
        mask.reshape(*folded, block, span),
        scale=scale,
        is_causal=False,
        enable_gqa=False,
    )
    return out.reshape(rows, heads, count * block, dim)[:, :, : stop - start]


@dataclass(frozen=True)
class Group:
    """Rows that attend together, as one ``[rows, width]`` grid.

    Packed, ``index`` holds each of the group's tokens' flat position in the
    grid (None for one unpadded row). ``masks`` are the SDPA masks per layer
    type; with ``band``, local layers attend in query blocks instead (long rows).
    """

    rows: int
    width: int
    masks: dict[str, torch.Tensor | None]
    packed: bool = False
    index: torch.Tensor | None = None
    band: Band | None = None

    def to_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token-major values (``[n, C]`` packed, ``[B, T, C]`` padded) as ``[rows, width, C]``."""
        if not self.packed:
            return tokens
        if self.index is None:
            return tokens[None]
        out = tokens.new_zeros(self.rows * self.width, tokens.shape[-1])
        out.index_copy_(0, self.index, tokens)
        return out.view(self.rows, self.width, -1)

    def to_tokens(self, rows: torch.Tensor) -> torch.Tensor:
        """The inverse of ``to_rows``: real rows only."""
        if not self.packed:
            return rows
        if self.index is None:
            return rows[0]
        return rows.reshape(self.rows * self.width, -1).index_select(0, self.index)


@dataclass(frozen=True)
class Layout:
    """Where each sequence's tokens sit while the layers run.

    A padded layout runs ``[rows, width]`` hidden states as they are, one
    group. A packed layout runs ``[N]`` tokens and attends in ``groups`` of
    rows of similar length, whose tokens lie back to back in the order the
    layers run: ``order`` (None: as given) gathers the input tokens into it,
    ``restore`` gathers the outputs back, ``sizes`` are the groups' tokens.
    """

    groups: tuple[Group, ...]
    order: torch.Tensor | None = None
    restore: torch.Tensor | None = None
    sizes: tuple[int, ...] = ()

    def split(self, tokens: torch.Tensor) -> list[tuple[Group, torch.Tensor]]:
        """Each group with its slice of the layers' token-major values."""
        if len(self.groups) == 1:
            return [(self.groups[0], tokens)]
        return list(
            zip(self.groups, torch.split(tokens, list(self.sizes)), strict=True)
        )


def padded_layout(
    attention_mask: torch.Tensor | None,
    rows: int,
    width: int,
    window: int,
    device: torch.device | str,
) -> Layout:
    """The layout of padded rows; reads ``attention_mask`` (any padding pattern) on the host."""
    key_valid = None
    if attention_mask is not None and not bool(attention_mask.all()):
        key_valid = attention_mask.to(device=device, dtype=torch.bool)
    masks = attention_masks(key_valid, rows, width, window, device)
    return Layout((Group(rows, width, masks),))


def length_groups(
    lengths: Sequence[int], width: int = 0, uniform: bool = False
) -> list[list[int]]:
    """Row indices by decreasing length, cut where a grid would pad too much (``pads_little``).

    ``uniform`` grids hold rows of one length only: no row is ever padded, so
    each attends exactly as it would alone.
    """
    order = sorted(range(len(lengths)), key=lambda row: -lengths[row])
    groups: list[list[int]] = []
    real = 0
    for row in order:
        if groups:
            longest = max(lengths[groups[-1][0]], width)
            grid = (len(groups[-1]) + 1) * longest
            if (
                lengths[row] == longest
                if uniform
                else pads_little(grid, real + lengths[row])
            ):
                groups[-1].append(row)
                real += lengths[row]
                continue
        groups.append([row])
        real = lengths[row]
    return groups


def packed_group(
    lengths: Sequence[int],
    window: int,
    device: torch.device | str,
    width: int,
    band_from: int,
    block: int,
) -> Group:
    """One grid over rows of ``lengths`` whose tokens lie back to back."""
    rows = len(lengths)
    width = max(*lengths, width)
    padded = any(length != width for length in lengths)
    index = key_valid = None
    if rows > 1 or padded:
        index = torch.cat(
            [
                torch.arange(length, dtype=torch.long) + row * width
                for row, length in enumerate(lengths)
            ]
        ).to(device)
    # Without padding a grid needs no key mask: each row attends as it would alone.
    if padded:
        key_valid = torch.arange(width)[None, :] < torch.tensor(list(lengths))[:, None]
    blocks = None
    if width >= band_from:
        blocks = band(key_valid, rows, width, window, block, device)
    device_valid = None if key_valid is None else key_valid.to(device)
    masks = attention_masks(
        device_valid, rows, width, window, device, local=blocks is None
    )
    return Group(rows, width, masks, packed=True, index=index, band=blocks)


def packed_layout(
    lengths: Sequence[int],
    window: int,
    device: torch.device | str,
    width: int | None = None,
    band_from: int | None = None,
    block: int = BAND_BLOCK,
    uniform: bool = False,
) -> Layout:
    """The layout of sequences of ``lengths`` packed back to back (grids at least ``width`` wide).

    Built from host-known lengths, so no device value is read back. Rows of
    very different lengths attend in separate grids (``length_groups``), so a
    long row does not widen every short one: attention costs rows x width^2.
    A single row without padding keeps the identity layout, which is the
    padded path. Rows of at least ``band_from`` tokens (``BAND_FROM`` for the
    device type) run local layers in query blocks. ``uniform`` grids hold rows
    of one length only, so every row attends exactly as it would alone.
    """
    if band_from is None:
        band_from = BAND_FROM.get(torch.device(device).type, BAND_FROM["cuda"])
    groups = length_groups(lengths, width or 0, uniform)
    if len(groups) == 1:
        group = packed_group(lengths, window, device, width or 0, band_from, block)
        return Layout((group,))
    starts = [0, *itertools.accumulate(lengths)]
    rows = [row for group in groups for row in group]
    order = torch.cat([torch.arange(starts[row], starts[row + 1]) for row in rows])
    restore = torch.empty_like(order)
    restore[order] = torch.arange(order.numel())
    return Layout(
        tuple(
            packed_group(
                [lengths[row] for row in group],
                window,
                device,
                width or 0,
                band_from,
                block,
            )
            for group in groups
        ),
        order=order.to(device),
        restore=restore.to(device),
        sizes=tuple(sum(lengths[row] for row in group) for group in groups),
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def layer_norm(config: dict[str, Any]) -> nn.LayerNorm:
    return nn.LayerNorm(
        config["hidden_size"],
        eps=config.get("norm_eps", 1e-5),
        bias=bool(config.get("norm_bias", False)),
    )


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = layer_norm(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self.norm(self.tok_embeddings(input_ids))
        return embeddings


class ModernBertAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config["hidden_size"]
        if hidden % config["num_attention_heads"]:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")
        self.head_dim = hidden // config["num_attention_heads"]
        self.scaling = self.head_dim**-0.5
        bias = bool(config.get("attention_bias", False))
        self.Wqkv = nn.Linear(hidden, 3 * hidden, bias=bias)
        self.Wo = nn.Linear(hidden, hidden, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: Rotary,
        kind: str,
        layout: Layout,
        kernels: KernelSet,
    ) -> torch.Tensor:
        """``rotary`` holds each grid width's cos / sin per layer type."""
        parts = [
            group.to_tokens(
                self.attend(
                    group.to_rows(projected),
                    rotary[group.width][kind],
                    kind,
                    group,
                    kernels,
                )
            )
            for group, projected in layout.split(self.Wqkv(hidden_states))
        ]
        out: torch.Tensor = self.Wo(parts[0] if len(parts) == 1 else torch.cat(parts))
        return out

    def attend(
        self,
        rows: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        kind: str,
        group: Group,
        kernels: KernelSet,
    ) -> torch.Tensor:
        """Attention over one grid's ``[rows, width, 3 * hidden]`` projections."""
        batch, width = rows.shape[:2]
        query, key, value = rows.view(batch, width, 3, -1, self.head_dim).unbind(dim=-3)
        query, key = kernels("rotary_half")(
            query.transpose(1, 2), key.transpose(1, 2), *rotary
        )
        blocks = group.band if kind == SLIDING else None
        if blocks is not None:
            output = banded_attention(
                kernels, query, key, value.transpose(1, 2), blocks, self.scaling
            )
        else:
            output = kernels("sdpa")(
                query,
                key,
                value.transpose(1, 2),
                group.masks[kind],
                scale=self.scaling,
                is_causal=False,
                enable_gqa=False,
            )
        return output.transpose(1, 2).contiguous().reshape(batch, width, -1)


class ModernBertMLP(nn.Module):
    """GeGLU: ``Wo(act(x Wi_in) * x Wi_gate)``."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        activation = config.get("hidden_activation", "gelu")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported ModernBERT activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        bias = bool(config.get("mlp_bias", False))
        self.Wi = nn.Linear(hidden, 2 * intermediate, bias=bias)
        self.Wo = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, hidden_states: torch.Tensor, kernels: KernelSet) -> torch.Tensor:
        out: torch.Tensor = self.Wo(kernels("geglu")(self.Wi(hidden_states), self.act))
        return out


class ModernBertLayer(nn.Module):
    def __init__(self, config: dict[str, Any], index: int, kind: str):
        super().__init__()
        self.kind = kind
        self.attn_norm = nn.Identity() if index == 0 else layer_norm(config)
        self.attn = ModernBertAttention(config)
        self.mlp_norm = layer_norm(config)
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: Rotary,
        layout: Layout,
        kernels: KernelSet,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), rotary, self.kind, layout, kernels
        )
        out: torch.Tensor = hidden_states + self.mlp(
            self.mlp_norm(hidden_states), kernels
        )
        return out


class ModernBertBackbone(nn.Module):
    """The ModernBERT encoder; ``encode`` returns final-normed hidden states at layer exits."""

    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        kinds = layer_types(config)
        if len(kinds) != config["num_hidden_layers"] or set(kinds) - {FULL, SLIDING}:
            raise ValueError(f"invalid ModernBERT layer types {kinds}")
        self.window = config.get("local_attention", 128) // 2
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertLayer(config, index, kind) for index, kind in enumerate(kinds)]
        )
        self.final_norm = layer_norm(config)
        self.rotary_emb = ModernBertRotary(config)
        self.kernels: KernelSet | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def packed(
        self,
        lengths: Sequence[int],
        device: torch.device | str,
        width: int | None = None,
        uniform: bool = False,
    ) -> Layout:
        """The layout of rows of ``lengths`` packed back to back (see ``packed_layout``)."""
        return packed_layout(lengths, self.window, device, width, uniform=uniform)

    def padded(
        self,
        attention_mask: torch.Tensor | None,
        rows: int,
        width: int,
        device: torch.device | str,
    ) -> Layout:
        """The layout of padded ``[rows, width]`` rows (see ``padded_layout``)."""
        return padded_layout(attention_mask, rows, width, self.window, device)

    def masked(
        self, valid: torch.Tensor, rows: int, width: int, device: torch.device | str
    ) -> Layout:
        """Padded rows whose masks come from a device-side key mask, never read back (graphs)."""
        masks = attention_masks(valid, rows, width, self.window, device)
        return Layout((Group(rows, width, masks),))

    def encode(
        self,
        input_ids: torch.Tensor,
        layout: Layout,
        exits: Sequence[int] = (),
        normalize_exits: bool = False,
    ) -> dict[int, torch.Tensor]:
        """Hidden states by exit in one pass; the last layer only when ``exits`` is empty.

        Exit 0 is the embedding output and exit ``k`` the residual stream after
        ``k`` layers. The last layer is final-normed (Transformers' output);
        intermediate exits are too only with ``normalize_exits``.
        ``input_ids`` is ``[N]`` for a packed layout, ``[rows, width]`` for a padded one.
        """
        assert self.kernels is not None, "bind kernels before running the backbone"
        exits = tuple(exits) or (self.num_layers,)
        if not all(0 <= layer <= self.num_layers for layer in exits):
            raise ValueError(f"layer exits must lie in 0..{self.num_layers}: {exits}")

        def exit_state(count: int, hidden: torch.Tensor) -> torch.Tensor:
            if count == self.num_layers or normalize_exits:
                normed: torch.Tensor = self.final_norm(hidden)
                return normed
            return hidden

        if layout.order is not None:
            input_ids = input_ids.index_select(0, layout.order)
        hidden_states = self.embeddings(input_ids)
        out: dict[int, torch.Tensor] = {}
        if 0 in exits:
            out[0] = exit_state(0, hidden_states)
        # One table per grid width: cos / sin of a longer table's prefix can differ
        # in the last bit (vectorized tails), and Transformers computes its own width.
        rotary = {
            width: self.rotary_emb(hidden_states, width)
            for width in {group.width for group in layout.groups}
        }
        for count, layer in enumerate(self.layers[: max(exits)], start=1):
            hidden_states = layer(hidden_states, rotary, layout, self.kernels)
            if count in exits:
                out[count] = exit_state(count, hidden_states)
        if layout.restore is not None:
            out = {
                layer: value.index_select(0, layout.restore)
                for layer, value in out.items()
            }
        return out

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Padded ``[B, T]`` rows to ``[B, T, hidden]`` final hidden states (Transformers' contract)."""
        rows, width = input_ids.shape
        layout = self.padded(attention_mask, rows, width, input_ids.device)
        return self.encode(input_ids, layout)[self.num_layers]
