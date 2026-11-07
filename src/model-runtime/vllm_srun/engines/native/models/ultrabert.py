"""UltraBERT encoder backbone (Vela 3.0 vela3-fast).

Every block reproduces the operation order of the UltraBERT modeling code the
packages were evaluated with (Transformers SDPA attention): pre-RMSNorm
layers, separate query / key / value / output projections, SwiGLU MLPs and a
final RMSNorm. Each layer attends globally or within ``attention_windows[i]``
positions (``|i - j| <= window``), with rotary positions of that layer's own
``rope_theta[i]``. Parameter names match the checkpoints' encoder namespace
(``encoder.`` inside the packages).

The padded layout passes the additive float masks the modeling code builds
(0 where a key is read, the dtype's minimum elsewhere) to every layer, so the
padded CPU path selects the same SDPA backend and is bit-identical to it. On
the CPU those masks never take ``width x width`` memory: global layers read a
key mask and local layers attend ``LOCAL_QUERIES`` queries at a time over
every key, which leaves each query's SDPA reduction unchanged. The
packed layout runs the ModernBERT layout (``modernbert.packed_layout``):
boolean masks per grid, local layers in query blocks on long rows; it is the
approximate path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .modernbert import (
    FULL,
    SLIDING,
    Group,
    Layout,
    attention_masks,
    banded_attention,
    packed_layout,
)

MODEL_TYPE = "ultrabert"
# Queries per CPU SDPA call of a local layer on padded rows: each call builds
# its own ``[rows, 1, LOCAL_QUERIES, width]`` slice of the dense mask.
LOCAL_QUERIES = 1024
# The CPU SDPA kernel blocks queries one way from this many queries on and
# another way below, which changes the bits, so no chunk of a longer row is
# shorter: a short tail joins the chunk before it.
SDPA_QUERY_BLOCKS = 768
# cos / sin per rope theta for each grid width of a layout.
Rotary = dict[int, dict[float, tuple[torch.Tensor, torch.Tensor]]]


def windows(config: dict[str, Any]) -> list[int | None]:
    """Per layer the local attention window, None for global layers."""
    found = list(config["attention_windows"])
    if len(found) != config["num_hidden_layers"]:
        raise ValueError("attention_windows must name every layer")
    return [None if window is None else int(window) for window in found]


def thetas(config: dict[str, Any]) -> list[float]:
    """Per layer the rotary base."""
    found = config["rope_theta"]
    if not isinstance(found, list):
        found = [found] * config["num_hidden_layers"]
    if len(found) != config["num_hidden_layers"]:
        raise ValueError("rope_theta must name every layer")
    return [float(theta) for theta in found]


def local_window(config: dict[str, Any]) -> int:
    """The one local window the layers use (UltraBERT has one)."""
    local = {window for window in windows(config) if window is not None}
    if len(local) > 1:
        raise ValueError(f"UltraBERT layers use one local window, got {sorted(local)}")
    return local.pop() if local else 0


def additive_masks(
    key_valid: torch.Tensor,
    rows: int,
    width: int,
    window: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """The modeling code's float masks per layer kind, ``[rows, 1, width, width]``.

    Every query reads the keys that are real tokens and, on local layers,
    within ``window`` positions; the others get the dtype's minimum.
    """
    device = key_valid.device
    padding = key_valid.bool()[:, None, :]
    full = torch.ones(1, width, width, dtype=torch.bool, device=device)
    masks = {FULL: full & padding}
    if window:
        masks[SLIDING] = full.tril(window).triu(-window) & padding
    minimum = torch.finfo(dtype).min
    return {
        kind: torch.where(mask, 0.0, minimum)
        .to(dtype)
        .unsqueeze(1)
        .expand(rows, 1, width, width)
        for kind, mask in masks.items()
    }


def local_attention(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Local attention of padded ``[rows, heads, width, dim]`` rows in query chunks.

    Every chunk reads all keys with its rows of the dense additive mask, built
    from ``key_mask`` (``[rows, 1, 1, width]``), and holds at least
    ``SDPA_QUERY_BLOCKS`` queries when the row does, so each query's output is
    the one a single call over the whole ``[width, width]`` mask returns.
    """
    rows, width = key_mask.shape[0], query.shape[2]
    keys = torch.arange(width, device=query.device)
    minimum = torch.finfo(key_mask.dtype).min
    out = torch.empty_like(query)
    starts = list(range(0, width, LOCAL_QUERIES))
    if len(starts) > 1 and width - starts[-1] < SDPA_QUERY_BLOCKS:
        starts.pop()
    for start, stop in zip(starts, [*starts[1:], width], strict=True):
        # Only keys in [first, last) lie within the window of some chunk query.
        first, last = max(0, start - window), min(width, stop + window)
        far = (keys[None, first:last] - keys[start:stop, None]).abs() > window
        mask = key_mask.new_full((rows, 1, stop - start, width), minimum)
        mask[..., first:last] = key_mask[..., first:last].masked_fill(far, minimum)
        out[:, :, start:stop] = kernels("sdpa")(
            query[:, :, start:stop],
            key,
            value,
            mask,
            scale=scale,
            is_causal=False,
            enable_gqa=False,
        )
    return out


class UltraBertRotary(nn.Module):
    """cos / sin per rope theta for positions ``0..T-1``; computed, never loaded."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = config.get("head_dim") or (
            config["hidden_size"] // config["num_attention_heads"]
        )
        self.thetas = sorted(set(thetas(config)))
        for index, theta in enumerate(self.thetas):
            steps = torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float)
            inv_freq = 1.0 / (theta ** (steps / dim))
            self.register_buffer(f"inv_freq_{index}", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, length: int
    ) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
        """``{theta: (cos, sin)}``, each ``[1, length, head_dim]`` in ``x.dtype``."""
        positions = torch.arange(length, device=x.device)[None, None, :].float()
        out = {}
        for index, theta in enumerate(self.thetas):
            inv_freq = getattr(self, f"inv_freq_{index}")
            inv_freq = inv_freq[None, :, None].to(dtype=torch.float, device=x.device)
            with torch.autocast(device_type=x.device.type, enabled=False):
                freqs = (inv_freq @ positions).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos, sin = emb.cos(), emb.sin()
            out[theta] = (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))
        return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half positions of ``[B, H, T, D]`` query and key in their own dtype, as the modeling code."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (
        _rotate_half(key) * sin
    )


class UltraBertRMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        out: torch.Tensor = self.weight * hidden_states.to(dtype)
        return out


class UltraBertAttention(nn.Module):
    def __init__(self, config: dict[str, Any], window: int | None = None):
        super().__init__()
        self.window = window
        hidden = config["hidden_size"]
        heads = config["num_attention_heads"]
        if config.get("num_key_value_heads", heads) != heads:
            raise ValueError(
                "UltraBERT attention has as many key / value heads as query heads"
            )
        self.head_dim = config.get("head_dim") or hidden // heads
        self.scaling = self.head_dim**-0.5
        bias = bool(config.get("attention_bias", False))
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: Rotary,
        theta: float,
        kind: str,
        layout: Layout,
        kernels: KernelSet,
    ) -> torch.Tensor:
        """``rotary`` holds each grid width's cos / sin per theta."""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if len(layout.groups) == 1:
            grids = [(layout.groups[0], query, key, value)]
        else:
            sizes = list(layout.sizes)
            grids = list(
                zip(
                    layout.groups,
                    torch.split(query, sizes),
                    torch.split(key, sizes),
                    torch.split(value, sizes),
                    strict=True,
                )
            )
        parts = [
            group.to_tokens(
                self.attend(
                    group.to_rows(q),
                    group.to_rows(k),
                    group.to_rows(v),
                    rotary[group.width][theta],
                    kind,
                    group,
                    kernels,
                )
            )
            for group, q, k, v in grids
        ]
        result: torch.Tensor = self.o_proj(
            parts[0] if len(parts) == 1 else torch.cat(parts)
        )
        return result

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        kind: str,
        group: Group,
        kernels: KernelSet,
    ) -> torch.Tensor:
        """Attention over one grid's ``[rows, width, hidden]`` projections."""
        batch, width = query.shape[:2]
        shape = (batch, width, -1, self.head_dim)
        query, key = apply_rotary(
            query.view(shape).transpose(1, 2), key.view(shape).transpose(1, 2), *rotary
        )
        value = value.view(shape).transpose(1, 2)
        blocks = group.band if kind == SLIDING else None
        mask = group.masks.get(kind)
        if blocks is not None:
            output = banded_attention(kernels, query, key, value, blocks, self.scaling)
        elif kind == SLIDING and mask is not None and mask.shape[-2] < width:
            # A key mask on a local layer: the CPU padded layout.
            assert self.window is not None
            output = local_attention(
                kernels, query, key, value, mask, self.window, self.scaling
            )
        else:
            output = kernels("sdpa")(
                query,
                key,
                value,
                mask,
                scale=self.scaling,
                is_causal=False,
                enable_gqa=False,
            )
        return (
            output.transpose(1, 2).contiguous().reshape(batch, width, -1).contiguous()
        )


class UltraBertMLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError(
                f"unsupported UltraBERT activation {config['hidden_act']!r}"
            )
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        bias = bool(config.get("mlp_bias", False))
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return out


class UltraBertLayer(nn.Module):
    def __init__(self, config: dict[str, Any], window: int | None, theta: float):
        super().__init__()
        self.kind = FULL if window is None else SLIDING
        self.theta = theta
        eps = config.get("rms_norm_eps", 1e-6)
        self.self_attn = UltraBertAttention(config, window)
        self.mlp = UltraBertMLP(config)
        self.input_layernorm = UltraBertRMSNorm(config["hidden_size"], eps)
        self.post_attention_layernorm = UltraBertRMSNorm(config["hidden_size"], eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: Rotary,
        layout: Layout,
        kernels: KernelSet,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            rotary,
            self.theta,
            self.kind,
            layout,
            kernels,
        )
        out: torch.Tensor = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return out


class UltraBertBackbone(nn.Module):
    """The UltraBERT encoder; ``encode`` returns hidden states at layer exits, the last one final-normed."""

    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.window = local_window(config)
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [
                UltraBertLayer(config, window, theta)
                for window, theta in zip(windows(config), thetas(config), strict=True)
            ]
        )
        self.norm = UltraBertRMSNorm(
            config["hidden_size"], config.get("rms_norm_eps", 1e-6)
        )
        self.rotary_emb = UltraBertRotary(config)
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
        """Rows of ``lengths`` packed back to back (``modernbert.packed_layout``): the approximate path."""
        return packed_layout(lengths, self.window, device, width, uniform=uniform)

    def padded(
        self,
        attention_mask: torch.Tensor | None,
        rows: int,
        width: int,
        device: torch.device | str,
    ) -> Layout:
        """Padded ``[rows, width]`` rows with the modeling code's float masks, padding or not.

        On the CPU every layer gets the ``[rows, 1, 1, width]`` key mask
        instead: local layers then attend in query chunks (``local_attention``)
        and global layers broadcast it, both bit-identical to the dense masks.
        """
        if attention_mask is None:
            valid = torch.ones(rows, width, dtype=torch.bool, device=device)
        else:
            valid = attention_mask.to(device=device, dtype=torch.bool)
        dtype = self.embed_tokens.weight.dtype
        if not dtype.is_floating_point or dtype == torch.float64:
            dtype = torch.float32
        if torch.device(device).type == "cpu":
            minimum = torch.finfo(dtype).min
            key = torch.where(valid, 0.0, minimum).to(dtype)[:, None, None, :]
            return Layout((Group(rows, width, {FULL: key, SLIDING: key}),))
        masks = additive_masks(valid, rows, width, self.window, dtype)
        return Layout((Group(rows, width, dict(masks)),))

    def masked(
        self, valid: torch.Tensor, rows: int, width: int, device: torch.device | str
    ) -> Layout:
        """Padded rows masked from a device-side key mask, never read back (graphs)."""
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
        ``k`` layers. The last layer is final-normed (the modeling code's
        output); intermediate exits are too only with ``normalize_exits``.
        ``input_ids`` is ``[N]`` for a packed layout, ``[rows, width]`` for a padded one.
        """
        assert self.kernels is not None, "bind kernels before running the backbone"
        exits = tuple(exits) or (self.num_layers,)
        if not all(0 <= layer <= self.num_layers for layer in exits):
            raise ValueError(f"layer exits must lie in 0..{self.num_layers}: {exits}")

        def exit_state(count: int, hidden: torch.Tensor) -> torch.Tensor:
            if count == self.num_layers or normalize_exits:
                normed: torch.Tensor = self.norm(hidden)
                return normed
            return hidden

        if layout.order is not None:
            input_ids = input_ids.index_select(0, layout.order)
        hidden_states = self.embed_tokens(input_ids)
        out: dict[int, torch.Tensor] = {}
        if 0 in exits:
            out[0] = exit_state(0, hidden_states)
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
        """Padded ``[B, T]`` rows to ``[B, T, hidden]`` final hidden states (the modeling code's contract)."""
        rows, width = input_ids.shape
        layout = self.padded(attention_mask, rows, width, input_ids.device)
        return self.encode(input_ids, layout)[self.num_layers]
