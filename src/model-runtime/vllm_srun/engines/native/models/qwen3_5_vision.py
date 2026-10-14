"""The Qwen3.5 vision tower: patch embedding, resampled learned positions, axial rotary blocks and the merger.

Decision 3.0 packages carry it under ``visual.``. Every operation follows the
Transformers 5.17 ``Qwen3_5VisionModel`` the packages were scored with, except
the patch embedding, which runs as the matrix product it equals (a Conv3d
whose kernel equals its stride over inputs already cut into patches), as the
released runtime runs it. Each image attends only to itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .common import rotate_half

MODEL_TYPE = "qwen3_5_vision"
LAYER_NORM_EPS = 1e-6


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu_pytorch_tanh": gelu_tanh,
    "gelu": gelu,
}


class VisionMLP(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.linear_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear_fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.act = ACTIVATIONS[config["hidden_act"]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear_fc2(self.act(self.linear_fc1(x)))
        return out


def rotate_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``apply_rotary_pos_emb_vision``: the rotation in FP32, results in the inputs' dtypes."""
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.dim = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: list[int],
        rotary: tuple[torch.Tensor, torch.Tensor],
        kernels: KernelSet,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = rotate_vision(query, key, *rotary)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        sdpa = kernels("sdpa")
        outputs = [
            sdpa(q, k, v, None, scale=self.scaling, is_causal=False, enable_gqa=False)
            .transpose(1, 2)
            .contiguous()
            for q, k, v in zip(
                *(torch.split(t, lengths, dim=2) for t in (query, key, value)),
                strict=True,
            )
        ]
        attended = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
        out: torch.Tensor = self.proj(attended)
        return out


class VisionBlock(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"], eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(config["hidden_size"], eps=LAYER_NORM_EPS)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: list[int],
        rotary: tuple[torch.Tensor, torch.Tensor],
        kernels: KernelSet,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), lengths, rotary, kernels
        )
        out: torch.Tensor = hidden_states + self.mlp(self.norm2(hidden_states))
        return out


class PatchEmbed(nn.Module):
    """The Conv3d patch embedding, computed as ``F.linear`` over flattened patches."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        size = (
            config["temporal_patch_size"],
            config["patch_size"],
            config["patch_size"],
        )
        self.embed_dim = config["hidden_size"]
        self.proj = nn.Conv3d(
            config["in_channels"],
            self.embed_dim,
            kernel_size=size,
            stride=size,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight
        flat = pixel_values.reshape(-1, weight[0].numel()).to(weight.dtype)
        out = F.linear(flat, weight.reshape(weight.shape[0], -1), self.proj.bias)
        return out.view(-1, self.embed_dim)


class PatchMerger(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.norm = nn.LayerNorm(config["hidden_size"], eps=LAYER_NORM_EPS)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config["out_hidden_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        out: torch.Tensor = self.linear_fc2(F.gelu(self.linear_fc1(x)))
        return out


class VisionRotary(nn.Module):
    """Axial 2D rotary frequencies: the same frequencies for the row and the column, over the whole head."""

    inv_freq: torch.Tensor

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        rope = config["rope_parameters"]
        if rope.get("rope_type") != "axial":
            raise ValueError(f"unsupported vision rope type {rope.get('rope_type')!r}")
        spatial = (config["hidden_size"] // config["num_heads"]) // 2
        inv_freq = 1.0 / (
            rope["rope_theta"]
            ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=positions.device.type, enabled=False):
            freqs = positions[..., None].float() * self.inv_freq.float()
            cos, sin = freqs.cos() * 1.0, freqs.sin() * 1.0
        return self._recompose(cos), self._recompose(sin)

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        hw = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)
        return torch.cat([hw, hw], dim=-1)


def position_ids(
    grids: list[tuple[int, int, int]], merge: int, device: torch.device
) -> torch.Tensor:
    """``get_vision_position_ids``: (row, column) of every patch, laid out by merge block."""
    out = []
    for t, h, w in grids:
        rows, cols = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        shape = (h // merge, merge, w // merge, merge)
        rows = rows.reshape(shape).transpose(1, 2).flatten()
        cols = cols.reshape(shape).transpose(1, 2).flatten()
        out.append(torch.stack([rows, cols], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def _axis_taps(
    index: torch.Tensor, size: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear taps and weights with ``align_corners=True`` into a table of ``side`` rows, clamped at the border."""
    index = index.to(torch.float32)
    src = index * (side - 1) / torch.clamp(size - 1, min=1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - distance).clamp(min=0)


def interpolation(
    grids: list[tuple[int, int, int]], side: int, merge: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``get_vision_interpolation_indices_and_weights`` (bilinear, align corners, merge-block order)."""
    grid = torch.tensor(grids, dtype=torch.long, device=device)
    counts = grid[:, 0] * grid[:, 1] * grid[:, 2]
    heights = torch.repeat_interleave(grid[:, 1], counts)
    widths = torch.repeat_interleave(grid[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    total = sum(t * h * w for t, h, w in grids)
    within = (torch.arange(total, device=device) - starts) % (heights * widths)
    blocks_w = widths // merge
    row = (within // (merge * merge * blocks_w)) * merge + (within // merge) % merge
    col = ((within // (merge * merge)) % blocks_w) * merge + within % merge
    h_taps, h_weights = _axis_taps(row, heights, side)
    w_taps, w_weights = _axis_taps(col, widths, side)
    n = h_taps.shape[1]
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return indices, weights


class Qwen3_5VisionBackbone(nn.Module):
    model_type = MODEL_TYPE

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.merge = config["spatial_merge_size"]
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config["num_position_embeddings"], config["hidden_size"]
        )
        self.side = int(config["num_position_embeddings"] ** 0.5)
        self.rotary_emb = VisionRotary(config)
        self.blocks = nn.ModuleList(
            [VisionBlock(config) for _ in range(config["depth"])]
        )
        self.merger = PatchMerger(config)
        self.kernels: KernelSet | None = None
        if config.get("deepstack_visual_indexes"):
            raise ValueError("deepstack vision features are not supported")

    def forward(
        self, pixel_values: torch.Tensor, grids: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        """Merged image features ``[tokens, out_hidden_size]`` for the patch rows of every image, in order.

        The resampling taps, weights and rotary positions are computed on the host
        (each weight is one correctly rounded FP32 operation, the same on any
        device), so the forward never reads the device back.
        """
        assert self.kernels is not None, "bind kernels before running the tower"
        device = pixel_values.device
        host = torch.device("cpu")
        indices, weights = interpolation(grids, self.side, self.merge, host)
        indices, weights = indices.to(device), weights.to(device)
        positions = position_ids(grids, self.merge, host).to(device)
        lengths = [h * w for t, h, w in grids for _ in range(t)]
        hidden = self.patch_embed(pixel_values.type(self.patch_embed.proj.weight.dtype))
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)
        rotary = self.rotary_emb(positions)
        for block in self.blocks:
            hidden = block(hidden, lengths, rotary, self.kernels)
        merged: torch.Tensor = self.merger(hidden)
        return merged
