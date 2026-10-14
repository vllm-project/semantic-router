"""CLAP's HTSAT audio encoder (Vela 1.0 Omni's environmental-sound branch), returning the pooled vector.

Every block reproduces the operation order of Transformers' unfused
``ClapAudioEncoder``: BatchNorm over the mel bins, the bicubic resize of the
time axis to the Swin input (``align_corners``), the mel-to-image fold, a
patch convolution with LayerNorm, Swin stages of windowed attention (the
relative-position bias, cyclic shifts on every second block with the -100
region mask, explicit matmul and softmax) joined by patch merging, the final
LayerNorm and the grouped average pool. Parameter names match the
checkpoints' ``ClapAudioModel.audio_encoder`` namespace; BatchNorm's running
statistics load as buffers, and the relative-position index is computed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .modernbert import ACTIVATIONS

if TYPE_CHECKING:
    from . import ComputedBuffers

MODEL_TYPE = "clap_audio_model"
SHIFT_MASK = -100.0
# Checkpoint tensors the encoder recomputes or never reads.
IGNORED_TENSORS = ("relative_position_index", "num_batches_tracked")


def window_partition(hidden_states: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = hidden_states.shape
    hidden_states = hidden_states.view(
        batch, height // window, window, width // window, window, channels
    )
    return (
        hidden_states.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window, window, channels)
    )


def window_reverse(
    windows: torch.Tensor, window: int, height: int, width: int
) -> torch.Tensor:
    channels = windows.shape[-1]
    windows = windows.view(
        -1, height // window, width // window, window, window, channels
    )
    return (
        windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, channels)
    )


def relative_position_index(window: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
    )
    flat = torch.flatten(coords, 1)
    relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window - 1
    relative[:, :, 1] += window - 1
    relative[:, :, 0] *= 2 * window - 1
    return relative.sum(-1)


def shift_mask(size: int, window: int, shift: int) -> torch.Tensor:
    """The SW-MSA mask of a ``size`` x ``size`` grid: -100 between tokens of different shifted regions."""
    regions = torch.zeros((1, size, size, 1))
    bounds = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    count = 0
    for rows in bounds:
        for columns in bounds:
            regions[:, rows, columns, :] = count
            count += 1
    windows = window_partition(regions, window).view(-1, window * window)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, SHIFT_MASK).masked_fill(mask == 0, 0.0)


class BatchNorm(nn.Module):
    """Inference BatchNorm over channel dim 1, with the checkpoint's running statistics."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


class ClapSelfAttention(nn.Module):
    def __init__(self, config: dict[str, Any], dim: int, heads: int):
        super().__init__()
        window = config["window_size"]
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, heads)
        )
        self.register_buffer(
            "relative_position_index", relative_position_index(window), persistent=False
        )
        bias = bool(config.get("qkv_bias", True))
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

    def computed_buffers(self) -> None:
        """The index buffer is computed, never loaded: rebuild it after a meta-device build."""
        self.relative_position_index = relative_position_index(self.window)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch, tokens, channels = hidden_states.shape
        shape = (batch, tokens, -1, self.head_dim)
        query = self.query(hidden_states).view(shape).transpose(1, 2)
        key = self.key(hidden_states).view(shape).transpose(1, 2)
        value = self.value(hidden_states).view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        scores = scores + bias.unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            scores = scores.view(batch // windows, windows, self.heads, tokens, tokens)
            scores = scores + mask.unsqueeze(1).unsqueeze(0)
            scores = scores.view(-1, self.heads, tokens, tokens)
        context = torch.matmul(F.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.view(*context.shape[:-2], channels)


class ClapDense(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.dense = nn.Linear(inputs, outputs)


class ClapAttention(nn.Module):
    def __init__(self, config: dict[str, Any], dim: int, heads: int):
        super().__init__()
        self.self = ClapSelfAttention(config, dim, heads)
        self.output = ClapDense(dim, dim)


class ClapLayer(nn.Module):
    """One Swin block; ``shift`` is 0 where the grid is no larger than a window."""

    def __init__(
        self, config: dict[str, Any], dim: int, size: int, heads: int, shift: int
    ):
        super().__init__()
        window = config["window_size"]
        if size <= window:
            window, shift = size, 0
        self.window, self.shift, self.size = window, shift, size
        eps = config["layer_norm_eps"]
        activation = config["hidden_act"]
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported CLAP activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.attention = ClapAttention(config, dim, heads)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)
        hidden = int(config["mlp_ratio"] * dim)
        self.intermediate = ClapDense(dim, hidden)
        self.output = ClapDense(hidden, dim)
        self.register_buffer(
            "mask",
            shift_mask(size, window, shift) if shift else None,
            persistent=False,
        )

    def computed_buffers(self) -> None:
        if self.shift:
            self.mask = shift_mask(self.size, self.window, self.shift)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, _, channels = hidden_states.shape
        size, window, shift = self.size, self.window, self.shift
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states).view(
            batch, size, size, channels
        )
        if shift:
            hidden_states = torch.roll(
                hidden_states, shifts=(-shift, -shift), dims=(1, 2)
            )
        windows = window_partition(hidden_states, window).view(
            -1, window * window, channels
        )
        mask = None if self.mask is None else self.mask.to(windows.dtype)
        attended = self.attention.self(windows, mask)
        attended = self.attention.output.dense(attended)
        attended = window_reverse(
            attended.view(-1, window, window, channels), window, size, size
        )
        if shift:
            attended = torch.roll(attended, shifts=(shift, shift), dims=(1, 2))
        hidden_states = shortcut + attended.view(batch, size * size, channels)
        layer_output = self.act(
            self.intermediate.dense(self.layernorm_after(hidden_states))
        )
        out: torch.Tensor = hidden_states + self.output.dense(layer_output)
        return out


class ClapPatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, hidden_states: torch.Tensor, size: int) -> torch.Tensor:
        batch, _, channels = hidden_states.shape
        grid = hidden_states.view(batch, size, size, channels)
        merged = torch.cat(
            [
                grid[:, 0::2, 0::2],
                grid[:, 1::2, 0::2],
                grid[:, 0::2, 1::2],
                grid[:, 1::2, 1::2],
            ],
            -1,
        ).view(batch, -1, 4 * channels)
        reduced: torch.Tensor = self.reduction(self.norm(merged))
        return reduced


class ClapStage(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        dim: int,
        size: int,
        depth: int,
        heads: int,
        last: bool,
    ):
        super().__init__()
        shift = config["window_size"] // 2
        self.size = size
        self.blocks = nn.ModuleList(
            [
                ClapLayer(config, dim, size, heads, 0 if index % 2 == 0 else shift)
                for index in range(depth)
            ]
        )
        self.downsample = None if last else ClapPatchMerging(dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states, self.size)
        return hidden_states


class ClapPatchEmbed(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        patch, stride = config["patch_size"], config["patch_stride"]
        stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        if tuple(stride) != (patch, patch):
            raise ValueError("only non-overlapping CLAP patches are supported")
        hidden = config["patch_embeds_hidden_size"]
        self.proj = nn.Conv2d(
            config["patch_embed_input_channels"],
            hidden,
            kernel_size=patch,
            stride=patch,
        )
        self.norm = (
            nn.LayerNorm(hidden)
            if config.get("enable_patch_layer_norm", True)
            else nn.Identity()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches: torch.Tensor = self.norm(self.proj(images).flatten(2).transpose(1, 2))
        return patches


class ClapAudioBackbone(nn.Module):
    """``input_features`` ``[B, 1, frames, mels]`` (one 10-second window each) to ``{"pooled": [B, hidden]}``."""

    model_type = MODEL_TYPE
    ignored_tensors = IGNORED_TENSORS

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if config.get("enable_fusion"):
            raise ValueError("fused CLAP audio encoders are not supported")
        if not config.get("flatten_patch_embeds", True):
            raise ValueError("CLAP patch embeddings must be flattened")
        self.config = config
        self.spec_size = config["spec_size"]
        self.mels = config["num_mel_bins"]
        self.freq_ratio = self.spec_size // self.mels
        depths, heads = config["depths"], config["num_attention_heads"]
        width = config["patch_embeds_hidden_size"]
        grid = self.spec_size // config["patch_size"]
        self.depths = len(depths)
        self.patch_embed = ClapPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                ClapStage(
                    config,
                    width * 2**index,
                    grid // 2**index,
                    depth,
                    heads[index],
                    last=index == len(depths) - 1,
                )
                for index, depth in enumerate(depths)
            ]
        )
        self.batch_norm = BatchNorm(self.mels)
        self.hidden = width * 2 ** (len(depths) - 1)
        self.norm = nn.LayerNorm(self.hidden)
        self.patch_stride = config["patch_size"]
        self.kernels: KernelSet | None = None

    def computed_buffers(self) -> None:
        """Rebuild every computed buffer (relative-position indices, shift masks) after a meta build."""
        for module in self.modules():
            if module is not self and hasattr(module, "computed_buffers"):
                cast("ComputedBuffers", module).computed_buffers()

    def images(self, input_features: torch.Tensor) -> torch.Tensor:
        """Normalized features as the Swin input image ``[B, 1, spec, spec]``."""
        normalized: torch.Tensor = self.batch_norm(
            input_features.transpose(1, 3)
        ).transpose(1, 3)
        _, _, frames, mels = normalized.shape
        width, height = (
            self.spec_size * self.freq_ratio,
            self.spec_size // self.freq_ratio,
        )
        if frames > width or mels > height:
            raise ValueError("the CLAP window is larger than the Swin input")
        if frames < width:
            normalized = F.interpolate(
                normalized, (width, mels), mode="bicubic", align_corners=True
            )
        if mels < height:
            normalized = F.interpolate(
                normalized, (frames, height), mode="bicubic", align_corners=True
            )
        batch, channels, time, freq = normalized.shape
        normalized = normalized.reshape(
            batch, channels * self.freq_ratio, time // self.freq_ratio, freq
        )
        normalized = normalized.permute(0, 1, 3, 2).contiguous()
        return normalized.reshape(
            batch, channels, freq * self.freq_ratio, time // self.freq_ratio
        )

    def forward(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        images = self.images(input_features)
        frames = images.shape[2]
        hidden_states = self.patch_embed(images)
        for stage in self.layers:
            hidden_states = stage(hidden_states)
        hidden_states = self.norm(hidden_states)
        batch, _, channels = hidden_states.shape
        side = frames // 2 ** (self.depths - 1) // self.patch_stride
        grid = (
            hidden_states.permute(0, 2, 1)
            .contiguous()
            .reshape(batch, channels, side, side)
        )
        bins = side // self.freq_ratio
        grid = grid.reshape(batch, channels, side // bins, bins, side)
        grid = (
            grid.permute(0, 1, 3, 2, 4).contiguous().reshape(batch, channels, bins, -1)
        )
        pooled = F.adaptive_avg_pool1d(torch.flatten(grid, 2), 1)
        return {"pooled": torch.flatten(pooled, 1)}
