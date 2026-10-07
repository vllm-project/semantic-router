"""Building blocks shared by the native Qwen backbones.

Each block reproduces the operation order of the Transformers 5.17 modules
the released packages were scored with, so the CPU path is bit-identical to
them. Parameter names match the Transformers checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ....accel.kernels import KernelSet
from .tree import Tree, is_tree, tree_attention

PADDING_MASK_RANK = 2
# Transformers passes enable_gqa to SDPA only up to this head size.
SDPA_GQA_MAX_HEAD_DIM = 256


class RMSNorm(nn.Module):
    """``weight * normalize(x)`` (Qwen3)."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ZeroCenteredRMSNorm(nn.Module):
    """``normalize(x) * (1 + weight)`` computed in FP32 (Qwen3.5)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GatedRMSNorm(nn.Module):
    """Normalize, scale, then gate with SiLU(gate) (Qwen3.5 gated delta output)."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full rotary embedding over the head dimension (Qwen3)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_partial_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding over the first ``cos.shape[-1]`` channels (Qwen3.5)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def causal_mask(
    attention_mask: torch.Tensor | None, length: int
) -> torch.Tensor | None:
    """The SDPA mask: None when no row is padded (then SDPA runs ``is_causal``), else causal AND padding.

    Rows are right-padded, so a padded batch always needs the explicit boolean mask.
    """
    if attention_mask is None:
        return None
    padding = attention_mask.to(dtype=torch.bool)
    if padding.sum() == padding.numel():
        return None
    device = padding.device
    batch = torch.arange(padding.shape[0], device=device)[:, None, None, None]
    q_index = torch.arange(length, device=device)[None, None, :, None]
    kv_index = torch.arange(length, device=device)[None, None, None, :]
    mask = q_index.new_ones((), dtype=torch.bool)
    mask = mask & (kv_index <= q_index)
    mask = mask & padding[batch, kv_index]
    return mask.expand(padding.shape[0], -1, length, length)


def recurrent_mask(
    attention_mask: torch.Tensor | None, length: int
) -> torch.Tensor | None:
    """The 2D padding mask the gated-delta layers multiply into their inputs, or None when unpadded."""
    if (
        attention_mask is None
        or attention_mask.ndim != PADDING_MASK_RANK
        or length == 1
    ):
        return None
    if torch.all(attention_mask == 1):
        return None
    return attention_mask[:, -length:].contiguous()


def attention(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | Tree | None,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """SDPA as Transformers dispatches it: GQA in-kernel only without a mask; returns [B, T, H, D].

    A shared-context ``Tree`` as the mask runs the tree attention instead.
    """
    if is_tree(mask):
        return tree_attention(query, key, value, mask, groups=groups, scaling=scaling)
    enable_gqa = False
    if groups > 1:
        if mask is None and key.shape[-1] == value.shape[-1] <= SDPA_GQA_MAX_HEAD_DIM:
            enable_gqa = True
        else:
            key = repeat_kv(key, groups)
            value = repeat_kv(value, groups)
    is_causal = query.shape[2] > 1 and mask is None
    output: torch.Tensor = kernels("sdpa")(
        query,
        key,
        value,
        mask,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )
    return output.transpose(1, 2).contiguous()
