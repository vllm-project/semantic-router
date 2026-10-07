"""Tree forward over padded rows: prefixes left-padded to one width, blocks right-padded to another.

Every block continues from the prefix row that owns it. Gated-delta layers:
the prefix rows run from a zero state (left padding is zeroed, so it never
moves the state); each block's convolution reads its row's last inputs and
its chunked rule starts from the row's final state. Full-attention layers: a
prefix row attends causally to its real tokens; a block attends to its row's
keys and to its own, in one causal call per block (lower-right aligned).
Projections, norms and MLPs run on the prefix rows and on the blocks as two
tensors. These are the tensor shapes and operations of the Vela 2.0 packages'
tree forward, so the states equal theirs on the same device; the packed tree
(``tree.py``) computes the same values from fewer tokens, with other rounding.
The element-wise steps are hooks (``project``, ``gate``, ``norm``) so the GPU
fast path can run them as fused kernels that round exactly like these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from .common import apply_partial_rotary

if TYPE_CHECKING:
    from ....accel.kernels import KernelSet
    from .qwen3_5 import GatedAttention, GatedDeltaNet


@dataclass(frozen=True)
class ForestShape:
    """The host-known shape of a forest: per prefix row its left padding, per block its length and row."""

    padding: tuple[int, ...]
    lengths: tuple[int, ...]
    owners: tuple[int, ...]


@dataclass(frozen=True)
class Forest:
    """Where the real tokens are: the device masks, each block's row (``owner``) and the shape."""

    prefix_mask: torch.Tensor
    block_mask: torch.Tensor
    owner: torch.Tensor
    shape: ForestShape


def _compute_dtype(device: torch.device, fallback: torch.dtype) -> torch.dtype:
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return fallback


def forest_gated_delta(
    m: GatedDeltaNet,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    forest: Forest,
    kernels: KernelSet,
    norm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``GatedDeltaNet`` over prefix rows ``[B, Lp, H]`` and blocks ``[N, Lb, H]``.

    ``norm(core, z)`` is the gated output norm over ``[tokens, head_v_dim]`` rows (``m.norm``).
    """
    norm = norm or m.norm
    prefix = prefix * forest.prefix_mask[..., None].to(prefix.dtype)
    blocks = blocks * forest.block_mask[..., None].to(blocks.dtype)
    rows, width, _ = prefix.shape
    count, block_width, _ = blocks.shape
    window, channels, weight = m.conv_kernel_size, m.conv_dim, m.conv1d.weight
    mixed_prefix, mixed_blocks = m.in_proj_qkv(prefix), m.in_proj_qkv(blocks)
    conv_prefix = F.conv1d(
        F.pad(mixed_prefix.transpose(1, 2), (window - 1, 0)),
        weight,
        m.conv1d.bias,
        groups=channels,
    )
    tail = (
        mixed_prefix[:, width - (window - 1) :]
        if width >= window - 1
        else F.pad(mixed_prefix, (0, 0, window - 1 - width, 0))
    )
    conv_blocks = F.conv1d(
        torch.cat([tail[forest.owner], mixed_blocks], 1).transpose(1, 2),
        weight,
        m.conv1d.bias,
        groups=channels,
    )
    conv_prefix = F.silu(conv_prefix).transpose(1, 2)
    conv_blocks = F.silu(conv_blocks).transpose(1, 2)

    def split(
        x: torch.Tensor, n: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(x, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        query = query.reshape(n, length, -1, m.head_k_dim)
        key = key.reshape(n, length, -1, m.head_k_dim)
        value = value.reshape(n, length, -1, m.head_v_dim)
        repeat = m.num_v_heads // m.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        return query, key, value

    def gates(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = m.in_proj_b(h).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(h).float() + m.dt_bias)
        return g, beta

    dtype = _compute_dtype(prefix.device, prefix.dtype)

    def cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).contiguous()

    qp, kp, vp = split(conv_prefix, rows, width)
    qb, kb, vb = split(conv_blocks, count, block_width)
    gp, bp = gates(prefix)
    gb, bb = gates(blocks)
    rule = kernels("chunk_gated_delta_rule")
    out_prefix, state = rule(
        cast(qp), cast(kp), cast(vp), gp.contiguous(), cast(bp), initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    out_blocks, _ = rule(
        cast(qb), cast(kb), cast(vb), gb.contiguous(), cast(bb), initial_state=state[forest.owner].contiguous(),
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip

    def output(
        core: torch.Tensor, h: torch.Tensor, n: int, length: int
    ) -> torch.Tensor:
        z = m.in_proj_z(h).reshape(-1, m.head_v_dim)
        core = norm(core.reshape(-1, m.head_v_dim), z).reshape(n, length, -1)
        out: torch.Tensor = m.out_proj(core)
        return out

    return output(out_prefix, prefix, rows, width), output(
        out_blocks, blocks, count, block_width
    )


def project_attention(
    m: GatedAttention, h: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries, keys and values ``[n, heads, L, head_dim]`` (in the values' dtype) and the output gate."""
    shape = h.shape[:-1]
    query, gate = torch.chunk(m.q_proj(h).view(*shape, -1, m.head_dim * 2), 2, dim=-1)
    query = m.q_norm(query).transpose(1, 2)
    key = m.k_norm(m.k_proj(h).view(*shape, -1, m.head_dim)).transpose(1, 2)
    value = m.v_proj(h).view(*shape, -1, m.head_dim).transpose(1, 2)
    query, key = apply_partial_rotary(query, key, *rotary)
    return query.to(value.dtype), key.to(value.dtype), value, gate.reshape(*shape, -1)


def gate_attention(
    m: GatedAttention, out: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """The gated output projection of attention outputs ``[n, heads, L, head_dim]``."""
    out = out.transpose(1, 2).reshape(*gate.shape)
    gated: torch.Tensor = m.o_proj(out * torch.sigmoid(gate))
    return gated


def forest_attention(
    m: GatedAttention,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    prefix_rotary: tuple[torch.Tensor, torch.Tensor],
    block_rotary: tuple[torch.Tensor, torch.Tensor],
    forest: Forest,
    project: Callable[
        [GatedAttention, torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = project_attention,
    gate: Callable[
        [GatedAttention, torch.Tensor, torch.Tensor], torch.Tensor
    ] = gate_attention,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``GatedAttention`` over prefix rows and blocks: one causal call per row and per block.

    ``project`` and ``gate`` are ``project_attention`` and ``gate_attention`` or fused equivalents.
    """
    qp, kp, vp, gate_prefix = project(m, prefix, prefix_rotary)
    qb, kb, vb, gate_blocks = project(m, blocks, block_rotary)

    def expand(t: torch.Tensor) -> torch.Tensor:
        return t.repeat_interleave(m.groups, dim=1) if m.groups > 1 else t

    rows = []
    shape = forest.shape
    for row, padding in enumerate(shape.padding):
        out = F.scaled_dot_product_attention(
            qp[row : row + 1, :, padding:], expand(kp[row : row + 1, :, padding:]),
            expand(vp[row : row + 1, :, padding:]), is_causal=True, scale=m.scaling,
        )  # fmt: skip
        rows.append(F.pad(out, (0, 0, padding, 0)))
    block_width = qb.shape[2]
    outs = []
    for index, (owner, length) in enumerate(
        zip(shape.owners, shape.lengths, strict=True)
    ):
        padding = shape.padding[owner]
        keys = torch.cat(
            [kp[owner : owner + 1, :, padding:], kb[index : index + 1, :, :length]], 2
        )
        values = torch.cat(
            [vp[owner : owner + 1, :, padding:], vb[index : index + 1, :, :length]], 2
        )
        out = F.scaled_dot_product_attention(
            qb[index : index + 1, :, :length], expand(keys), expand(values), scale=m.scaling,
            attn_mask=causal_lower_right(length, keys.shape[2]),
        )  # fmt: skip
        outs.append(F.pad(out, (0, 0, 0, block_width - length)))

    return gate(m, torch.cat(rows, 0), gate_prefix), gate(
        m, torch.cat(outs, 0), gate_blocks
    )
