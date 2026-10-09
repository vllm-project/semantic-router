"""Shared-context tree forward: a multi-question request's shared prefix computed once.

The prefix and every question's suffix are packed into one row with the exact
path's positions. Full-attention layers: the prefix attends causally to itself;
every suffix query attends to the prefix keys (one kernel for all queries) and
to its own suffix (one causal kernel over the padded suffix rows), and the two
results are merged by their log-sum-exp. Gated-delta layers: the prefix runs
from a zero state; the suffixes run as variable-length sequences from the
prefix-end recurrent state, each convolution window starting with the prefix's
last inputs. These are the operations of the released runtime's shared-context
switch (``decision2/shared_ctx.py`` tree mode), so the answers equal it; they
can differ from the exact path by rounding (opt-in ``shared_context`` profile).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ....accel.kernels import KernelSet
    from .qwen3_5 import GatedDeltaNet


class Tree:
    """A request packed into one row: the shared prefix, then every suffix in order.

    ``rows`` / ``packed`` move suffix tokens between the packed row and a padded
    [questions, width] layout. Padding sits after each suffix and repeats its last
    token: causal attention, convolution and recurrence never carry it into a real
    token, and the readout never reads it.
    """

    is_tree = True

    def __init__(
        self,
        prefix: int,
        lengths: list[int],
        width: int,
        device: torch.device,
        padded_exact: bool,
    ):
        self.prefix = prefix
        self.lengths = lengths
        starts = [0]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        index = torch.empty((len(lengths), width), dtype=torch.long)
        valid = torch.zeros((len(lengths), width), dtype=torch.bool)
        for row, (start, length) in enumerate(zip(starts, lengths, strict=False)):
            index[row, :length] = torch.arange(start, start + length)
            index[row, length:] = start + length - 1
            valid[row, :length] = True
        self.index = index.to(device)
        # Index (not mask) gathers: a boolean mask would sync the host on every use.
        self.flat = valid.reshape(-1).nonzero().squeeze(1).to(device)
        ends = [start + length for start, length in zip(starts, lengths, strict=False)]
        self.cu_seqlens_cpu = torch.tensor([0, *ends], dtype=torch.long)
        self.cu_seqlens = self.cu_seqlens_cpu.to(device)
        self.positions = torch.cat(
            [torch.arange(prefix)]
            + [torch.arange(prefix, prefix + length) for length in lengths]
        )[None].to(device)
        # The exact batch of a request whose rows differ in padded length runs SDPA with an explicit mask; the
        # prefix keeps that mask regime.
        self.prefix_mask = (
            torch.ones(prefix, prefix, dtype=torch.bool, device=device).tril()[
                None, None
            ]
            if padded_exact
            else None
        )

    def rows(self, packed: torch.Tensor) -> torch.Tensor:
        """[suffix tokens, ...] -> [questions, width, ...]."""
        return packed[self.index]

    def packed(self, rows: torch.Tensor) -> torch.Tensor:
        """[questions, width, ...] -> [suffix tokens, ...]."""
        return rows.reshape(-1, *rows.shape[2:])[self.flat]


def is_tree(mask: object) -> TypeIs[Tree]:
    """Whether a layer's mask is a shared-context ``Tree``, which runs the tree forward."""
    return bool(getattr(mask, "is_tree", False))


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output and its log-sum-exp per query ([B, H, L, D], [B, H, L])."""
    if query.is_cuda:
        out = torch.ops.aten._scaled_dot_product_efficient_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            None,
            True,
            0.0,
            causal,
            scale=scale,
        )
        return out[0], out[1][..., : query.shape[2]]
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    if causal:
        q, k = scores.shape[-2:]
        keep = torch.ones(q, k, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, -float("inf"))
    lse = scores.logsumexp(-1)
    out = (scores - lse[..., None]).exp() @ value.float()
    return out.to(query.dtype), lse


def tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tree: Tree,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Attention over a packed tree ([1, H, L, D] inputs); returns [1, L, H, D] like ``common.attention``."""
    if groups > 1:
        b, h, n, d = key.shape
        key = key[:, :, None].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)
        value = (
            value[:, :, None].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)
        )
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    q, k, v = query.to(dtype), key.to(dtype), value.to(dtype)
    p = tree.prefix
    head = F.scaled_dot_product_attention(
        q[:, :, :p], k[:, :, :p], v[:, :, :p], attn_mask=tree.prefix_mask, is_causal=tree.prefix_mask is None,
        scale=scaling,
    )  # fmt: skip
    out_prefix, lse_prefix = _attend(
        q[:, :, p:], k[:, :, :p], v[:, :, :p], False, scaling
    )

    def rows(x: torch.Tensor) -> torch.Tensor:
        return tree.rows(x[0].transpose(0, 1)).permute(0, 2, 1, 3)

    out_own, lse_own = _attend(
        rows(q[:, :, p:]), rows(k[:, :, p:]), rows(v[:, :, p:]), True, scaling
    )
    out_own = tree.packed(out_own.transpose(1, 2))
    lse_own = tree.packed(lse_own.transpose(1, 2))
    # softmax over [prefix; own] = the two parts weighted by sigmoid of their lse gap
    weight = torch.sigmoid(lse_prefix[0].transpose(0, 1) - lse_own)[..., None]
    tail = torch.lerp(out_own.float(), out_prefix[0].transpose(0, 1).float(), weight)
    out = torch.cat([head[0].transpose(0, 1), tail.to(dtype)], dim=0)
    return out[None]


def varlen(rule: Any) -> bool:
    """Whether the bound chunked gated-delta kernel takes ``cu_seqlens`` (FLA)."""
    return "cu_seqlens" in inspect.signature(rule).parameters


def suffix_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    """The gated delta rule over every suffix from the prefix-end state ``start`` ([1, S, ...] packed inputs)."""
    if varlen(rule):
        out, _ = rule(
            q, k, v, g=g, beta=beta, initial_state=start, output_final_state=False, use_qk_l2norm_in_kernel=True,
            cu_seqlens=tree.cu_seqlens, cu_seqlens_cpu=tree.cu_seqlens_cpu,
        )  # fmt: skip
        return out
    out_rows, _ = rule(
        tree.rows(q[0]), tree.rows(k[0]), tree.rows(v[0]), g=tree.rows(g[0]), beta=tree.rows(beta[0]),
        initial_state=start, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    return tree.packed(out_rows)[None]


def tree_gated_delta(
    m: GatedDeltaNet, hidden_states: torch.Tensor, tree: Tree, kernels: KernelSet
) -> torch.Tensor:
    """``GatedDeltaNet.forward`` over a packed tree (no padding in the row), eager ops."""
    _, length, _ = hidden_states.shape
    p, n = tree.prefix, len(tree.lengths)
    mixed = m.in_proj_qkv(hidden_states)
    z = m.in_proj_z(hidden_states).reshape(1, length, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)
    conv = kernels("causal_conv1d")
    weight, bias, window = m.conv1d.weight.squeeze(1), m.conv1d.bias, m.conv_kernel_size
    head = conv(mixed[:, :p].transpose(1, 2), weight, bias, activation=m.activation)
    rows = torch.cat(
        [mixed[:, p - (window - 1) : p].expand(n, -1, -1), tree.rows(mixed[0, p:])],
        dim=1,
    )
    tail = conv(rows.transpose(1, 2), weight, bias, activation=m.activation)[
        :, :, window - 1 :
    ]
    mixed = torch.cat(
        [head[0].transpose(0, 1), tree.packed(tail.transpose(1, 2))], dim=0
    )[None]
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(1, length, -1, m.head_k_dim)
    key = key.reshape(1, length, -1, m.head_k_dim)
    value = value.reshape(1, length, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    rule = kernels("chunk_gated_delta_rule")
    out_prefix, state = rule(
        query[:, :p], key[:, :p], value[:, :p], g=g[:, :p], beta=beta[:, :p], initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    start = state.expand(n, *state.shape[1:]).contiguous()
    out_suffix = suffix_rule(
        rule, tree, query[:, p:], key[:, p:], value[:, p:], g[:, p:], beta[:, p:], start
    )
    core = torch.cat([out_prefix, out_suffix], dim=1)
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    out: torch.Tensor = m.out_proj(core.reshape(1, length, -1))
    return out
