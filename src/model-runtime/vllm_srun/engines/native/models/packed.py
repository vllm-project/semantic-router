"""Packed variable-length forwards: the rows of a merged forward back to back in one row, without padding.

A forward that merges requests (``LoadedModel.run_approximate``) may lay its
rows out as one sequence of their real tokens (``Packed``) instead of a padded
``[rows, longest row]`` block, so a row costs its own tokens. Every layer keeps
the rows apart. Full attention runs one variable-length kernel
(``varlen_attention``) in which each row attends to itself only. A Gated
DeltaNet layer runs its causal convolution over the rows separated by
``window - 1`` zero rows (``gapped``), so each row starts from the zero history
a left-padded row sees, and its chunked delta rule over ``cu_seqlens``, each
row from a zero state. Rotary positions count from 0 in every row, as a row
alone has them; relative to a left-padded row only the rounding of the
rotation tables differs. Packed forwards round differently from padded ones,
so only merged forwards run them; requests' own forwards stay padded.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .tree import varlen

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ....accel.kernels import KernelSet
    from .qwen3_5 import GatedDeltaNet


class Packed:
    """Rows of ``lengths`` tokens laid back to back in one row of ``sum(lengths)`` tokens.

    ``cu_seqlens`` holds the row boundaries (``[0, end of row 0, ...]``) for
    the gated-delta kernel, ``cu_seqlens_int32`` the same for attention, and
    ``positions`` each token's position within its row. ``last`` indexes each
    row's last token.
    """

    is_packed = True

    def __init__(self, lengths: list[int], device: torch.device):
        if not lengths or min(lengths) < 1:
            raise ValueError("packed rows need at least one token each")
        self.lengths = list(lengths)
        self.rows = len(lengths)
        self.total = sum(lengths)
        self.max_length = max(lengths)
        ends = list(itertools.accumulate(lengths))
        self.cu_seqlens_cpu = torch.tensor([0, *ends], dtype=torch.long)
        self.cu_seqlens = self.cu_seqlens_cpu.to(device)
        self.cu_seqlens_int32 = self.cu_seqlens.to(torch.int32)
        self.positions = torch.cat([torch.arange(n) for n in lengths]).to(device)
        self.last = torch.tensor([end - 1 for end in ends], dtype=torch.long).to(device)
        self.device = device
        self._gaps: dict[int, torch.Tensor] = {}

    def _real(self, window: int) -> torch.Tensor:
        """Where each token sits in the gapped layout: ``window - 1`` zero rows before every row."""
        index = self._gaps.get(window)
        if index is None:
            row = torch.repeat_interleave(
                torch.arange(self.rows), torch.tensor(self.lengths)
            )
            index = (torch.arange(self.total) + (row + 1) * (window - 1)).to(
                self.device
            )
            self._gaps[window] = index
        return index

    def gapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """``[1, T, C]`` -> ``[1, T + rows * (window - 1), C]`` with zero rows before every row."""
        out = x.new_zeros((1, self.total + self.rows * (window - 1), *x.shape[2:]))
        out[0, self._real(window)] = x[0]
        return out

    def ungapped(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """The inverse of ``gapped``: ``[1, T', ...]`` -> ``[1, T, ...]``."""
        return x[:, self._real(window)]


def is_packed(mask: object) -> TypeIs[Packed]:
    """Whether a layer's mask is a ``Packed`` layout, which runs the packed forward."""
    return bool(getattr(mask, "is_packed", False))


def packed_attention(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed: Packed,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Each packed row attending to itself ([1, H, T, D] inputs); returns [1, T, H, D] like ``common.attention``."""
    q, k, v = (x[0].transpose(0, 1) for x in (query, key, value))
    return varlen_attention(kernels, q, k, v, packed, groups=groups, scaling=scaling)[
        None
    ]


def varlen_attention(
    kernels: KernelSet,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed: Packed,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Each packed row attending to itself, ``[T, H, D]`` in and out, in the autocast dtype when autocast is on."""
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    q, k, v = (x.to(dtype).contiguous() for x in (query, key, value))
    out: torch.Tensor = kernels.preferred("varlen_attention")(
        q, k, v, packed, scale=scaling, groups=groups
    )
    return out


def packed_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    packed: Packed,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """The chunked gated delta rule over every packed row from a zero state ([1, T, ...] inputs)."""
    if varlen(rule):
        out, _ = rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
            cu_seqlens=packed.cu_seqlens, cu_seqlens_cpu=packed.cu_seqlens_cpu,
        )  # fmt: skip
        return out
    outputs = []
    for start, end in itertools.pairwise(packed.cu_seqlens_cpu.tolist()):
        row = slice(start, end)
        out, _ = rule(
            q[:, row], k[:, row], v[:, row], g=g[:, row], beta=beta[:, row], initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def packed_gated_delta(
    m: GatedDeltaNet, hidden_states: torch.Tensor, packed: Packed, kernels: KernelSet
) -> torch.Tensor:
    """``GatedDeltaNet.forward`` over packed rows ([1, T, hidden]), eager ops.

    The convolution reads the gapped rows; its input projection has no bias,
    so the gaps stay zero through it.
    """
    _, length, _ = hidden_states.shape
    window = m.conv_kernel_size
    z = m.in_proj_z(hidden_states).reshape(1, length, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)
    convolved = kernels("causal_conv1d")(
        m.in_proj_qkv(packed.gapped(hidden_states, window)).transpose(1, 2),
        m.conv1d.weight.squeeze(1),
        m.conv1d.bias,
        activation=m.activation,
    )
    mixed = packed.ungapped(convolved.transpose(1, 2), window)
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(1, length, -1, m.head_k_dim)
    key = key.reshape(1, length, -1, m.head_k_dim)
    value = value.reshape(1, length, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    core = packed_rule(
        kernels("chunk_gated_delta_rule"), packed, query, key, value, g, beta
    )
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    out: torch.Tensor = m.out_proj(core.reshape(1, length, -1))
    return out


def packed_unavailable(backbone: Any, kernels: KernelSet) -> str | None:
    """Why this backbone cannot run packed forwards with these kernels; None when it can.

    Packing pays only with a variable-length attention kernel and a
    gated-delta kernel that takes ``cu_seqlens``; the reference loops over
    rows that stand in for them elsewhere run slower than padding.
    """
    if not hasattr(backbone, "forward_packed"):
        return "the backbone has no packed forward"
    if not getattr(backbone, "noncausal", False):
        return "only noncausal backbones pack merged rows"
    if kernels.preferred_source("varlen_attention") == REFERENCE_SOURCE:
        return "no variable-length attention kernel on this device"
    kinds = set(getattr(backbone, "config", {}).get("layer_types") or ())
    if "linear_attention" in kinds and not varlen(kernels("chunk_gated_delta_rule")):
        return "the gated-delta kernel takes no cu_seqlens"
    return None


REFERENCE_SOURCE = "torch-reference"
