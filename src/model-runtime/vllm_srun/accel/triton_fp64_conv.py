"""FP64-accumulating causal convolutions of released runtimes on gfx942.

Decision-1.0-Eos's released runtime replaced the gated-delta layers' depthwise
causal convolution with this Triton kernel for inference calls with BF16
inputs whose batch times length is at least 2048: the four taps accumulate in
FP64, the sum is rounded to BF16 as the reference convolution's output is, and
SiLU runs in FP32. Every other call keeps the device's default convolution.
The Decision 3.0 runtime ran ``F.conv1d``, which MIOpen computes with its naive
kernel, the same FP64 accumulation, for every call. The ROCm accelerator
registers them as the ``causal_conv1d`` variants ``fp64_accumulate`` and
``fp64_naive``, which only models that name them run
(``ModelSpec.kernel_variants``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP64_ACCUMULATE = "fp64_accumulate"
FP64_NAIVE = "fp64_naive"
MINIMUM_BATCH_TIMES_LENGTH = 2048
TAPS = 4
BLOCK = 256


@triton.jit
def _causal_silu(
    X, W, Y,
    L: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    K: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    d = i % D
    t = (i // D) % L
    b = i // (D * L)
    acc = tl.full((BLOCK,), 0, tl.float64)
    for k in tl.static_range(K):
        pos = t - (K - 1) + k
        x = tl.load(
            X + b * S0 + d * S1 + pos * S2, mask=valid & (pos >= 0), other=0
        ).to(tl.float64)
        w = tl.load(W + d * K + k, mask=valid, other=0).to(tl.float64)
        acc = tl.fma(x, w, acc)
    v = acc.to(tl.float32).to(X.dtype.element_ty).to(tl.float32)
    y = tl.div_rn(v, 1.0 + libdevice.exp(-v))
    tl.store(Y + i, y, mask=valid)


def causal_silu(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``SiLU(conv(x))`` over ``[batch, channels, time]``; returned as the released kernel lays it out."""
    batch, channels, length = hidden_states.shape
    out = torch.empty(
        (batch, length, channels),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    count = batch * length * channels
    _causal_silu[(triton.cdiv(count, BLOCK),)](
        hidden_states, weight, out, length, channels, count, *hidden_states.stride(), TAPS, BLOCK,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out.transpose(1, 2)


@triton.jit
def _gdn_prep_fp64(
    X, W, B, A, ALOG, DTB, Q, K, V, G, BETA,
    T, C, NK, NV,
    DK: tl.constexpr, BT: tl.constexpr, KW: tl.constexpr,
):  # fmt: skip
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    nblk = tl.cdiv(T, BT)
    bidx = (pid_t // nblk).to(tl.int64)
    t = (pid_t % nblk) * BT + tl.arange(0, BT)
    tmask = t < T
    d = tl.arange(0, DK)
    c = j * DK + d
    row = bidx * T + t
    acc = tl.zeros([BT, DK], dtype=tl.float64)
    for w in tl.static_range(KW):
        tt = t - (KW - 1) + w
        m = (tt >= 0) & tmask
        xv = tl.load(
            X + (bidx * T + tt)[:, None] * C + c[None, :], mask=m[:, None], other=0.0
        )
        wv = tl.load(W + c * KW + w)
        acc = acc + wv.to(tl.float64)[None, :] * xv.to(tl.float64)
    conv = acc.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    y = tl.div_rn(conv, 1.0 + libdevice.exp(-conv)).to(tl.bfloat16)
    if j < NK:
        tl.store(
            Q + row[:, None] * (NK * DK) + (j * DK + d)[None, :], y, mask=tmask[:, None]
        )
    elif j < 2 * NK:
        tl.store(
            K + row[:, None] * (NK * DK) + ((j - NK) * DK + d)[None, :],
            y,
            mask=tmask[:, None],
        )
    else:
        hv = j - 2 * NK
        tl.store(
            V + row[:, None] * (NV * DK) + (hv * DK + d)[None, :],
            y,
            mask=tmask[:, None],
        )
        bb = tl.load(B + row * NV + hv, mask=tmask, other=0.0).to(tl.float32)
        tl.store(
            BETA + row * NV + hv,
            (1.0 / (1.0 + libdevice.exp(-bb))).to(tl.bfloat16),
            mask=tmask,
        )
        s = tl.load(A + row * NV + hv, mask=tmask, other=0.0).to(tl.float32) + tl.load(
            DTB + hv
        )
        sp = tl.where(s > 20.0, s, libdevice.log1p(libdevice.exp(s)))
        tl.store(G + row * NV + hv, -libdevice.exp(tl.load(ALOG + hv)) * sp, mask=tmask)


def gdn_prep_fp64(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    k_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``triton_gfx942.gdn_prep`` with the convolution of ``fp64_naive``: taps summed in FP64, rounded to BF16, then SiLU."""
    batch, length, channels = mixed_qkv.shape
    nv = (channels - 2 * k_heads * head_dim) // head_dim
    device = mixed_qkv.device
    q = torch.empty(
        batch, length, k_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.empty_like(q)
    v = torch.empty(batch, length, nv, head_dim, dtype=torch.bfloat16, device=device)
    g = torch.empty(batch, length, nv, dtype=torch.float32, device=device)
    beta = torch.empty(batch, length, nv, dtype=torch.bfloat16, device=device)
    _gdn_prep_fp64[(batch * triton.cdiv(length, 16), 2 * k_heads + nv)](
        mixed_qkv, conv_weight, b, a, A_log, dt_bias, q, k, v, g, beta,
        length, channels, k_heads, nv,
        DK=head_dim, BT=16, KW=conv_weight.shape[-1], num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return q, k, v, g, beta


def fp64_conv(default: Callable, minimum: int = MINIMUM_BATCH_TIMES_LENGTH) -> Callable:
    """A variant: the FP64 kernel where the released runtime used it (batch times length at least
    ``minimum``), ``default`` everywhere else."""

    def conv(hidden_states, weight, bias=None, activation=None):
        if (
            not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and hidden_states.ndim == 3
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and bias is None
            and activation in ("silu", "swish")
            and tuple(weight.shape) == (hidden_states.shape[1], TAPS)
            and weight.is_contiguous()
            and hidden_states.shape[0] * hidden_states.shape[2] >= minimum
        ):
            return causal_silu(hidden_states, weight)
        return default(hidden_states, weight, bias, activation=activation)

    return conv
