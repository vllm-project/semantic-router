"""The causal convolution Decision-1.0-Eos was released with on gfx942: FP64 accumulation.

Its released runtime replaced the gated-delta layers' depthwise causal
convolution with this Triton kernel for inference calls with BF16 inputs whose
batch times length is at least 2048: the four taps accumulate in FP64, the sum
is rounded to BF16 as the reference convolution's output is, and SiLU runs in
FP32. Every other call keeps the device's default convolution. The ROCm
accelerator registers it as the ``causal_conv1d`` variant ``fp64_accumulate``,
which only models that name it run (``ModelSpec.kernel_variants``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP64_ACCUMULATE = "fp64_accumulate"
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


def fp64_conv(default: Callable) -> Callable:
    """The variant: the FP64 kernel where the released runtime used it, ``default`` everywhere else."""

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
            and hidden_states.shape[0] * hidden_states.shape[2]
            >= MINIMUM_BATCH_TIMES_LENGTH
        ):
            return causal_silu(hidden_states, weight)
        return default(hidden_states, weight, bias, activation=activation)

    return conv
