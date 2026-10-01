# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""Eos's ROCm depthwise causal convolution + SiLU, as its native runtime runs it.

The native Decision-1.0-Eos runtime replaced the Qwen3.5 gated-delta layers'
``causal_conv1d_fn`` with this Triton kernel on gfx942 GPUs for full-forward
inference calls whose batch times length is at least 2048: the four-tap
convolution accumulates in FP64, is rounded to BF16 like the reference
convolution's output, and SiLU is computed in FP32. Every other call keeps the
reference function. Only this model's layers are rebound; nothing global changes.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None

MINIMUM_BATCH_TIMES_LENGTH = 2048
if triton is None:
    raise ImportError("Eos's native ROCm convolution needs Triton")


@triton.jit
def _causal_silu(
    X,
    W,
    Y,
    L: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
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


def causal_silu(x, weight, bias=None, activation=None, **kwargs):
    b, d, length = x.shape
    y = torch.empty((b, length, d), device=x.device, dtype=x.dtype)
    _causal_silu[(triton.cdiv(b * length * d, 256),)](
        x,
        weight,
        y,
        length,
        d,
        b * length * d,
        *x.stride(),
        4,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y.transpose(1, 2)


class ConvController:
    def __init__(self, reference):
        self.reference = reference

    def __call__(self, x, weight, bias=None, activation=None, **kwargs):
        if (
            not torch.is_grad_enabled()
            and x.device.type == "cuda"
            and x.ndim == 3
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and bias is None
            and activation in ("silu", "swish")
            and tuple(weight.shape) == (x.shape[1], 4)
            and weight.is_contiguous()
            and x.shape[0] * x.shape[2] >= MINIMUM_BATCH_TIMES_LENGTH
        ):
            return causal_silu(x, weight, bias, activation=activation, **kwargs)
        return self.reference(x, weight, bias, activation=activation, **kwargs)
