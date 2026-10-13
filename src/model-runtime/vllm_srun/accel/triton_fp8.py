"""Per-row FP8 quantization of activations in one Triton kernel.

An FP8 linear layer (``engines/native/reduced.py``) multiplies FP8 weights with
FP8 activations scaled per row: each row's scale is its largest magnitude over
the FP8 format's largest value. Written with PyTorch operations the scaling
takes five kernels per call, which costs more than the FP8 matrix product
saves at the row counts the decision models run; this kernel reads each row
twice (its maximum, then the scaled values) and writes the FP8 row and its
scale.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _quantize_rows(
    X, Q, S, n_cols, stride_x, stride_q, fmax, BLOCK: tl.constexpr
):  # fmt: skip
    row = tl.program_id(0)
    x_row = X + row * stride_x
    peak = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        peak = tl.maximum(peak, tl.abs(x))
    scale = tl.maximum(tl.max(peak, axis=0), 1e-12) / fmax
    tl.store(S + row, scale)
    inverse = 1.0 / scale
    q_row = Q + row * stride_q
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * inverse, -fmax), fmax)
        tl.store(q_row + cols, q.to(Q.dtype.element_ty), mask=cols < n_cols)


def quantize_rows(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [rows, cols] with unit column stride as ``dtype`` and its per-row FP32 scales [rows, 1]."""
    rows, cols = x.shape
    q = torch.empty((rows, cols), device=x.device, dtype=dtype)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        _quantize_rows[(rows,)](
            x,
            q,
            scale,
            cols,
            x.stride(0),
            q.stride(0),
            torch.finfo(dtype).max,
            BLOCK=min(triton.next_power_of_2(cols), BLOCK),
            num_warps=8,
        )
    return q, scale
