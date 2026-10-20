"""Per-row FP8 quantization of activations in one Triton kernel.

An FP8 linear layer (``engines/native/reduced.py``) multiplies FP8 weights with
FP8 activations scaled per row: each row's scale is its largest magnitude over
the FP8 format's largest value. Written with PyTorch operations the scaling
takes five kernels per call, which costs more than the FP8 matrix product
saves at the row counts the decision models run; this kernel reads each row
twice (its maximum, then the scaled values) and writes the FP8 row and its
scale.

Triton's launcher binds and specializes the arguments on every call, about
17 us of host time on an MI300X, more than the kernel itself takes, and
without graphs a model pays it once per FP8 layer and forward. The compiled
kernel of each argument shape is therefore launched directly after its first
call, once a direct launch has written the same bytes as the JIT path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 4096
# compiled kernel per argument shape, or None where only the JIT path launches it
_launches: dict[tuple, object] = {}


@triton.jit
def _quantize_rows(
    X, Q, S, n_cols, stride_x, stride_q, fmax, BLOCK: tl.constexpr
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)  # rows * stride passes 2**31 on long prompts
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
    if not rows:
        return q, scale
    block = min(triton.next_power_of_2(cols), BLOCK)
    args = (x, q, scale, cols, x.stride(0), q.stride(0), torch.finfo(dtype).max)
    # every argument Triton specializes on: integer values and pointer alignment
    key = (x.device, x.dtype, dtype, cols, x.stride(0), x.data_ptr() % 16)
    kernel = _launches.get(key)
    # Triton before 3.4 has no knobs; its launches take the JIT path only
    knobs = getattr(triton, "knobs", None)
    hooks = (
        None
        if knobs is None
        else (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    )
    # a hook (a profiler's) wants the launch metadata only the JIT path builds
    if (
        kernel is not None
        and hooks is not None
        and not any(getattr(h, "calls", h) for h in hooks)
    ):
        _launch(kernel, rows, (*args, block))
        return q, scale
    compiled = _quantize_rows[(rows,)](*args, BLOCK=block, num_warps=8)
    if key not in _launches and not torch.cuda.is_current_stream_capturing():
        _launches[key] = compiled if _direct_matches(compiled, args, block) else None
    return q, scale


def _launch(kernel, rows: int, args: tuple) -> None:
    """``kernel`` on ``rows`` programs, its arguments as the JIT launcher passes them."""
    stream = triton.runtime.driver.active.get_current_stream(args[0].device.index)
    meta = kernel.function, kernel.packed_metadata, None, None, None
    kernel.run(rows, 1, 1, stream, *meta, *args)


def _direct_matches(kernel, args: tuple, block: int) -> bool:
    """Whether launching ``kernel`` directly writes what the JIT launch of ``args`` wrote."""
    x, q, scale = args[:3]
    q2, scale2 = torch.empty_like(q), torch.empty_like(scale)
    try:
        _launch(kernel, q.shape[0], (x, q2, scale2, *args[3:], block))
    except Exception:  # a Triton whose launcher takes other arguments
        return False
    return torch.equal(q2.view(torch.uint8), q.view(torch.uint8)) and torch.equal(
        scale2, scale
    )
