"""Triton helpers that sum squares in exactly the order of ATen's inner-dimension reduction (ROCm).

``x.pow(2).mean(-1)`` on a contiguous FP32 row of H elements runs ATen's ``reduce_kernel`` with
one 64-lane wavefront per row. For H > 128 the input is vectorised by ``VEC`` (4 for FP32):
lane t keeps VEC accumulators, accumulator i adds element ``VEC*(t + 64*c) + i`` for
c = 0, 1, ... in order, the accumulators are combined left to right, and the 64 lane values
are combined by ``shfl_down`` with offsets 1, 2, 4, ..., 32 (a pairwise tree in lane order).
For H = 128 the vector width is still 4 but only 32 lanes take part (one load each). Reproducing that
association makes the FP32 mean bit-identical, so the RMSNorm kernels round exactly like the
reference. ``probe_mean`` checked the order (and ``rsqrt_rn``) bit for bit against ATen for every
hidden size we use; ``probe_reduce`` checks the kernels built on them.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def rsqrt_rn(x):
    """``torch.rsqrt`` on ROCm is correctly rounded; OCML's FP32 rsqrt is not (90% agreement)."""
    return libdevice.rsqrt(x.to(tl.float64)).to(tl.float32)


@triton.jit
def lane_tree64(v, R: tl.constexpr):
    """[R, 64] -> [R]: ((v0 + v1) + (v2 + v3)) + ... (shfl_down offsets 1, 2, ..., 32)."""
    a, b = tl.split(tl.reshape(v, [R, 32, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 16, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 8, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 4, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 2, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 1, 2]))
    v = a + b
    return tl.reshape(v, [R])


@triton.jit
def combine_vec4(acc, R: tl.constexpr):
    """[R, 64, 4] accumulators -> [R, 64]: ((a0 + a1) + a2) + a3."""
    even, odd = tl.split(tl.reshape(acc, [R, 64, 2, 2]))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    return ((a0 + a1) + a2) + a3


@triton.jit
def sumsq_vec4_chunk(x, R: tl.constexpr):
    """Squares of one [R, 256] chunk laid out as the [R, 64 lanes, 4 accumulators] update."""
    return tl.reshape(x * x, [R, 64, 4])


@triton.jit
def sumsq_128(x, R: tl.constexpr):
    """[R, 128] FP32 -> [R] sum of squares in ATen's order for 128-wide rows.

    The input is vectorised by 4 over 32 lanes (one load each); the lane tree then has five
    levels (``probe_mean`` variant 1, bit-identical; unvectorised orders are not).
    """
    even, odd = tl.split(tl.reshape(x * x, [R, 32, 2, 2]))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    v = ((a0 + a1) + a2) + a3
    a, b = tl.split(tl.reshape(v, [R, 16, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 8, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 4, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 2, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [R, 1, 2]))
    return tl.reshape(a + b, [R])


@triton.jit
def sumsq_256(x, R: tl.constexpr):
    """[R, 256] FP32 -> [R] sum of squares in ATen's vectorised order (one VEC=4 load per lane)."""
    return lane_tree64(combine_vec4(sumsq_vec4_chunk(x, R), R), R)


__all__ = [
    "lane_tree64",
    "combine_vec4",
    "sumsq_vec4_chunk",
    "sumsq_128",
    "sumsq_256",
    "triton",
]
