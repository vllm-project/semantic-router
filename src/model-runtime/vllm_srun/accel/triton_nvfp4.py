"""NVFP4 quantization of activations in one Triton kernel (``engines/native/quantized.py``, ``NVFP4Linear``).

Each program reads ``BLOCKS`` groups of 16 consecutive values of a row-major
[rows, K] input, writes each group's FP8 (E4M3) scale ``amax * (1 / 6) *
global`` (clamped to 448, so a block beyond anything seen in calibration
saturates instead of overflowing to NaN) and its 16 values rounded to E2M1
codes, two per byte. It writes the bytes ``nvfp4.quantize`` computes, which
the GPU tests check: correctly rounded divisions and one rounding to E4M3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .nvfp4 import GROUP

BLOCKS = 64  # groups of 16 values per program


@triton.jit
def _round_e4m3(x):
    """``x`` (0 to 448) rounded once to the nearest FP8 E4M3 value, ties to even, still in FP32.

    Triton casts FP32 to E4M3 through FP16, which rounds twice: 116.0000076
    becomes 116 and then the even neighbour 112, where torch gives 120.
    Rounding here first leaves a value both casts keep.
    """
    bits = x.to(tl.int32, bitcast=True)
    # three mantissa bits of twenty-three: drop twenty, ties to even
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
    normal = bits.to(tl.float32, bitcast=True)
    subnormal = (
        libdevice.rint(x * 512.0) * 0.001953125
    )  # below 2**-6: multiples of 2**-9
    return tl.where(x < 0.015625, subnormal, normal)


@triton.jit
def _quantize(X, Q, S, global_scale, n_blocks, BLOCKS: tl.constexpr):  # fmt: skip
    blocks = tl.program_id(0).to(tl.int64) * BLOCKS + tl.arange(0, BLOCKS)
    mask = blocks < n_blocks
    lanes = tl.arange(0, 16)
    x = tl.load(
        X + blocks[:, None] * 16 + lanes[None, :], mask=mask[:, None], other=0.0
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    # correctly rounded divisions, as torch's: a value on a rounding threshold gets the reference's code
    scale = _round_e4m3(tl.minimum(amax * (1.0 / 6.0) * global_scale, 448.0))
    scale = scale.to(tl.float8e4nv)
    tl.store(S + blocks, scale, mask=mask)
    denom = tl.math.div_rn(scale.to(tl.float32), global_scale)
    v = tl.math.div_rn(x, tl.where(denom > 0, denom, 1.0)[:, None])
    a = tl.minimum(tl.abs(v), 6.0)
    # nearest E2M1 magnitude (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even code
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = code | ((v < 0).to(tl.int32) << 3)
    lo, hi = tl.split(tl.reshape(code, (BLOCKS, 8, 2)))
    tl.store(
        Q + blocks[:, None] * 8 + tl.arange(0, 8)[None, :],
        (lo | (hi << 4)).to(tl.uint8),
        mask=mask[:, None],
    )


def quantize(x: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] contiguous activations -> packed codes [rows, K / 2] (uint8), block scales [rows, K / 16] (FP8)."""
    rows, k = x.shape
    q = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(rows, k // GROUP, dtype=torch.float8_e4m3fn, device=x.device)
    n_blocks = rows * k // GROUP
    _quantize[(triton.cdiv(n_blocks, BLOCKS),)](
        x, q, s, global_scale, n_blocks, BLOCKS=BLOCKS
    )
    return q, s
