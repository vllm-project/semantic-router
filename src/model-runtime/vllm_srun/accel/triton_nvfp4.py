"""NVFP4 quantization of activations in one Triton kernel (``engines/native/quantized.py``, ``NVFP4Linear``).

Each program reads ``BLOCKS`` groups of 16 consecutive values of a row-major
[rows, K] input, writes each group's FP8 (E4M3) scale ``amax * (1 / 6) *
global`` (clamped to 448, so a block beyond anything seen in calibration
saturates instead of overflowing to NaN) and its 16 values rounded to E2M1
codes, two per byte. It writes the bytes ``nvfp4.quantize`` computes, which
the GPU tests check: correctly rounded divisions and one rounding to E4M3.

The scales go straight to cuBLAS's 32x4x4 layout (``nvfp4.swizzle``): the
kernel runs over the padded scale matrix (rows to 128, groups to 4) and writes
zero where it pads, so a call launches this kernel and the FP4 product and
nothing between them. An NVFP4 layer runs one call per forward, about 400 in a
d3 forward, so host time counts as it does for ``triton_fp8``: after a shape's
first call its compiled kernel is launched directly, once a direct launch has
written the same bytes as the JIT path. The row count is not specialized on,
so every request of a layer shape takes the same compiled kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .nvfp4 import GROUP

BLOCKS = 64  # groups of 16 values per program
# compiled kernel per argument shape, or None where only the JIT path launches it
_launches: dict[tuple, object] = {}


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


@triton.jit(do_not_specialize=["rows", "n_blocks"])
def _quantize(
    X, Q, S, global_scale, rows, n_blocks,
    K: tl.constexpr, GROUPS: tl.constexpr, PADDED: tl.constexpr, BLOCKS: tl.constexpr,
):  # fmt: skip
    # one block per (row, group) of the padded [rows to 128, GROUPS to PADDED] scale matrix
    blocks = tl.program_id(0).to(tl.int64) * BLOCKS + tl.arange(0, BLOCKS)
    inside = blocks < n_blocks
    row, group = blocks // PADDED, blocks % PADDED
    real = inside & (row < rows) & (group < GROUPS)
    lanes = tl.arange(0, 16)
    x = tl.load(
        X + (row * K + group * 16)[:, None] + lanes[None, :],
        mask=real[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)  # zero where the matrix is padded
    scale = _round_e4m3(tl.minimum(amax * (1.0 / 6.0) * global_scale, 448.0))
    scale = scale.to(tl.float8e4nv)
    # cuBLAS's 32x4x4 tiles: 128 rows by 4 groups, row r % 32 then (r % 128) // 32, then the group
    tile = (row // 128) * (PADDED // 4) + group // 4
    at = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
    tl.store(S + at, scale, mask=inside)
    # correctly rounded divisions, as torch's: a value on a rounding threshold gets the reference's code
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
        Q + (row * (K // 2) + group * 8)[:, None] + tl.arange(0, 8)[None, :],
        (lo | (hi << 4)).to(tl.uint8),
        mask=real[:, None],
    )


def quantize(x: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] contiguous activations -> packed codes [rows, K / 2] (uint8) and block scales (FP8 E4M3).

    The scales are flat in cuBLAS's 32x4x4 layout of the [rows, K / 16] matrix
    padded to 128 rows and a multiple of 4 groups, zero where padded.
    """
    rows, k = x.shape
    groups = k // GROUP
    padded = -(-groups // 4) * 4
    n_blocks = -(-rows // 128) * 128 * padded
    q = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(n_blocks, dtype=torch.float8_e4m3fn, device=x.device)
    args = (x, q, s, global_scale, rows, n_blocks)
    grid = triton.cdiv(n_blocks, BLOCKS)
    # every argument Triton specializes on: the shape constants and pointer alignment
    key = (
        x.device,
        x.dtype,
        k,
        x.data_ptr() % 16,
        q.data_ptr() % 16,
        s.data_ptr() % 16,
    )
    kernel = _launches.get(key)
    # Triton before 3.4 has no knobs; its launches take the JIT path only
    knobs = getattr(triton, "knobs", None)
    hooks = (
        None
        if knobs is None
        else (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    )
    # a hook (a profiler's) wants the launch metadata only the JIT path builds
    constants = (k, groups, padded, BLOCKS)  # the launcher takes the constexprs too
    if (
        kernel is not None
        and hooks is not None
        and not any(getattr(h, "calls", h) for h in hooks)
    ):
        _launch(kernel, grid, (*args, *constants))
        return q, s
    compiled = _quantize[(grid,)](
        *args, K=k, GROUPS=groups, PADDED=padded, BLOCKS=BLOCKS
    )
    if key not in _launches and not torch.cuda.is_current_stream_capturing():
        direct = _direct_matches(compiled, grid, (*args, *constants))
        _launches[key] = compiled if direct else None
    return q, s


def _launch(kernel, grid: int, args: tuple) -> None:
    """``kernel`` on ``grid`` programs, its arguments as the JIT launcher passes them (constexprs last)."""
    stream = triton.runtime.driver.active.get_current_stream(args[0].device.index)
    meta = kernel.function, kernel.packed_metadata, None, None, None
    kernel.run(grid, 1, 1, stream, *meta, *args)


def _direct_matches(kernel, grid: int, args: tuple) -> bool:
    """Whether launching ``kernel`` directly writes what the JIT launch of ``args`` wrote."""
    x, q, s = args[:3]
    q2, s2 = torch.empty_like(q), torch.empty_like(s)
    try:
        _launch(kernel, grid, (x, q2, s2, *args[3:]))
    except Exception:  # a Triton whose launcher takes other arguments
        return False
    return torch.equal(q2, q) and torch.equal(s2.view(torch.uint8), s.view(torch.uint8))
