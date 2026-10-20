"""Unpack compressed-tensors int4 weights (``pack-quantized``, grouped) to BF16 with one Triton kernel.

Each int32 word holds eight 4-bit values of one row, the first in the lowest
bits, stored as ``value + 8``; ``weight = (nibble - 8 - zero) * scale`` with one
scale (and, for asymmetric weights, one zero point packed eight rows per word)
per ``group`` input columns, computed in FP32 and rounded once to the output
dtype, as compressed-tensors dequantizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _unpack(
    packed_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    n_words,
    words_per_row,
    groups_per_row,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    ZERO: tl.constexpr,
):
    words = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = words < n_words
    word = tl.load(packed_ptr + words, mask=mask, other=0)
    row = words // words_per_row
    column = (words % words_per_row) * 8
    scale = tl.load(
        scale_ptr + row * groups_per_row + column // GROUP, mask=mask, other=0.0
    ).to(tl.float32)
    nibbles = (word[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF
    values = (nibbles - 8).to(tl.float32)
    if ZERO:
        zero_word = tl.load(
            zero_ptr + (row // 8) * groups_per_row + column // GROUP,
            mask=mask,
            other=0,
        )
        zero = ((zero_word >> ((row % 8) * 4)) & 0xF) - 8
        values -= zero.to(tl.float32)[:, None]
    values = values * scale[:, None]
    out = words[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(out_ptr + out, values.to(out_ptr.dtype.element_ty), mask=mask[:, None])


def unpack(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
    group: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``packed`` [N, K / 8] int32, ``scale`` [N, K / group], ``zero`` [N / 8, K / group] -> the weight [N, K]."""
    rows, words_per_row = packed.shape
    out = torch.empty((rows, words_per_row * 8), device=packed.device, dtype=dtype)
    n_words = packed.numel()
    _unpack[(triton.cdiv(n_words, BLOCK),)](
        packed,
        scale,
        packed if zero is None else zero,
        out,
        n_words,
        words_per_row,
        scale.shape[1],
        GROUP=group,
        BLOCK=BLOCK,
        ZERO=zero is not None,
    )
    return out


@triton.jit
def _matmul(
    a_ptr,
    packed_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    M,
    N,
    K,
    groups_per_row,
    GROUP: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    words_per_row = K // 8
    shifts = (tl.arange(0, BK) % 8) * 4
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0
        )
        rw = k0 // 8 + tl.arange(0, BK // 8)
        word = tl.load(
            packed_ptr + rn[:, None] * words_per_row + rw[None, :],
            mask=rn[:, None] < N,
            other=0,
        )
        # each word eight times in a row, then its eight nibbles: [BN, BK]
        word = tl.interleave(word, word)
        word = tl.interleave(word, word)
        word = tl.interleave(word, word)
        nibbles = (word >> shifts[None, :]) & 0xF
        group = k0 // GROUP
        scale = tl.load(scale_ptr + rn * groups_per_row + group, mask=rn < N, other=0.0)
        if ZERO:
            zero_word = tl.load(
                zero_ptr + (rn // 8) * groups_per_row + group, mask=rn < N, other=0
            )
            zero = (zero_word >> ((rn % 8) * 4)) & 0xF
        else:
            zero = tl.full((BN,), 8, tl.int32)
        weight = (nibbles - zero[:, None]).to(tl.float32) * scale.to(tl.float32)[
            :, None
        ]
        acc += tl.dot(a, tl.trans(weight.to(tl.bfloat16)))
    tl.store(
        out_ptr + rm[:, None] * N + rn[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
    group: int,
) -> torch.Tensor:
    """``x`` [M, K] (BF16, contiguous) times the int4 weight [N, K] transposed, unpacked tile by tile.

    For few rows (``FUSED_ROWS``), where reading the weight dominates, without
    writing the unpacked weight; the sums run in another order than the BLAS
    GEMM's, so it belongs to approximate kernels.
    """
    rows, columns = x.shape
    n = packed.shape[0]
    block_k = min(group, 128)
    block_m = 16 if rows <= 16 else 32 if rows <= 32 else 64
    out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    _matmul[(triton.cdiv(rows, block_m), triton.cdiv(n, 64))](
        x,
        packed,
        scale,
        packed if zero is None else zero,
        out,
        rows,
        n,
        columns,
        scale.shape[1],
        GROUP=group,
        ZERO=zero is not None,
        BM=block_m,
        BN=64,
        BK=block_k,
        num_warps=4,
        num_stages=3,
    )
    return out
