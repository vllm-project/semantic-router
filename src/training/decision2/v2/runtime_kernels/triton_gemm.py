"""Autotuned Triton BF16 GEMM for ``y = x W^T`` (comparison point for hipBLASLt; gfx942 config grid)."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_M": 8,
            "waves_per_eu": w,
            "matrix_instr_nonkdim": 16,
        },
        num_warps=nw,
        num_stages=2,
    )
    for bm, bn, bk, nw in (
        (16, 64, 128, 4),
        (32, 64, 128, 4),
        (32, 128, 64, 4),
        (64, 64, 64, 4),
        (64, 128, 64, 4),
        (64, 256, 64, 8),
        (128, 128, 64, 8),
        (128, 256, 64, 8),
        (256, 128, 64, 8),
        (256, 256, 64, 8),
    )
    for w in (0, 2)
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    w = w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = rk + k * BLOCK_K < K
        av = tl.load(a, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        wv = tl.load(w, mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        acc += tl.dot(av, wv)
        a += BLOCK_K
        w += BLOCK_K
    c = c_ptr + rm[:, None].to(tl.int64) * N + rn[None, :]
    tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def linear(x: Any, weight: Any) -> Any:
    """``x`` [M, K] BF16, ``weight`` [N, K] BF16 -> [M, N] BF16 (FP32 accumulation)."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    def grid(meta: dict[str, Any]) -> tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](x, weight, out, M, N, K)
    return out
