"""Tree attention for packed prefix-tree rows: ancestor bitmask, key-block skipping, sigmoid gate.

FlashAttention-2-style online softmax over key blocks; a key block that no query row of
the program can see is skipped. The mask arrives as bits: ``bits[b, i, w]`` (int32) holds
"row i sees key 32*w + j" in bit j, built once per request and shared by every layer and
head. The epilogue applies the gated-attention gate with the reference's rounding points
(attention output rounded to BF16, BF16 sigmoid, BF16 product). GQA: query head h reads
key/value head h // (H / Hkv). Written for head dim 256 (Qwen3.5 full attention).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "waves_per_eu": w}, num_warps=nw, num_stages=1
    )
    for bm, bn, nw in (
        (32, 32, 4),
        (64, 32, 4),
        (64, 64, 4),
        (64, 32, 8),
        (128, 32, 8),
        (128, 64, 8),
        (32, 64, 4),
    )
    for w in (0, 2)
]


@triton.autotune(configs=_CONFIGS, key=["N", "H", "HKV", "D", "PV_FP32"])
@triton.jit
def _tree_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bits_ptr,
    out_ptr,
    N,
    H,
    HKV,
    W,
    scale,
    sqb,
    sqh,
    sqt,
    skb,
    skh,
    skt,
    svb,
    svh,
    svt,
    sgb,
    sgt,
    sgh,
    sbb,
    D: tl.constexpr,
    GATE: tl.constexpr,
    PV_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = bh % H
    hk = h // (H // HKV)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    d = tl.arange(0, D)
    q = tl.load(
        q_ptr + b * sqb + h * sqh + rows[:, None] * sqt + d[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    cols0 = tl.arange(0, BLOCK_N)
    WPB: tl.constexpr = BLOCK_N // 32
    bits_row = bits_ptr + b * sbb + rows * W
    for kb in range(0, tl.cdiv(N, BLOCK_N)):
        w0 = tl.load(bits_row + kb * WPB, mask=rmask, other=0)
        if WPB == 2:
            w1 = tl.load(
                bits_row + kb * WPB + 1, mask=rmask & (kb * WPB + 1 < W), other=0
            )
            visible = (w0 | w1) != 0
        else:
            w1 = w0
            visible = w0 != 0
        if tl.sum(visible.to(tl.int32), axis=0) > 0:
            cols = kb * BLOCK_N + cols0
            cmask = cols < N
            k = tl.load(
                k_ptr + b * skb + hk * skh + cols[:, None] * skt + d[None, :],
                mask=cmask[:, None],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * scale
            if WPB == 2:
                word = tl.where(cols0[None, :] < 32, w0[:, None], w1[:, None])
            else:
                word = w0[:, None]
            bit = (word >> (cols0[None, :] % 32)) & 1
            s = tl.where(bit != 0, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                v_ptr + b * svb + hk * svh + cols[:, None] * svt + d[None, :],
                mask=cmask[:, None],
                other=0.0,
            )
            if PV_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
    o = (acc / l_i[:, None]).to(tl.bfloat16)
    if GATE:
        gt = tl.load(
            g_ptr + b * sgb + rows[:, None] * sgt + h * sgh + d[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        gt = gt.to(tl.float32)
        sg = (1.0 / (1.0 + libdevice.exp(-gt))).to(tl.bfloat16).to(tl.float32)
        o = (o.to(tl.float32) * sg).to(tl.bfloat16)
    tl.store(
        out_ptr + b * N * H * D + rows[:, None] * (H * D) + h * D + d[None, :],
        o,
        mask=rmask[:, None],
    )


def pack_mask(mask: Any) -> Any:
    """Bool [B, N, N] (row sees key) -> int32 [B, N, ceil(N/32)] bit words (bit j of word w = key 32w+j)."""
    B, N, _ = mask.shape
    W = (N + 31) // 32
    padded = torch.zeros(B, N, W * 32, dtype=torch.int64, device=mask.device)
    padded[:, :, :N] = mask.to(torch.int64)
    weights = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=mask.device),
        torch.arange(32, device=mask.device),
    )
    words = (padded.view(B, N, W, 32) * weights).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).contiguous()


def tree_attention(
    q: Any,
    k: Any,
    v: Any,
    bits: Any,
    gate: Any | None,
    scale: float,
    pv_fp32: bool = False,
) -> Any:
    """q [B, H, N, D], k/v [B, Hkv, N, D] (any strides, unit last stride), bits from ``pack_mask``,
    gate [B, N, H, D] view or None. Returns [B, N, H*D] BF16 (gated when ``gate`` is given).
    """
    B, H, N, D = q.shape
    HKV = k.shape[1]
    out = torch.empty(B, N, H * D, dtype=torch.bfloat16, device=q.device)
    g = gate if gate is not None else q
    gs = (
        (gate.stride(0), gate.stride(1), gate.stride(2))
        if gate is not None
        else (0, 0, 0)
    )

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    _tree_attn_kernel[grid](
        q, k, v, g, bits, out, N, H, HKV, bits.shape[-1], scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        *gs, bits.stride(0),
        D=D, GATE=gate is not None, PV_FP32=pv_fp32,
    )  # fmt: skip
    return out
