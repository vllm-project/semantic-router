"""Fused Triton kernels for the non-GEMM ops of a Qwen3.5 decoder layer (ROCm / gfx942 first).

Each kernel replaces one ``reference`` function and keeps its BF16 rounding
points: FP32 residual stream and norms, BF16 at every place the reference
rounds (Linear inputs, SiLU / sigmoid outputs, the gated norm's intermediate
cast, the conv output before FLA's L2 norm). Transcendentals use the same
OCML functions the ATen kernels call (``exp``, ``log1p``, ``rsqrt``) and
floating-point contraction is disabled, as in the ROCm builds of ATen and
causal-conv1d (``probe_conv`` checks the conv arithmetic bit for bit).
Reductions (RMSNorm means, the L2 norm) cannot follow ATen's reduction tree,
so those outputs can differ from the reference in the last FP32 bit, which
moves a BF16 result by one unit in rare cases.

Kernels:
    add_rmsnorm      residual add + zero-centred RMSNorm, BF16 Linear input
    silu_mul         SiLU(gate) * up on the gate/up GEMM output(s)
    gdn_prep         causal conv + SiLU + q/k/v split + q/k L2 norm + beta + g
    gated_rmsnorm    Gated DeltaNet output norm with the SiLU(z) gate
    attn_prep        q/gate split, q/k RMSNorm, partial RoPE, [B, H, T, D] layout
    sigmoid_gate     attention output * sigmoid(gate)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .aten_reduce import (
    combine_vec4,
    lane_tree64,
    sumsq_128,
    sumsq_256,
    sumsq_vec4_chunk,
)

EXACT = {"enable_fp_fusion": False}


@triton.jit
def _add_rmsnorm_kernel(
    res_ptr,
    delta_ptr,
    w1_ptr,
    hidden_ptr,
    out_ptr,
    H,
    inv_h,
    eps,
    HAS_DELTA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(res_ptr + row * H + cols, mask=mask, other=0.0)
    if HAS_DELTA:
        x = x + tl.load(delta_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(hidden_ptr + row * H + cols, x, mask=mask)
    var = tl.sum(x * x, axis=0) * inv_h
    rstd = libdevice.rsqrt(var + eps)
    w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0)
    y = (x * rstd) * w1
    tl.store(out_ptr + row * H + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _add_rmsnorm_exact_kernel(
    res_ptr,
    delta_ptr,
    w1_ptr,
    hidden_ptr,
    out_ptr,
    M,
    H,
    inv_h,
    eps,
    HAS_DELTA: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    rmask = (rows < M)[:, None]
    base = rows[:, None].to(tl.int64) * H
    cols = tl.arange(0, 256)[None, :]
    acc = tl.zeros([R, 64, 4], dtype=tl.float32)
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0)
        if HAS_DELTA:
            x = x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
            tl.store(hidden_ptr + offs, x, mask=rmask)
        acc = acc + sumsq_vec4_chunk(x, R)
    var = lane_tree64(combine_vec4(acc, R), R) * inv_h
    rstd = libdevice.rsqrt(var + eps)[:, None]
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0)
        if HAS_DELTA:
            x = x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
        y = (x * rstd) * tl.load(w1_ptr + c * 256 + cols)
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=rmask)


def add_rmsnorm(
    residual: Any,
    delta: Any | None,
    weight_plus_one: Any,
    eps: float,
    hidden_out: Any | None = None,
    num_warps: int | None = None,
    exact: bool = True,
    rows_per_program: int = 2,
):
    """Returns (hidden FP32, normed BF16); ``weight_plus_one`` is ``1.0 + weight.float()``.

    ``exact`` (H a multiple of 256) sums squares in ATen's order, so the mean, and with it
    every output, is bit-identical to the reference; otherwise a single ``tl.sum``.
    """
    H = residual.shape[-1]
    rows = residual.numel() // H
    hidden = (
        residual
        if delta is None
        else (hidden_out if hidden_out is not None else torch.empty_like(residual))
    )
    out = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
    if exact and H % 256 == 0:
        R = rows_per_program
        _add_rmsnorm_exact_kernel[(triton.cdiv(rows, R),)](
            residual, delta if delta is not None else residual, weight_plus_one, hidden, out, rows, H,
            float(np.float32(1.0) / np.float32(H)), eps, HAS_DELTA=delta is not None, R=R,
            num_warps=num_warps or 4, **EXACT,
        )  # fmt: skip
        return hidden, out
    block = triton.next_power_of_2(H)
    warps = num_warps or (16 if block >= 8192 else 8 if block >= 2048 else 4)
    _add_rmsnorm_kernel[(rows,)](
        residual,
        delta if delta is not None else residual,
        weight_plus_one,
        hidden,
        out,
        H,
        float(np.float32(1.0) / np.float32(H)),
        eps,
        HAS_DELTA=delta is not None,
        BLOCK=block,
        num_warps=warps,
        **EXACT,
    )
    return hidden, out


@triton.jit
def _silu_mul_kernel(
    g_ptr, u_ptr, out_ptr, n_cols, g_stride, u_stride, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(g_ptr + row * g_stride + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * u_stride + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gate: Any, up: Any | None = None, block: int = 1024, num_warps: int = 4):
    """SiLU(gate) * up. With ``up=None`` ``gate`` is the merged [.., 2I] gate/up output."""
    if up is None:
        n = gate.shape[-1] // 2
        rows = gate.numel() // gate.shape[-1]
        g, u, gs, us = gate, gate[..., n:], gate.shape[-1], gate.shape[-1]
    else:
        n = gate.shape[-1]
        rows = gate.numel() // n
        g, u, gs, us = gate, up, n, n
    out = torch.empty((*gate.shape[:-1], n), dtype=torch.bfloat16, device=gate.device)
    _silu_mul_kernel[(rows, triton.cdiv(n, block))](
        g, u, out, n, gs, us, BLOCK=block, num_warps=num_warps, **EXACT
    )
    return out


@triton.jit
def _gdn_prep_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    a_ptr,
    alog_ptr,
    dtb_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    T,
    C,
    NK,
    NV,
    eps,
    DK: tl.constexpr,
    BT: tl.constexpr,
    KW: tl.constexpr,
):
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    nblk = tl.cdiv(T, BT)
    bidx = (pid_t // nblk).to(tl.int64)
    t = (pid_t % nblk) * BT + tl.arange(0, BT)
    tmask = t < T
    d = tl.arange(0, DK)
    c = j * DK + d
    row = bidx * T + t
    acc = tl.zeros([BT, DK], dtype=tl.float32)
    for w in tl.static_range(KW):
        tt = t - (KW - 1) + w
        m = (tt >= 0) & tmask
        xv = tl.load(
            x_ptr + (bidx * T + tt)[:, None] * C + c[None, :],
            mask=m[:, None],
            other=0.0,
        )
        wv = tl.load(w_ptr + c * KW + w)
        # the ROCm causal-conv1d build rounds every product and sum (no FMA): bit-exact this way
        acc = acc + wv[None, :] * xv.to(tl.float32)
    y = (acc / (1.0 + libdevice.exp(-acc))).to(tl.bfloat16)
    if j < 2 * NK:
        yf = y.to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(yf * yf, 1) + eps)
        out = (yf * rstd[:, None]).to(tl.bfloat16)
        if j < NK:
            dst = q_ptr + row[:, None] * (NK * DK) + (j * DK + d)[None, :]
        else:
            dst = k_ptr + row[:, None] * (NK * DK) + ((j - NK) * DK + d)[None, :]
        tl.store(dst, out, mask=tmask[:, None])
    else:
        hv = j - 2 * NK
        tl.store(
            v_ptr + row[:, None] * (NV * DK) + (hv * DK + d)[None, :],
            y,
            mask=tmask[:, None],
        )
        bb = tl.load(b_ptr + row * NV + hv, mask=tmask, other=0.0).to(tl.float32)
        tl.store(
            beta_ptr + row * NV + hv,
            (1.0 / (1.0 + libdevice.exp(-bb))).to(tl.bfloat16),
            mask=tmask,
        )
        s = tl.load(a_ptr + row * NV + hv, mask=tmask, other=0.0).to(
            tl.float32
        ) + tl.load(dtb_ptr + hv)
        sp = tl.where(s > 20.0, s, libdevice.log1p(libdevice.exp(s)))
        g = -libdevice.exp(tl.load(alog_ptr + hv)) * sp
        tl.store(g_ptr + row * NV + hv, g, mask=tmask)


def gdn_prep(
    mixed_qkv: Any,
    b: Any,
    a: Any,
    conv_weight: Any,
    A_log: Any,
    dt_bias: Any,
    k_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
    block_t: int = 16,
    num_warps: int = 4,
):
    """(q, k, v, g, beta) for FLA's chunk stage; q/k stay at ``k_heads`` (FLA's GVA path).

    ``mixed_qkv`` [B, T, C] BF16 (``in_proj_qkv`` output), ``conv_weight`` [C, KW] FP32.
    q/k are L2-normalised like FLA's ``use_qk_l2norm_in_kernel``.
    """
    B, T, C = mixed_qkv.shape
    nv = (C - 2 * k_heads * head_dim) // head_dim
    dev = mixed_qkv.device
    q = torch.empty(B, T, k_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty_like(q)
    v = torch.empty(B, T, nv, head_dim, dtype=torch.bfloat16, device=dev)
    g = torch.empty(B, T, nv, dtype=torch.float32, device=dev)
    beta = torch.empty(B, T, nv, dtype=torch.bfloat16, device=dev)
    grid = (B * triton.cdiv(T, block_t), 2 * k_heads + nv)
    _gdn_prep_kernel[grid](
        mixed_qkv,
        conv_weight,
        b,
        a,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        T,
        C,
        k_heads,
        nv,
        eps,
        DK=head_dim,
        BT=block_t,
        KW=conv_weight.shape[-1],
        num_warps=num_warps,
        **EXACT,
    )
    return q, k, v, g, beta


@triton.jit
def _gated_rmsnorm_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    N,
    eps,
    inv_d,
    D: tl.constexpr,
    BR: tl.constexpr,
    EXACT_SUM: tl.constexpr,
):
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    d = tl.arange(0, D)
    m = (r < N)[:, None]
    offs = r[:, None] * D + d[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if EXACT_SUM:
        var = sumsq_128(x, BR) * inv_d
    else:
        var = tl.sum(x * x, axis=1) * inv_d
    xn = (x * libdevice.rsqrt(var + eps)[:, None]).to(tl.bfloat16).to(tl.float32)
    y = tl.load(w_ptr + d)[None, :] * xn
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (z / (1.0 + libdevice.exp(-z)))
    tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=m)


def gated_rmsnorm(
    core: Any,
    z: Any,
    weight: Any,
    eps: float,
    block_rows: int = 16,
    num_warps: int = 4,
    exact: bool = True,
):
    """``exact`` (head dim 128) sums squares in ATen's order: bit-identical to the reference."""
    D = core.shape[-1]
    n = core.numel() // D
    out = torch.empty(core.shape, dtype=torch.bfloat16, device=core.device)
    _gated_rmsnorm_kernel[(triton.cdiv(n, block_rows),)](
        core,
        z,
        weight,
        out,
        n,
        eps,
        float(np.float32(1.0) / np.float32(D)),
        D=D,
        BR=block_rows,
        EXACT_SUM=exact and D == 128,
        num_warps=num_warps,
        **EXACT,
    )
    return out


@triton.jit
def _attn_prep_kernel(
    qg_ptr,
    k_ptr,
    qw1_ptr,
    kw1_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    T,
    NH,
    NKV,
    eps,
    inv_d,
    cs_bstride,
    D: tl.constexpr,
    ROT: tl.constexpr,
    EXACT_SUM: tl.constexpr,
):
    bt = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1)
    b = bt // T
    t = bt % T
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    if hid < NH:
        src = qg_ptr + bt * (NH * 2 * D) + hid * (2 * D)
        wp = qw1_ptr
        dst = qo_ptr + ((b * NH + hid) * T + t) * D
    else:
        src = k_ptr + bt * (NKV * D) + (hid - NH) * D
        wp = kw1_ptr
        dst = ko_ptr + ((b * NKV + (hid - NH)) * T + t) * D
    x = tl.load(src + d).to(tl.float32)
    if EXACT_SUM:
        ss = tl.reshape(sumsq_256(tl.reshape(x, [1, D]), 1), [])
    else:
        ss = tl.sum(x * x, axis=0)
    rstd = libdevice.rsqrt(ss * inv_d + eps)
    y = ((x * rstd) * tl.load(wp + d)).to(tl.bfloat16).to(tl.float32)
    xp = tl.load(src + partner).to(tl.float32)
    yp = ((xp * rstd) * tl.load(wp + partner)).to(tl.bfloat16).to(tl.float32)
    yp = tl.where(d < half, -yp, yp)
    rot = d < ROT
    cs = b * cs_bstride + t * ROT + d
    c = tl.load(cos_ptr + cs, mask=rot, other=1.0)
    s = tl.load(sin_ptr + cs, mask=rot, other=0.0)
    out = tl.where(rot, (y * c) + (yp * s), y)
    tl.store(dst + d, out.to(tl.bfloat16))


def attn_prep(
    q_proj_out: Any,
    k_proj_out: Any,
    q_norm_w1: Any,
    k_norm_w1: Any,
    cos: Any,
    sin: Any,
    heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
    num_warps: int = 2,
    exact: bool = True,
):
    """(q [B, H, T, D], k [B, Hkv, T, D]) BF16 for attention; ``*_w1`` are ``1 + weight`` (FP32).

    ``exact`` (head dim 256) sums squares in ATen's order: bit-identical to the reference.

    ``cos``/``sin`` are [B or 1, T, rotary_dim] FP32 (the rotary module's output).
    The gate stays a strided view of ``q_proj_out`` and v a view of ``k``'s sibling.
    """
    B, T = q_proj_out.shape[:2]
    dev = q_proj_out.device
    q = torch.empty(B, heads, T, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty(B, kv_heads, T, head_dim, dtype=torch.bfloat16, device=dev)
    rot = cos.shape[-1]
    _attn_prep_kernel[(B * T, heads + kv_heads)](
        q_proj_out,
        k_proj_out,
        q_norm_w1,
        k_norm_w1,
        cos,
        sin,
        q,
        k,
        T,
        heads,
        kv_heads,
        eps,
        float(np.float32(1.0) / np.float32(head_dim)),
        0 if cos.shape[0] == 1 else T * rot,
        D=head_dim,
        ROT=rot,
        EXACT_SUM=exact and head_dim == 256,
        num_warps=num_warps,
        **EXACT,
    )
    return q, k


@triton.jit
def _sigmoid_gate_kernel(
    a_ptr,
    g_ptr,
    out_ptr,
    T,
    H,
    sab,
    sat,
    sah,
    sgb,
    sgt,
    sgh,
    D: tl.constexpr,
    HB: tl.constexpr,
):
    bt = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    b = bt // T
    t = bt % T
    d = tl.arange(0, D)
    m = (h < H)[:, None]
    a = tl.load(
        a_ptr + b * sab + t * sat + h[:, None] * sah + d[None, :], mask=m, other=0.0
    ).to(tl.float32)
    gt = tl.load(
        g_ptr + b * sgb + t * sgt + h[:, None] * sgh + d[None, :], mask=m, other=0.0
    ).to(tl.float32)
    s = (1.0 / (1.0 + libdevice.exp(-gt))).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out_ptr + bt * (H * D) + h[:, None] * D + d[None, :],
        (a * s).to(tl.bfloat16),
        mask=m,
    )


def sigmoid_gate(
    attn_out: Any, gate: Any, heads_per_program: int = 4, num_warps: int = 4
):
    """``attn_out`` and ``gate`` are [B, T, H, D] views with unit last stride (any other strides).

    ``attn_out`` can be SDPA's output transposed to [B, T, H, D]; ``gate`` can be the
    gate half of ``q_proj_out.view(B, T, H, 2 * D)`` read in place. Returns [B, T, H*D] BF16.
    """
    B, T, H, D = attn_out.shape
    out = torch.empty(B, T, H * D, dtype=torch.bfloat16, device=attn_out.device)
    _sigmoid_gate_kernel[(B * T, triton.cdiv(H, heads_per_program))](
        attn_out,
        gate,
        out,
        T,
        H,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        D=D,
        HB=heads_per_program,
        num_warps=num_warps,
        **EXACT,
    )
    return out
