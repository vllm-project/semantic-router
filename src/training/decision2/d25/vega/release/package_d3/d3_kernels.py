"""Fused Triton kernels for the element-wise ops of the d3 Qwen3.5 decoder layers (ROCm gfx942).

Each kernel computes what the BF16 eager forward computes, rounding where the eager ops round: the BF16
residual adds, the norms' FP32 math rounded to BF16 at the end, SiLU / sigmoid outputs in BF16, the depthwise
convolution summed in FP64 and rounded to BF16 like MIOpen's naive kernel, RoPE products and sums each rounded
to BF16. Transcendentals call the same OCML functions as the ATen kernels, floating-point contraction is off,
the RMSNorm means are summed in the order of ATen's ROCm row reduction (one 64-lane wavefront per row, four
accumulators per lane, then a lane tree; rows of 128 use 32 lanes) and ``torch.rsqrt``, which is correctly
rounded on ROCm, is reproduced through FP64. GEMMs, attention and the gated-delta chunk kernel are unchanged.

Kernels:
    add_rmsnorm     BF16 residual add + zero-centred RMSNorm (optionally zeroing padding rows)
    silu_mul        SiLU(gate) * up
    gdn_prep        causal conv + SiLU + q / k (repeated to the value heads) / v split + beta + g
    gated_rmsnorm   Gated DeltaNet output norm with the SiLU(z) gate
    attn_prep       q / gate split, q / k RMSNorm, partial RoPE, [B, H, T, D] layout (k repeated)
    sigmoid_gate    attention output * sigmoid(gate)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

EXACT = {"enable_fp_fusion": False}


@triton.jit
def rsqrt_rn(x):
    """``torch.rsqrt`` on ROCm is correctly rounded; OCML's FP32 rsqrt is not."""
    return libdevice.rsqrt(x.to(tl.float64)).to(tl.float32)


@triton.jit
def bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def lane_tree64(v, R: tl.constexpr):
    """[R, 64] -> [R]: the shfl_down tree of offsets 1, 2, ..., 32."""
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
def sumsq_128(x, R: tl.constexpr):
    """[R, 128] -> [R]: 32 lanes of four accumulators, then a five-level lane tree."""
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
    """[R, 256] -> [R]: one four-wide load per lane, then the lane tree."""
    return lane_tree64(combine_vec4(tl.reshape(x * x, [R, 64, 4]), R), R)


@triton.jit
def _add_rmsnorm_kernel(
    res_ptr,
    delta_ptr,
    w1_ptr,
    rowmask_ptr,
    hidden_ptr,
    out_ptr,
    M,
    H,
    inv_h,
    eps,
    HAS_DELTA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    rmask = (rows < M)[:, None]
    base = rows[:, None].to(tl.int64) * H
    cols = tl.arange(0, 256)[None, :]
    acc = tl.zeros([R, 64, 4], dtype=tl.float32)
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
        if HAS_DELTA:
            x = bf16(
                x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
            )
            tl.store(hidden_ptr + offs, x.to(tl.bfloat16), mask=rmask)
        acc = acc + tl.reshape(x * x, [R, 64, 4])
    var = lane_tree64(combine_vec4(acc, R), R) * inv_h
    rstd = rsqrt_rn(var + eps)[:, None]
    if HAS_MASK:
        keep = tl.load(rowmask_ptr + rows, mask=rows < M, other=0).to(tl.float32)[
            :, None
        ]
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
        if HAS_DELTA:
            x = bf16(
                x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
            )
        y = bf16((x * rstd) * tl.load(w1_ptr + c * 256 + cols))
        if HAS_MASK:
            y = y * keep
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=rmask)


def add_rmsnorm(
    residual: Any,
    delta: Any | None,
    weight_plus_one: Any,
    eps: float,
    rowmask: Any | None = None,
) -> tuple[Any, Any]:
    """(hidden, normed) BF16: ``hidden = residual + delta`` (or ``residual``), then ``Qwen3_5RMSNorm``.

    ``weight_plus_one`` is the FP32 ``1 + w``; ``rowmask`` (one integer per row) zeroes the normed rows of
    padding as the Gated DeltaNet layer's padding multiply does. Rows of a multiple of 256.
    """
    H = residual.shape[-1]
    rows = residual.numel() // H
    hidden = residual if delta is None else torch.empty_like(residual)
    out = torch.empty_like(residual)
    _add_rmsnorm_kernel[(triton.cdiv(rows, 2),)](
        residual,
        delta if delta is not None else residual,
        weight_plus_one,
        rowmask if rowmask is not None else residual,
        hidden,
        out,
        rows,
        H,
        float(np.float32(1.0) / np.float32(H)),
        eps,
        HAS_DELTA=delta is not None,
        HAS_MASK=rowmask is not None,
        R=2,
        num_warps=4,
        **EXACT,
    )
    return hidden, out


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(g_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    s = bf16(g / (1.0 + libdevice.exp(-g)))
    tl.store(out_ptr + row * n_cols + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gate: Any, up: Any) -> Any:
    """``silu(gate) * up`` of two contiguous BF16 tensors of one shape."""
    n = gate.shape[-1]
    rows = gate.numel() // n
    out = torch.empty_like(gate)
    _silu_mul_kernel[(rows, triton.cdiv(n, 1024))](
        gate, up, out, n, BLOCK=1024, num_warps=4, **EXACT
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
    DK: tl.constexpr,
    REP: tl.constexpr,
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
    acc = tl.zeros([BT, DK], dtype=tl.float64)
    for w in tl.static_range(KW):
        tt = t - (KW - 1) + w
        m = (tt >= 0) & tmask
        xv = tl.load(
            x_ptr + (bidx * T + tt)[:, None] * C + c[None, :],
            mask=m[:, None],
            other=0.0,
        )
        wv = tl.load(w_ptr + c * KW + w)
        acc = acc + wv.to(tl.float64)[None, :] * xv.to(tl.float64)
    conv = bf16(acc.to(tl.float32))
    y = (conv / (1.0 + libdevice.exp(-conv))).to(tl.bfloat16)
    if j < NK:
        for r in tl.static_range(REP):
            tl.store(
                q_ptr + row[:, None] * (NV * DK) + ((j * REP + r) * DK + d)[None, :],
                y,
                mask=tmask[:, None],
            )
    elif j < 2 * NK:
        for r in tl.static_range(REP):
            tl.store(
                k_ptr
                + row[:, None] * (NV * DK)
                + (((j - NK) * REP + r) * DK + d)[None, :],
                y,
                mask=tmask[:, None],
            )
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
    head_dim: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """(q, k, v, g, beta) as the eager layer hands them to the chunk kernel (q / k repeated to the value heads).

    ``mixed_qkv`` is the contiguous [B, T, C] BF16 ``in_proj_qkv`` output, ``conv_weight`` the [C, KW] BF16
    depthwise filter, ``A_log`` / ``dt_bias`` FP32 copies of the parameters; q / k / v are SiLU(conv) in BF16,
    beta = sigmoid(b) in BF16 and g = -exp(A_log) * softplus(a + dt_bias) in FP32.
    """
    B, T, C = mixed_qkv.shape
    nv = (C - 2 * k_heads * head_dim) // head_dim
    dev = mixed_qkv.device
    q = torch.empty(B, T, nv, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(B, T, nv, dtype=torch.float32, device=dev)
    beta = torch.empty(B, T, nv, dtype=torch.bfloat16, device=dev)
    _gdn_prep_kernel[(B * triton.cdiv(T, 16), 2 * k_heads + nv)](
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
        DK=head_dim,
        REP=nv // k_heads,
        BT=16,
        KW=conv_weight.shape[-1],
        num_warps=4,
        **EXACT,
    )
    return q, k, v, g, beta


@triton.jit
def _gated_rmsnorm_kernel(
    x_ptr, z_ptr, w_ptr, out_ptr, N, eps, inv_d, D: tl.constexpr, BR: tl.constexpr
):
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    d = tl.arange(0, D)
    m = (r < N)[:, None]
    offs = r[:, None] * D + d[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    rstd = rsqrt_rn(sumsq_128(x, BR) * inv_d + eps)
    xn = bf16(x * rstd[:, None])
    y = bf16(tl.load(w_ptr + d).to(tl.float32)[None, :] * xn)
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (z / (1.0 + libdevice.exp(-z)))
    tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=m)


def gated_rmsnorm(core: Any, z: Any, weight: Any, eps: float) -> Any:
    """``Qwen3_5RMSNormGated`` (BF16 weight) on contiguous [N, 128] BF16 rows ``core`` and gates ``z``."""
    D = core.shape[-1]
    n = core.numel() // D
    out = torch.empty_like(core)
    _gated_rmsnorm_kernel[(triton.cdiv(n, 16),)](
        core,
        z,
        weight,
        out,
        n,
        eps,
        float(np.float32(1.0) / np.float32(D)),
        D=D,
        BR=16,
        num_warps=4,
        **EXACT,
    )
    return out


@triton.jit
def _attn_prep_kernel(
    x_ptr,
    w1_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    T,
    NH,
    eps,
    inv_d,
    cs_bstride,
    D: tl.constexpr,
    ROT: tl.constexpr,
    QW: tl.constexpr,
    REP: tl.constexpr,
):
    bt = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1)
    b = bt // T
    t = bt % T
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    src = x_ptr + bt * (NH * QW) + hid * QW
    x = tl.load(src + d).to(tl.float32)
    sumsq = sumsq_256(tl.reshape(x, [1, D]), 1)
    rstd = rsqrt_rn(tl.reshape(sumsq, []) * inv_d + eps)
    xp = tl.load(src + partner).to(tl.float32)
    y = bf16((x * rstd) * tl.load(w1_ptr + d))
    yp = bf16((xp * rstd) * tl.load(w1_ptr + partner))
    yp = tl.where(d < half, -yp, yp)
    rot = d < ROT
    cs = b * cs_bstride + t * ROT + d
    c = tl.load(cos_ptr + cs, mask=rot, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + cs, mask=rot, other=0.0).to(tl.float32)
    out = tl.where(rot, bf16(bf16(y * c) + bf16(yp * s)), y).to(tl.bfloat16)
    for r in tl.static_range(REP):
        tl.store(o_ptr + ((b * NH * REP + hid * REP + r) * T + t) * D + d, out)


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
) -> tuple[Any, Any]:
    """(q [B, H, T, D], k [B, H, T, D]) in BF16: head RMSNorms, then the partial RoPE, k repeated to H heads.

    ``q_proj_out`` [B, T, H * 2D] holds query and gate per head, ``k_proj_out`` [B, T, Hkv * D] (both
    contiguous BF16); the norms multiply by ``1 + w`` (FP32) and round to BF16 before the RoPE, whose products
    and sum round to BF16 like the eager BF16 ops; ``cos`` / ``sin`` are the contiguous [B or 1, T, rotary dim]
    BF16 rotary tables; the head dim is 256.
    """
    if head_dim != 256:
        raise ValueError("attn_prep reduces rows of 256")
    B, T = q_proj_out.shape[:2]
    dev = q_proj_out.device
    q = torch.empty(B, heads, T, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty_like(q)
    rot = cos.shape[-1]
    # One launch per tensor: Triton's AMD pointer canonicalization fails on a runtime branch between two
    # pointers when only one of their tensors fits the 2 GiB buffer range.
    for x, w1, out, n, width, rep in (
        (q_proj_out, q_norm_w1, q, heads, 2 * head_dim, 1),
        (k_proj_out, k_norm_w1, k, kv_heads, head_dim, heads // kv_heads),
    ):
        _attn_prep_kernel[(B * T, n)](
            x,
            w1,
            cos,
            sin,
            out,
            T,
            n,
            eps,
            float(np.float32(1.0) / np.float32(head_dim)),
            0 if cos.shape[0] == 1 else T * rot,
            D=head_dim,
            ROT=rot,
            QW=width,
            REP=rep,
            num_warps=2,
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
    s = bf16(1.0 / (1.0 + libdevice.exp(-gt)))
    tl.store(
        out_ptr + bt * (H * D) + h[:, None] * D + d[None, :],
        (a * s).to(tl.bfloat16),
        mask=m,
    )


def sigmoid_gate(attn_out: Any, gate: Any) -> Any:
    """``attn_out * sigmoid(gate)`` -> [B, T, H * D] BF16 from [B, T, H, D] views with unit last stride."""
    B, T, H, D = attn_out.shape
    out = torch.empty(B, T, H * D, dtype=torch.bfloat16, device=attn_out.device)
    _sigmoid_gate_kernel[(B * T, triton.cdiv(H, 4))](
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
        HB=4,
        num_warps=4,
        **EXACT,
    )
    return out
