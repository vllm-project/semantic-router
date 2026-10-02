"""Fused Triton kernels for the element-wise ops of Qwen3.5 and Qwen3 decoder layers (ROCm gfx942).

Each kernel computes exactly what the released eager forward computes under BF16
autocast, bit for bit: FP32 residual stream and norms, BF16 wherever the eager
ops round (Linear inputs, SiLU / sigmoid outputs, the gated norm's intermediate
cast, the conv output). Transcendentals call the same OCML functions as the ATen
and causal-conv1d kernels, floating-point contraction is off (those builds do
not fuse multiply-adds), the RMSNorm means are summed in the order of ATen's
ROCm row reduction (one 64-lane wavefront per row, four accumulators per lane,
then a lane tree; rows of 128 use 32 lanes), and ``torch.rsqrt``, which is
correctly rounded on ROCm, is reproduced through FP64. That order holds for
the row counts the runtime runs (padded batches are multiples of 8 tokens);
``fast.py`` only enables these kernels on gfx942 with the verified stack.

Kernels:
    add_rmsnorm     residual add + RMSNorm (by 1 + w for Qwen3.5, w for Qwen3), BF16 Linear input
    silu_mul        SiLU(gate) * up
    gdn_prep        causal conv + SiLU + q / k / v split + beta + g
    gated_rmsnorm   Gated DeltaNet output norm with the SiLU(z) gate
    attn_prep       (Qwen3.5: q / gate split,) q / k RMSNorm, RoPE, [B, H, T, D] layout
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
def sumsq_vec4_chunk(x, R: tl.constexpr):
    """Squares of one [R, 256] chunk as the [R, 64 lanes, 4 accumulators] update."""
    return tl.reshape(x * x, [R, 64, 4])


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
    return lane_tree64(combine_vec4(sumsq_vec4_chunk(x, R), R), R)


@triton.jit
def _add_rmsnorm_kernel(
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
    rstd = rsqrt_rn(var + eps)[:, None]
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0)
        if HAS_DELTA:
            x = x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
        y = (x * rstd) * tl.load(w1_ptr + c * 256 + cols)
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=rmask)


def add_rmsnorm(
    residual: Any, delta: Any | None, weight_plus_one: Any, eps: float
) -> tuple[Any, Any]:
    """(hidden FP32, normed BF16): ``hidden = residual + delta`` (or ``residual``), RMSNorm by the FP32 weight.

    The weight is ``1 + w`` for Qwen3.5's zero-centred norm and ``w`` for Qwen3's (whose cast to the
    FP32 input dtype is a no-op).

    ``residual`` is the contiguous FP32 stream, ``delta`` a contiguous BF16 or FP32 block output,
    the hidden size a multiple of 256.
    """
    H = residual.shape[-1]
    rows = residual.numel() // H
    hidden = residual if delta is None else torch.empty_like(residual)
    out = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
    _add_rmsnorm_kernel[(triton.cdiv(rows, 2),)](
        residual,
        delta if delta is not None else residual,
        weight_plus_one,
        hidden,
        out,
        rows,
        H,
        float(np.float32(1.0) / np.float32(H)),
        eps,
        HAS_DELTA=delta is not None,
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
    s = (g / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gate: Any, up: Any) -> Any:
    """``silu(gate) * up`` of two contiguous BF16 tensors of one shape."""
    n = gate.shape[-1]
    rows = gate.numel() // n
    out = torch.empty(gate.shape, dtype=torch.bfloat16, device=gate.device)
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
        # causal-conv1d's ROCm build rounds every product and every sum (no FMA)
        acc = acc + wv[None, :] * xv.to(tl.float32)
    y = (acc / (1.0 + libdevice.exp(-acc))).to(tl.bfloat16)
    if j < NK:
        tl.store(
            q_ptr + row[:, None] * (NK * DK) + (j * DK + d)[None, :],
            y,
            mask=tmask[:, None],
        )
    elif j < 2 * NK:
        tl.store(
            k_ptr + row[:, None] * (NK * DK) + ((j - NK) * DK + d)[None, :],
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
    """(q, k, v, g, beta) as the eager layer hands them to the chunk kernel, q / k at ``k_heads``.

    ``mixed_qkv`` is the contiguous [B, T, C] BF16 ``in_proj_qkv`` output, ``conv_weight`` the
    [C, KW] FP32 depthwise filter; q / k / v are SiLU(conv) in BF16, beta = sigmoid(b) in BF16 and
    g = -exp(A_log) * softplus(a + dt_bias) in FP32.
    """
    B, T, C = mixed_qkv.shape
    nv = (C - 2 * k_heads * head_dim) // head_dim
    dev = mixed_qkv.device
    q = torch.empty(B, T, k_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty_like(q)
    v = torch.empty(B, T, nv, head_dim, dtype=torch.bfloat16, device=dev)
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
    xn = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    y = tl.load(w_ptr + d)[None, :] * xn
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (z / (1.0 + libdevice.exp(-z)))
    tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=m)


def gated_rmsnorm(core: Any, z: Any, weight: Any, eps: float) -> Any:
    """``Qwen3_5RMSNormGated`` on contiguous [N, 128] BF16 rows ``core`` and gates ``z``."""
    D = core.shape[-1]
    n = core.numel() // D
    out = torch.empty(core.shape, dtype=torch.bfloat16, device=core.device)
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
    GATED: tl.constexpr,
    ROUND_BEFORE_WEIGHT: tl.constexpr,
):
    bt = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1)
    b = bt // T
    t = bt % T
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    QW: tl.constexpr = 2 * D if GATED else D
    if hid < NH:
        src = qg_ptr + bt * (NH * QW) + hid * QW
        wp = qw1_ptr
        dst = qo_ptr + ((b * NH + hid) * T + t) * D
    else:
        src = k_ptr + bt * (NKV * D) + (hid - NH) * D
        wp = kw1_ptr
        dst = ko_ptr + ((b * NKV + (hid - NH)) * T + t) * D
    x = tl.load(src + d).to(tl.float32)
    if D == 256:
        sumsq = sumsq_256(tl.reshape(x, [1, D]), 1)
    else:
        sumsq = sumsq_128(tl.reshape(x, [1, D]), 1)
    rstd = rsqrt_rn(tl.reshape(sumsq, []) * inv_d + eps)
    xp = tl.load(src + partner).to(tl.float32)
    if ROUND_BEFORE_WEIGHT:
        # Qwen3RMSNorm: w * (x * rstd).to(bf16), kept in FP32
        y = tl.load(wp + d) * (x * rstd).to(tl.bfloat16).to(tl.float32)
        yp = tl.load(wp + partner) * (xp * rstd).to(tl.bfloat16).to(tl.float32)
    else:
        # Qwen3_5RMSNorm: ((x * rstd) * (1 + w)).to(bf16)
        y = ((x * rstd) * tl.load(wp + d)).to(tl.bfloat16).to(tl.float32)
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
    *,
    gated: bool = True,
    zero_centred: bool = True,
) -> tuple[Any, Any]:
    """(q [B, H, T, D], k [B, Hkv, T, D]) in BF16: head RMSNorms, then RoPE, as SDPA receives them.

    Qwen3.5 (``gated``, ``zero_centred``): ``q_proj_out`` [B, T, H * 2D] holds query and gate per
    head, the norms multiply by ``1 + w`` (passed as ``q_norm_w1`` / ``k_norm_w1``) and round to
    BF16 before the partial RoPE. Qwen3 (neither): ``q_proj_out`` is [B, T, H * D], the norms
    round ``x * rstd`` to BF16 and multiply by ``w`` in FP32, and the full RoPE runs in FP32.
    ``k_proj_out`` [B, T, Hkv * D] is contiguous BF16 like ``q_proj_out``; ``cos`` / ``sin`` the
    contiguous [B or 1, T, rotary dim] FP32 rotary tables; the head dim is 128 or 256.
    """
    if head_dim not in (128, 256):
        raise ValueError("attn_prep reduces rows of 128 or 256")
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
        GATED=gated,
        ROUND_BEFORE_WEIGHT=not zero_centred,
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
    s = (1.0 / (1.0 + libdevice.exp(-gt))).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out_ptr + bt * (H * D) + h[:, None] * D + d[None, :],
        (a * s).to(tl.bfloat16),
        mask=m,
    )


def sigmoid_gate(attn_out: Any, gate: Any) -> Any:
    """``attn_out * sigmoid(gate)`` -> [B, T, H * D] BF16.

    ``attn_out`` and ``gate`` are [B, T, H, D] BF16 views with unit last stride: SDPA's output and
    the gate half of the query projection, read in place.
    """
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
