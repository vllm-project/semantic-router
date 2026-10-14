"""Fused Triton kernels for the element-wise ops of Qwen3.5 / Qwen3 decoder layers and ModernBERT (ROCm gfx942).

Each kernel computes exactly what the eager decoder layer computes under BF16
autocast, bit for bit: FP32 residual stream and norms, BF16 wherever the eager
ops round (Linear inputs, SiLU / sigmoid outputs, the gated norm's intermediate
cast, the conv output). A backbone whose parameters are BF16 (the Decision 1.0
decoders) streams BF16, and the kernels follow PyTorch's type rules for the
dtypes they receive: the residual sum rounds to BF16 before the norm, a BF16
gated-norm weight rounds its product, and BF16 rotary tables round each RoPE
product and the sum. Transcendentals call the same OCML functions as the ATen
and causal-conv1d kernels, floating-point contraction is off (those builds do
not fuse multiply-adds), the RMSNorm means are summed in the order of ATen's
ROCm row reduction (one 64-lane wavefront per row, four accumulators per lane,
then a lane tree; rows of 128 use 32 lanes), and ``torch.rsqrt``, which is
correctly rounded on ROCm, is reproduced through FP64. That order holds for
the row counts the engine runs (padded batches are multiples of 8 tokens); the
ROCm accelerator registers these kernels only on gfx942. The released packages'
runtime ships the same kernels (its ``decision2/fast_kernels.py``).

Kernels:
    add_rmsnorm     residual add + RMSNorm (by 1 + w for Qwen3.5, w for Qwen3), BF16 Linear input
    residual_add    the MLP residual add into the FP32 stream
    silu_mul        SiLU(gate) * up
    gdn_prep        causal conv + SiLU + q / k / v split + beta + g
    gated_rmsnorm   Gated DeltaNet output norm with the SiLU(z) gate
    attn_prep       (Qwen3.5: q / gate split,) q / k RMSNorm, RoPE, [B, H, T, D] layout
    sigmoid_gate    attention output * sigmoid(gate)
    rotary_half     ModernBERT's FP32 rotate-half rotary of q and k, [B, H, T, D] out
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
    BF16_STREAM: tl.constexpr,
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
        if BF16_STREAM:
            x = x.to(tl.float32)
        if HAS_DELTA:
            x = x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
            if BF16_STREAM:
                x = x.to(tl.bfloat16).to(tl.float32)
            tl.store(hidden_ptr + offs, x, mask=rmask)
        acc = acc + sumsq_vec4_chunk(x, R)
    var = lane_tree64(combine_vec4(acc, R), R) * inv_h
    rstd = rsqrt_rn(var + eps)[:, None]
    for c in range(0, H // 256):
        offs = base + c * 256 + cols
        x = tl.load(res_ptr + offs, mask=rmask, other=0.0)
        if BF16_STREAM:
            x = x.to(tl.float32)
        if HAS_DELTA:
            x = x + tl.load(delta_ptr + offs, mask=rmask, other=0.0).to(tl.float32)
            if BF16_STREAM:
                x = x.to(tl.bfloat16).to(tl.float32)
        y = (x * rstd) * tl.load(w1_ptr + c * 256 + cols)
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=rmask)


def add_rmsnorm(
    residual: Any, delta: Any | None, weight_plus_one: Any, eps: float
) -> tuple[Any, Any]:
    """(hidden, normed BF16): ``hidden = residual + delta`` (or ``residual``), RMSNorm by the FP32 weight.

    The weight is ``1 + w`` for Qwen3.5's zero-centred norm and ``w`` for Qwen3's (whose cast to the
    FP32 input dtype is a no-op).

    ``residual`` is the contiguous FP32 or BF16 stream (``hidden`` keeps its dtype; a BF16 sum is
    rounded before the norm reads it), ``delta`` a contiguous BF16 or FP32 block output, the hidden
    size a multiple of 256.
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
        BF16_STREAM=residual.dtype == torch.bfloat16,
        R=2,
        num_warps=4,
        **EXACT,
    )
    return hidden, out


@triton.jit
def _residual_add_kernel(
    res_ptr, delta_ptr, out_ptr, N, BF16_STREAM: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(res_ptr + offs, mask=mask, other=0.0)
    d = tl.load(delta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if BF16_STREAM:
        tl.store(out_ptr + offs, (x.to(tl.float32) + d).to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, x + d, mask=mask)


def residual_add(residual: Any, delta: Any) -> Any:
    """``residual + delta``: the contiguous FP32 or BF16 stream plus a contiguous BF16 or FP32 block output.

    One FP32 addition per element, as ATen's type-promoting add computes it (ATen's mixed-dtype
    kernel runs several times slower at some hidden sizes); a BF16 stream rounds the sum to BF16.
    """
    out = torch.empty_like(residual)
    n = residual.numel()
    _residual_add_kernel[(triton.cdiv(n, 4096),)](
        residual,
        delta,
        out,
        n,
        BF16_STREAM=residual.dtype == torch.bfloat16,
        BLOCK=4096,
        num_warps=8,
        **EXACT,
    )
    return out


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
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    N,
    eps,
    inv_d,
    D: tl.constexpr,
    BF16_WEIGHT: tl.constexpr,
    BR: tl.constexpr,
):
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    d = tl.arange(0, D)
    m = (r < N)[:, None]
    offs = r[:, None] * D + d[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    rstd = rsqrt_rn(sumsq_128(x, BR) * inv_d + eps)
    xn = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    if BF16_WEIGHT:
        # a BF16 weight times the BF16 normalized value is a BF16 product
        y = (
            (tl.load(w_ptr + d)[None, :].to(tl.float32) * xn)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        y = tl.load(w_ptr + d)[None, :] * xn
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (z / (1.0 + libdevice.exp(-z)))
    tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=m)


def gated_rmsnorm(core: Any, z: Any, weight: Any, eps: float) -> Any:
    """``Qwen3_5RMSNormGated`` on contiguous [N, 128] BF16 rows ``core`` and gates ``z`` (FP32 or BF16 weight)."""
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
        BF16_WEIGHT=weight.dtype == torch.bfloat16,
        BR=16,
        num_warps=4,
        **EXACT,
    )
    return out


@triton.jit
def _head_prep_kernel(
    src_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    dst_ptr,
    BT,
    T,
    NH,
    eps,
    inv_d,
    cs_bstride,
    D: tl.constexpr,
    ROT: tl.constexpr,
    SRC_HEAD: tl.constexpr,
    ROUND_BEFORE_WEIGHT: tl.constexpr,
    BF16_ROPE: tl.constexpr,
    R: tl.constexpr,
):
    # R tokens of one head per program; every row's arithmetic is the one-row order. One tensor per launch:
    # choosing the query or key pointer at run time trips Triton 3.7's AMD CanonicalizePointers pass when
    # only one of them exceeds 2 GiB (no 32-bit pointer range).
    bt = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    rmask = (bt < BT)[:, None]
    hid = tl.program_id(1)
    b = bt // T
    t = bt % T
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    src = src_ptr + (bt * (NH * SRC_HEAD) + hid * SRC_HEAD)[:, None]
    wp = w_ptr
    dst = dst_ptr + (((b * NH + hid) * T + t) * D)[:, None]
    x = tl.load(src + d[None, :], mask=rmask, other=0.0).to(tl.float32)
    if D == 256:
        sumsq = sumsq_256(x, R)
    else:
        sumsq = sumsq_128(x, R)
    rstd = rsqrt_rn(sumsq * inv_d + eps)[:, None]
    xp = tl.load(src + partner[None, :], mask=rmask, other=0.0).to(tl.float32)
    w = tl.load(wp + d)[None, :]
    wq = tl.load(wp + partner)[None, :]
    if ROUND_BEFORE_WEIGHT:
        # Qwen3RMSNorm: w * (x * rstd).to(bf16), kept in FP32
        y = w * (x * rstd).to(tl.bfloat16).to(tl.float32)
        yp = wq * (xp * rstd).to(tl.bfloat16).to(tl.float32)
    else:
        # Qwen3_5RMSNorm: ((x * rstd) * (1 + w)).to(bf16)
        y = ((x * rstd) * w).to(tl.bfloat16).to(tl.float32)
        yp = ((xp * rstd) * wq).to(tl.bfloat16).to(tl.float32)
    yp = tl.where(d[None, :] < half, -yp, yp)
    rot = d[None, :] < ROT
    cs = (b * cs_bstride + t * ROT)[:, None] + d[None, :]
    c = tl.load(cos_ptr + cs, mask=rot & rmask, other=1.0)
    s = tl.load(sin_ptr + cs, mask=rot & rmask, other=0.0)
    if BF16_ROPE:
        # BF16 q / k times BF16 tables: each product and the sum round to BF16
        yc = (y * c.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        ys = (yp * s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        out = tl.where(rot, yc + ys, y)
    else:
        out = tl.where(rot, (y * c) + (yp * s), y)
    tl.store(dst + d[None, :], out.to(tl.bfloat16), mask=rmask)


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
    contiguous [B or 1, T, rotary dim] rotary tables in the stream's dtype (BF16 tables round the
    RoPE products and sum to BF16); the head dim is 128 or 256.
    """
    if head_dim not in (128, 256):
        raise ValueError("attn_prep reduces rows of 128 or 256")
    B, T = q_proj_out.shape[:2]
    dev = q_proj_out.device
    q = torch.empty(B, heads, T, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.empty(B, kv_heads, T, head_dim, dtype=torch.bfloat16, device=dev)
    rot = cos.shape[-1]
    rows = 16 if head_dim == 128 else 8
    for src, weight, dst, count, width in (
        (q_proj_out, q_norm_w1, q, heads, 2 * head_dim if gated else head_dim),
        (k_proj_out, k_norm_w1, k, kv_heads, head_dim),
    ):
        _head_prep_kernel[(triton.cdiv(B * T, rows), count)](
            src,
            weight,
            cos,
            sin,
            dst,
            B * T,
            T,
            count,
            eps,
            float(np.float32(1.0) / np.float32(head_dim)),
            0 if cos.shape[0] == 1 else T * rot,
            D=head_dim,
            ROT=rot,
            SRC_HEAD=width,
            ROUND_BEFORE_WEIGHT=not zero_centred,
            BF16_ROPE=cos.dtype == torch.bfloat16,
            R=rows,
            num_warps=4,
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


@triton.jit
def _rotary_half_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    sb,
    sh,
    st,
    H,
    T,
    HALF: tl.constexpr,
    HP: tl.constexpr,
    BT: tl.constexpr,
):
    bh = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)[:, None]
    d = tl.arange(0, HP)[None, :]
    m = (t < T) & (d < HALF)
    src = (bh // H) * sb + (bh % H) * sh + t * st
    dst = (bh * T + t) * (2 * HALF)
    table = t * (2 * HALF)
    c1 = tl.load(cos_ptr + table + d, mask=m)
    c2 = tl.load(cos_ptr + table + HALF + d, mask=m)
    s1 = tl.load(sin_ptr + table + d, mask=m)
    s2 = tl.load(sin_ptr + table + HALF + d, mask=m)
    q1 = tl.load(q_ptr + src + d, mask=m).to(tl.float32)
    q2 = tl.load(q_ptr + src + HALF + d, mask=m).to(tl.float32)
    k1 = tl.load(k_ptr + src + d, mask=m).to(tl.float32)
    k2 = tl.load(k_ptr + src + HALF + d, mask=m).to(tl.float32)
    out = qo_ptr.dtype.element_ty
    tl.store(qo_ptr + dst + d, (q1 * c1 - q2 * s1).to(out), mask=m)
    tl.store(qo_ptr + dst + HALF + d, (q2 * c2 + q1 * s2).to(out), mask=m)
    tl.store(ko_ptr + dst + d, (k1 * c1 - k2 * s1).to(out), mask=m)
    tl.store(ko_ptr + dst + HALF + d, (k2 * c2 + k1 * s2).to(out), mask=m)


def rotary_half(query: Any, key: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    """``kernels.rotary_half_ref`` in one launch: contiguous ``[B, H, T, D]`` query and key.

    ``query`` and ``key`` are views with one stride set and a unit last stride
    (the halves of a fused QKV projection); other layouts run the reference.
    The FP32 products round before the sum, as the eager ``x * cos + rotate(x) * sin``.
    """
    from .kernels import rotary_half_ref

    B, H, T, D = query.shape
    if query.stride() != key.stride() or query.stride(-1) != 1 or D % 2:
        return rotary_half_ref(query, key, cos, sin)
    cos = cos.reshape(-1, D)[:T].float().contiguous()
    sin = sin.reshape(-1, D)[:T].float().contiguous()
    query_out = torch.empty((B, H, T, D), dtype=query.dtype, device=query.device)
    key_out = torch.empty_like(query_out)
    _rotary_half_kernel[(B * H, triton.cdiv(T, 64))](
        query,
        key,
        cos,
        sin,
        query_out,
        key_out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        H,
        T,
        HALF=D // 2,
        HP=triton.next_power_of_2(D // 2),
        BT=64,
        **EXACT,
    )
    return query_out, key_out
