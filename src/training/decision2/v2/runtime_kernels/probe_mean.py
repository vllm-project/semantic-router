"""Which summation order does ATen's ``x.pow(2).mean(-1)`` use on ROCm? (diagnostic, node side)

    python3 -m v2.runtime_kernels.probe_mean --out RUN

Per hidden size, compares ATen's FP32 mean of squares bit for bit with Triton sums in several
candidate orders (vector width 2 / 4 / 8 with per-lane accumulators, or unvectorised with four
unrolled accumulators; lane tree pairing neighbours or halves), and ``torch.rsqrt`` with OCML's
``rsqrt`` on identical inputs. Writes RUN/probe_mean.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tree_neighbours(v):
    a, b = tl.split(tl.reshape(v, [32, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [16, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [8, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [4, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [2, 2]))
    v = a + b
    a, b = tl.split(tl.reshape(v, [1, 2]))
    return tl.reshape(a + b, [])


@triton.jit
def _tree_halves(v):
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 32]), (1, 0)))
    v = a + b
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 16]), (1, 0)))
    v = a + b
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 8]), (1, 0)))
    v = a + b
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 4]), (1, 0)))
    v = a + b
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 2]), (1, 0)))
    v = a + b
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 1]), (1, 0)))
    return tl.reshape(a + b, [])


@triton.jit
def _sum_kernel(
    x_ptr, out_ptr, H, VEC: tl.constexpr, UNVEC: tl.constexpr, HALVES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * H
    if UNVEC:
        # thread_reduce_impl: accumulator i of lane t takes t + 64*(i + 4*step)
        t = tl.arange(0, 64)
        a0 = tl.zeros([64], tl.float32)
        a1 = tl.zeros([64], tl.float32)
        a2 = tl.zeros([64], tl.float32)
        a3 = tl.zeros([64], tl.float32)
        for s in range(0, H // 256):
            x0 = tl.load(base + s * 256 + t)
            x1 = tl.load(base + s * 256 + 64 + t)
            x2 = tl.load(base + s * 256 + 128 + t)
            x3 = tl.load(base + s * 256 + 192 + t)
            a0 = a0 + x0 * x0
            a1 = a1 + x1 * x1
            a2 = a2 + x2 * x2
            a3 = a3 + x3 * x3
        v = ((a0 + a1) + a2) + a3
    else:
        cols = tl.arange(0, 64 * VEC)
        acc = tl.zeros([64, VEC], tl.float32)
        for c in range(0, H // (64 * VEC)):
            x = tl.load(base + c * 64 * VEC + cols)
            acc = acc + tl.reshape(x * x, [64, VEC])
        if VEC == 2:
            a0, a1 = tl.split(acc)
            v = a0 + a1
        elif VEC == 4:
            even, odd = tl.split(tl.reshape(acc, [64, 2, 2]))
            a0, a2 = tl.split(even)
            a1, a3 = tl.split(odd)
            v = ((a0 + a1) + a2) + a3
        else:
            e, o = tl.split(
                tl.reshape(acc, [64, 4, 2])
            )  # e: i even (0,2,4,6), o: i odd
            ee, eo = tl.split(
                tl.reshape(e, [64, 2, 2])
            )  # ee: i in {0,4}, eo: i in {2,6}
            oe, oo = tl.split(
                tl.reshape(o, [64, 2, 2])
            )  # oe: i in {1,5}, oo: i in {3,7}
            a0, a4 = tl.split(ee)
            a2, a6 = tl.split(eo)
            a1, a5 = tl.split(oe)
            a3, a7 = tl.split(oo)
            v = ((((((a0 + a1) + a2) + a3) + a4) + a5) + a6) + a7
    if HALVES:
        s = _tree_halves(v)
    else:
        s = _tree_neighbours(v)
    tl.store(out_ptr + row, s)


@triton.jit
def _rsqrt_kernel(v_ptr, out_ptr, eps, N, BLOCK: tl.constexpr, MODE: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    v = tl.load(v_ptr + i, mask=m, other=1.0) + eps
    if MODE == 0:
        r = libdevice.rsqrt(v)
    elif MODE == 1:
        r = tl.math.rsqrt(v)
    elif MODE == 2:
        r = 1.0 / tl.sqrt_rn(v)
    elif MODE == 3:
        r = (1.0 / libdevice.sqrt(v.to(tl.float64))).to(tl.float32)
    elif MODE == 4:
        r = libdevice.rsqrt(v.to(tl.float64)).to(tl.float32)
    else:
        r = libdevice.rsqrt(v)
        r = r * (1.5 - 0.5 * v * r * r)
    tl.store(out_ptr + i, r, mask=m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    out: dict = {}
    for H in (1024, 2048, 2560, 4096, 5120):
        x = torch.randn(8192, H, device=dev) * 3
        ref_mean = x.pow(2).mean(-1)
        inv = float(np.float32(1.0) / np.float32(H))
        for label, vec, unvec, halves in (
            ("vec4_neighbours", 4, False, False),
            ("vec4_halves", 4, False, True),
            ("vec2_neighbours", 2, False, False),
            ("vec8_neighbours", 8, False, False),
            ("unvec4_neighbours", 4, True, False),
            ("unvec4_halves", 4, True, True),
            ("vec8_halves", 8, False, True),
        ):
            s = torch.empty(8192, device=dev)
            _sum_kernel[(8192,)](
                x,
                s,
                H,
                VEC=vec,
                UNVEC=unvec,
                HALVES=halves,
                num_warps=1,
                enable_fp_fusion=False,
            )
            mean = s * inv
            out[f"H{H}/{label}"] = (
                (mean.view(torch.int32) == ref_mean.view(torch.int32))
                .double()
                .mean()
                .item()
            )
            # the factor as a division instead of a product
            out[f"H{H}/{label}/div"] = (
                ((s / H).view(torch.int32) == ref_mean.view(torch.int32))
                .double()
                .mean()
                .item()
            )
        r_ref = torch.rsqrt(ref_mean + 1e-6)
        for mode, name in (
            (0, "ocml_rsqrt"),
            (1, "tl_math_rsqrt"),
            (2, "one_over_sqrt_rn"),
            (3, "fp64_div_sqrt"),
            (4, "fp64_ocml_rsqrt"),
            (5, "ocml_rsqrt_newton"),
        ):
            r = torch.empty_like(ref_mean)
            _rsqrt_kernel[(triton.cdiv(8192, 1024),)](
                ref_mean, r, 1e-6, 8192, BLOCK=1024, MODE=mode, enable_fp_fusion=False
            )
            out[f"H{H}/rsqrt/{name}"] = (
                (r.view(torch.int32) == r_ref.view(torch.int32)).double().mean().item()
            )
    (args.out / "probe_mean.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    for key, val in out.items():
        print(f"{key:40s} {val:.6f}")


if __name__ == "__main__":
    main()
