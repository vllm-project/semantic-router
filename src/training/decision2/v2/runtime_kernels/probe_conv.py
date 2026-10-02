"""Which arithmetic does the ROCm causal-conv1d package use? (exactness probe, node side)

    python3 -m v2.runtime_kernels.probe_conv --out RUN

Runs Transformers' ``causal_conv1d_fn`` dispatch (the causal-conv1d package) on
random [1, T, C] BF16 inputs with FP32 weights and compares it bit for bit with
Triton variants of the same depthwise conv + SiLU: explicit FMAs or separate
multiply / add, OCML ``exp`` or the fast ``exp``, and a division or a
reciprocal multiply. Writes RUN/probe_conv.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _conv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    T,
    C,
    FMA: tl.constexpr,
    FAST_EXP: tl.constexpr,
    RECIP: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    KW: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm = t < T
    acc = tl.zeros([BT, BC], dtype=tl.float32)
    for w in tl.static_range(KW):
        tt = t - (KW - 1) + w
        xv = tl.load(
            x_ptr + tt[:, None] * C + c[None, :],
            mask=((tt >= 0) & tm)[:, None],
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(w_ptr + c * KW + w)[None, :]
        if FMA:
            acc = tl.fma(wv, xv, acc)
        else:
            acc = acc + wv * xv
    if FAST_EXP:
        e = tl.exp(-acc)
    else:
        e = libdevice.exp(-acc)
    if RECIP:
        y = acc * (1.0 / (1.0 + e))
    else:
        y = acc / (1.0 + e)
    tl.store(out_ptr + t[:, None] * C + c[None, :], y.to(tl.bfloat16), mask=tm[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from . import reference as ref
    from .fidelity import compare

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, C = 1024, 10240
    x = (torch.randn(1, T, C, device=dev) * 2).to(torch.bfloat16)
    w = torch.randn(C, 4, device=dev) * 0.3
    pkg = ref.causal_conv1d_silu(torch, x, w).contiguous()
    results = {}
    for fma, fast, recip in itertools.product(
        (True, False), (False, True), (False, True)
    ):
        out = torch.empty_like(x)
        _conv_kernel[(triton.cdiv(T, 32), C // 128)](
            x,
            w,
            out,
            T,
            C,
            FMA=fma,
            FAST_EXP=fast,
            RECIP=recip,
            BT=32,
            BC=128,
            KW=4,
            enable_fp_fusion=False,
        )
        results[f"fma={fma},fast_exp={fast},recip={recip}"] = compare(torch, pkg, out)
    # the same with fp fusion allowed (the compiler may contract the multiply-adds itself)
    out = torch.empty_like(x)
    _conv_kernel[(triton.cdiv(T, 32), C // 128)](
        x,
        w,
        out,
        T,
        C,
        FMA=False,
        FAST_EXP=False,
        RECIP=False,
        BT=32,
        BC=128,
        KW=4,
        enable_fp_fusion=True,
    )
    results["fusion_allowed"] = compare(torch, pkg, out)
    (args.out / "probe_conv.json").write_text(
        json.dumps(results, indent=1, sort_keys=True)
    )
    for k, v in results.items():
        print(k, round(v["match"], 7), v["max_ulp"])


if __name__ == "__main__":
    main()
