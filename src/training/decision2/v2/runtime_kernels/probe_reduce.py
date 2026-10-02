"""Are the exact-order RMSNorm kernels bit-identical to the reference? (node side, random inputs)

    python3 -m v2.runtime_kernels.probe_reduce --out RUN

For every hidden size of the released backbones (and head dims 128 / 256) compares the
``exact`` and the plain ``tl.sum`` variants of ``add_rmsnorm``, ``gated_rmsnorm`` and
``attn_prep`` with the reference op sequences on 4096 random rows. Writes RUN/probe_reduce.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from . import reference as ref
    from . import triton_elementwise as tk
    from .fidelity import compare

    dev = torch.device("cuda")
    torch.manual_seed(0)
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for H in (1024, 2048, 2560, 4096, 5120):
            res = torch.randn(4096, H, device=dev) * 3
            delta = (torch.randn(4096, H, device=dev) * 0.5).to(torch.bfloat16)
            w = torch.randn(H, device=dev) * 0.1
            _, n_ref = ref.add_rmsnorm(torch, res, delta, w, 1e-6)
            for exact in (True, False):
                _, n = tk.add_rmsnorm(res, delta, 1.0 + w.float(), 1e-6, exact=exact)
                out[f"add_rmsnorm/H{H}/exact={exact}"] = compare(torch, n_ref, n)
            _, n_ref0 = ref.add_rmsnorm(torch, res, None, w, 1e-6)
            _, n0 = tk.add_rmsnorm(res, None, 1.0 + w.float(), 1e-6)
            out[f"rmsnorm/H{H}/exact=True"] = compare(torch, n_ref0, n0)
        core = torch.randn(48 * 4096, 128, device=dev).to(torch.bfloat16)
        z = (torch.randn(48 * 4096, 128, device=dev) * 2).to(torch.bfloat16)
        wg = 1.0 + torch.randn(128, device=dev) * 0.1
        g_ref = ref.gated_rmsnorm(torch, core, z, wg, 1e-6)
        for exact in (True, False):
            out[f"gated_rmsnorm/D128/exact={exact}"] = compare(
                torch, g_ref, tk.gated_rmsnorm(core, z, wg, 1e-6, exact=exact)
            )
        T, nh, nkv, hd = 2048, 24, 4, 256
        qp = (torch.randn(1, T, nh * hd * 2, device=dev) * 2).to(torch.bfloat16)
        kp = (torch.randn(1, T, nkv * hd, device=dev) * 2).to(torch.bfloat16)
        vp = torch.randn(1, T, nkv * hd, device=dev).to(torch.bfloat16)
        qw, kw = torch.randn(hd, device=dev) * 0.1, torch.randn(hd, device=dev) * 0.1
        inv_freq = 1.0 / (
            10000000 ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64)
        )
        cos, sin = ref.rotary_cos_sin(
            torch, inv_freq, torch.arange(T, device=dev)[None], torch.float32
        )
        q_ref, k_ref, _, _ = ref.attn_prep(
            torch, qp, kp, vp, qw, kw, cos, sin, hd, 1e-6
        )
        for exact in (True, False):
            q, k = tk.attn_prep(
                qp,
                kp,
                1.0 + qw.float(),
                1.0 + kw.float(),
                cos,
                sin,
                nh,
                nkv,
                hd,
                1e-6,
                exact=exact,
            )
            out[f"attn_prep.q/exact={exact}"] = compare(torch, q_ref, q)
            out[f"attn_prep.k/exact={exact}"] = compare(torch, k_ref, k)
    (args.out / "probe_reduce.json").write_text(
        json.dumps(out, indent=1, sort_keys=True)
    )
    for key, val in out.items():
        print(f"{key:36s} match={val['match']:.8f} max_ulp={val['max_ulp']}")


if __name__ == "__main__":
    main()
