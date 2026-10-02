"""FLA ``chunk_gated_delta_rule`` on MI325X at the released backbones' Gated DeltaNet shapes.

    python3 -m v2.runtime_kernels.bench_gdn --out RUN [--v-heads 16,32,48] [--lengths 128,...,4096]
        [--tune] [--profile]

One sequence (B=1) of T tokens, 16 key heads, ``--v-heads`` value heads, head dim 128.
Variants per shape:
    ref          the runtime's call: q/k repeated to the value-head count, L2 norm in the kernel
    gva          q/k at 16 heads (FLA's grouped-value path), L2 norm in the kernel
    gva_prenorm  q/k at 16 heads already L2-normalised (the fused prep kernel's output)
``--profile`` adds a per-kernel breakdown (torch.profiler) of ``ref``. ``--tune`` replaces
the autotune config lists of FLA's forward kernels with a wider gfx942 grid (block sizes,
num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim), re-autotunes at every shape
(FLA's autotune keys omit T, so the shipped runtime keeps whichever config its first
shape picked) and records the tuned time, the picked configs and the fidelity against
the default configs. Writes RUN/gdn.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import traceback
from pathlib import Path
from typing import Any


def fla_kernels() -> dict[str, Any]:
    from fla.modules import l2norm
    from fla.ops.common import chunk_delta_h, chunk_o
    from fla.ops.gated_delta_rule import chunk_fwd, wy_fast
    from fla.ops.utils import cumsum

    def autotuner(kernel: Any) -> Any:
        # triton.heuristics wraps the autotuner; walk .fn down to the object holding the configs
        while not hasattr(kernel, "configs"):
            kernel = kernel.fn
        return kernel

    return {
        name: autotuner(k)
        for name, k in {
            "l2norm": l2norm.l2norm_fwd_kernel,
            "cumsum": cumsum.chunk_local_cumsum_scalar_kernel,
            "kkt_solve": chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel,
            "w_u": wy_fast.recompute_w_u_fwd_kernel,
            "fwd_h": chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            "fwd_o": chunk_o.chunk_fwd_kernel_o,
        }.items()
    }


def wide_configs(triton: Any) -> dict[str, list[Any]]:
    C = triton.Config
    amd = [
        {"waves_per_eu": w, "matrix_instr_nonkdim": m}
        for w, m in itertools.product((0, 2), (16, 32))
    ]
    return {
        "kkt_solve": [
            C({"BK": bk, **x}, num_warps=nw, num_stages=ns)
            for bk in (32, 64, 128)
            for nw in (1, 2, 4, 8)
            for ns in (1, 2)
            for x in amd[:2]
        ],
        "w_u": [
            C(dict(x), num_warps=nw, num_stages=ns)
            for nw in (2, 4, 8)
            for ns in (1, 2)
            for x in amd
        ],
        "fwd_h": [
            C({"BV": bv, **x}, num_warps=nw, num_stages=ns)
            for bv in (16, 32, 64)
            for nw in (2, 4, 8)
            for ns in (1, 2)
            for x in amd
        ],
        "fwd_o": [
            C({"BK": bk, "BV": bv, **x}, num_warps=nw, num_stages=ns)
            for bk, bv in (
                (32, 32),
                (64, 32),
                (32, 64),
                (64, 64),
                (128, 64),
                (64, 128),
                (128, 128),
            )
            for nw in (2, 4, 8)
            for ns in (1, 2)
            for x in amd[:2]
        ],
    }


def describe(cfg: Any) -> dict[str, Any] | None:
    if cfg is None:
        return None
    return {
        "kwargs": dict(cfg.kwargs),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }


def make_inputs(
    torch: Any, T: int, hk: int, hv: int, dev: Any, seed: int = 0
) -> dict[str, Any]:
    g = torch.Generator(device=dev).manual_seed(seed + T * 131 + hv)
    q = torch.randn(1, T, hk, 128, generator=g, device=dev).to(torch.bfloat16)
    k = torch.randn(1, T, hk, 128, generator=g, device=dev).to(torch.bfloat16)
    v = torch.randn(1, T, hv, 128, generator=g, device=dev).to(torch.bfloat16)
    a = torch.randn(1, T, hv, generator=g, device=dev) * 2
    gate = -torch.exp(
        torch.rand(hv, generator=g, device=dev) * 2 - 1
    ) * torch.nn.functional.softplus(a)
    beta = torch.rand(1, T, hv, generator=g, device=dev).to(torch.bfloat16)
    from fla.modules.l2norm import l2norm_fwd

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": gate.float(),
        "beta": beta,
        "q_rep": q.repeat_interleave(hv // hk, 2).contiguous(),
        "k_rep": k.repeat_interleave(hv // hk, 2).contiguous(),
        "q_n": l2norm_fwd(q)[0],
        "k_n": l2norm_fwd(k)[0],
    }


def variants(x: dict[str, Any]) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as gdr

    return {
        "ref": lambda: gdr(
            x["q_rep"],
            x["k_rep"],
            x["v"],
            x["g"],
            x["beta"],
            use_qk_l2norm_in_kernel=True,
        )[0],
        "gva": lambda: gdr(
            x["q"], x["k"], x["v"], x["g"], x["beta"], use_qk_l2norm_in_kernel=True
        )[0],
        "gva_prenorm": lambda: gdr(x["q_n"], x["k_n"], x["v"], x["g"], x["beta"])[0],
    }


def profile_kernels(torch: Any, fn: Any, iters: int = 20) -> list[dict[str, Any]]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg: dict[str, list[float]] = {}
    for e in prof.events():
        if (
            e.device_type == DeviceType.CUDA
            and "emcpy" not in e.name
            and "emset" not in e.name
        ):
            agg.setdefault(e.name[:80], []).append(e.time_range.elapsed_us())
    return sorted(
        (
            {
                "kernel": k,
                "us_per_call": sum(v) / iters,
                "launches_per_call": len(v) / iters,
            }
            for k, v in agg.items()
        ),
        key=lambda r: -r["us_per_call"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v-heads", default="16,32,48")
    ap.add_argument("--lengths", default="128,256,512,1024,2048,4096")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    import torch
    import triton

    from .fidelity import compare
    from .timing import time_call, time_graph

    dev = torch.device("cuda")
    kern = fla_kernels()
    default_configs = {name: list(k.configs) for name, k in kern.items()}
    wide = wide_configs(triton)
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "default_config_counts": {k: len(v) for k, v in default_configs.items()},
        "wide_config_counts": {k: len(v) for k, v in wide.items()},
        "shapes": {},
    }
    for hv in [int(h) for h in args.v_heads.split(",")]:
        for T in [int(t) for t in args.lengths.split(",")]:
            key = f"HV{hv}/T{T}"
            rec: dict[str, Any] = {}
            x = make_inputs(torch, T, 16, hv, dev)
            fns = variants(x)
            # default configs, autotuned at this shape
            for name, k in kern.items():
                k.configs = default_configs[name]
                k.cache.clear()
            outs = {}
            for vname, fn in fns.items():
                try:
                    outs[vname] = fn()
                    rec[vname] = {
                        "eager": time_call(fn, torch, iters=args.iters),
                        "graph": time_graph(fn, torch),
                    }
                except Exception:  # noqa: BLE001
                    rec[vname] = {"error": traceback.format_exc()[-1500:]}
            rec["default_picked"] = {
                n: describe(getattr(k, "best_config", None)) for n, k in kern.items()
            }
            if "ref" in outs:
                for vname in ("gva", "gva_prenorm"):
                    if vname in outs:
                        rec[vname]["fidelity_vs_ref"] = compare(
                            torch, outs["ref"], outs[vname]
                        )
            if args.profile:
                try:
                    rec["ref_breakdown"] = profile_kernels(torch, fns["ref"])
                    rec["gva_prenorm_breakdown"] = profile_kernels(
                        torch, fns["gva_prenorm"]
                    )
                except Exception:  # noqa: BLE001
                    rec["profile_error"] = traceback.format_exc()[-1500:]
            if args.tune:
                try:
                    for name, cfgs in wide.items():
                        kern[name].configs = cfgs
                        kern[name].cache.clear()
                    tuned_out = fns["gva_prenorm"]()
                    rec["tuned"] = {
                        "eager": time_call(fns["gva_prenorm"], torch, iters=args.iters),
                        "graph": time_graph(fns["gva_prenorm"], torch),
                        "picked": {
                            n: describe(getattr(k, "best_config", None))
                            for n, k in kern.items()
                        },
                        "fidelity_vs_default": compare(
                            torch, outs["gva_prenorm"], tuned_out
                        ),
                    }
                    if args.profile:
                        rec["tuned_breakdown"] = profile_kernels(
                            torch, fns["gva_prenorm"]
                        )
                except Exception:  # noqa: BLE001
                    rec["tuned"] = {"error": traceback.format_exc()[-2000:]}
                finally:
                    for name, k in kern.items():
                        k.configs = default_configs[name]
                        k.cache.clear()
            report["shapes"][key] = rec
            summary = {
                v: round(r["eager"]["median_us"], 1)
                for v, r in rec.items()
                if isinstance(r, dict) and "eager" in r
            }
            summary.update(
                {
                    f"{v}_graph": round(r["graph"]["median_us"], 1)
                    for v, r in rec.items()
                    if isinstance(r, dict) and (r.get("graph") or {}).get("median_us")
                }
            )
            print(json.dumps({"key": key, **summary}), flush=True)
            (args.out / "gdn.json").write_text(
                json.dumps(report, indent=1, sort_keys=True, default=str)
            )
            del x, fns, outs
            torch.cuda.empty_cache()
    print(json.dumps({"done": str(args.out / "gdn.json")}))


if __name__ == "__main__":
    main()
