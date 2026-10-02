"""Per-shape GEMM tuning at the released backbones' Linear shapes (node side, one GPU).

    python3 -m v2.runtime_kernels.bench_gemm --out RUN [--models vega-27b,eos-0.8b,nox-4b]
        [--rows 64,128,256,512,1024,2048] [--triton] [--names gate_up_merged,...]

Builds ``hipblaslt_search.cpp`` with hipcc (split-K variant first, plain if that fails), runs it
over every (M, N, K) of the selected backbones' per-layer GEMMs (see ``shapes.Backbone.gemms``),
and times PyTorch's own ``F.linear`` on the same shapes (hipBLASLt default heuristic, and rocBLAS)
with rotating weight copies so the 256 MB Infinity Cache cannot serve the weights. ``--triton``
adds an autotuned Triton GEMM (gfx942 config grid). Writes RUN/gemm.json and RUN/hipblaslt.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def build(out: Path) -> tuple[Path | None, dict[str, Any]]:
    src = out / "hipblaslt_search.cpp"
    src.write_text((HERE / "hipblaslt_search.cpp").read_text())
    log: dict[str, Any] = {}
    for label, flags in (("splitk", ["-DSPLITK"]), ("plain", [])):
        exe = out / f"hipblaslt_search_{label}"
        cmd = [
            "hipcc",
            "-O2",
            "-std=c++17",
            "--offload-arch=gfx942",
            *flags,
            str(src),
            "-lhipblaslt",
            "-o",
            str(exe),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log[label] = {"returncode": proc.returncode, "stderr": proc.stderr[-3000:]}
        if proc.returncode == 0:
            return exe, log
    return None, log


def torch_linear_us(torch: Any, M: int, N: int, K: int, iters: int = 50) -> float:
    """Mean µs per F.linear over back-to-back calls with rotating weights (GPU time, no syncs between)."""
    dev = torch.device("cuda")
    copies = max(1, min(64, -(-(640 << 20) // (N * K * 2))))
    ws = [torch.randn(N, K, device=dev).to(torch.bfloat16) for _ in range(copies)]
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    for i in range(3):
        torch.nn.functional.linear(x, ws[i % copies])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for i in range(iters):
        torch.nn.functional.linear(x, ws[i % copies])
    end.record()
    end.synchronize()
    us = 1000.0 * start.elapsed_time(end) / iters
    del ws
    return us


def triton_gemm(torch: Any, M: int, N: int, K: int, iters: int = 50) -> dict[str, Any]:
    from .triton_gemm import linear, matmul_kernel

    dev = torch.device("cuda")
    copies = max(1, min(64, -(-(640 << 20) // (N * K * 2))))
    ws = [torch.randn(N, K, device=dev).to(torch.bfloat16) for _ in range(copies)]
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, ws[0])
    out = linear(x, ws[0])
    for i in range(3):
        linear(x, ws[i % copies])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for i in range(iters):
        linear(x, ws[i % copies])
    end.record()
    end.synchronize()
    from .fidelity import compare

    best = getattr(matmul_kernel, "best_config", None)
    return {
        "us": 1000.0 * start.elapsed_time(end) / iters,
        "config": (
            None
            if best is None
            else {
                **best.kwargs,
                "num_warps": best.num_warps,
                "num_stages": best.num_stages,
            }
        ),
        "fidelity_vs_torch": compare(torch, ref, out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="vega-27b,eos-0.8b,nox-4b")
    ap.add_argument("--rows", default="64,128,256,512,1024,2048")
    ap.add_argument("--names", default="")
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--quick-iters", type=int, default=3)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import torch

    from .shapes import BACKBONES

    names = set(n for n in args.names.split(",") if n)
    rows = [int(r) for r in args.rows.split(",")]
    jobs: dict[tuple[int, int, int], list[str]] = {}
    for model in args.models.split(","):
        for gname, (N, K, count) in BACKBONES[model].gemms().items():
            if names and gname not in names:
                continue
            for M in rows:
                jobs.setdefault((M, N, K), []).append(f"{model}/{gname}x{count}")
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "blas_default": str(torch.backends.cuda.preferred_blas_library()),
        "shapes": {},
    }
    exe, report["build"] = build(args.out)
    shapes_file = args.out / "shapes.txt"
    shapes_file.write_text("".join(f"{m} {n} {k}\n" for (m, n, k) in jobs))
    hip_results: dict[tuple[int, int, int], Any] = {}
    if exe is not None:
        with open(args.out / "hipblaslt.jsonl", "w") as sink:
            proc = subprocess.run(
                [
                    str(exe),
                    str(shapes_file),
                    str(args.quick_iters),
                    str(args.top),
                    str(args.iters),
                ],
                stdout=sink,
                stderr=subprocess.PIPE,
                text=True,
            )
        report["search_stderr"] = proc.stderr[-2000:]
        for line in (args.out / "hipblaslt.jsonl").read_text().splitlines():
            r = json.loads(line)
            hip_results[(r["M"], r["N"], r["K"])] = r
    for (M, N, K), users in jobs.items():
        rec: dict[str, Any] = {"users": users, "hipblaslt": hip_results.get((M, N, K))}
        try:
            torch.backends.cuda.preferred_blas_library("hipblaslt")
            rec["torch_hipblaslt_us"] = torch_linear_us(torch, M, N, K, args.iters)
            torch.backends.cuda.preferred_blas_library("rocblas")
            rec["torch_rocblas_us"] = torch_linear_us(torch, M, N, K, args.iters)
            torch.backends.cuda.preferred_blas_library("hipblaslt")
        except Exception:  # noqa: BLE001
            rec["torch_error"] = traceback.format_exc()[-1000:]
        if args.triton:
            try:
                rec["triton"] = triton_gemm(torch, M, N, K, args.iters)
            except Exception:  # noqa: BLE001
                rec["triton"] = {"error": traceback.format_exc()[-1500:]}
        report["shapes"][f"M{M}/N{N}/K{K}"] = rec
        h = rec["hipblaslt"] or {}
        print(
            json.dumps(
                {
                    "shape": f"M{M}/N{N}/K{K}",
                    "torch": round(rec.get("torch_hipblaslt_us", -1), 1),
                    "rocblas": round(rec.get("torch_rocblas_us", -1), 1),
                    "heur": round(h.get("heuristic_us", -1), 1),
                    "best": round(h.get("best_us", -1), 1),
                    "splitk": [
                        round(h.get("best_splitk_us", -1), 1),
                        h.get("best_splitk"),
                    ],
                    "triton": round((rec.get("triton") or {}).get("us", -1), 1),
                }
            ),
            flush=True,
        )
        (args.out / "gemm.json").write_text(
            json.dumps(report, indent=1, sort_keys=True)
        )
    print(json.dumps({"done": str(args.out / "gemm.json")}))


if __name__ == "__main__":
    main()
