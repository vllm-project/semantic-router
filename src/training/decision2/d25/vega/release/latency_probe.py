"""Time chosen requests repeatedly in one process under different cuDNN settings (diagnosis of latency spikes).

    python -m d25.vega.release.latency_probe --package <dir> --rows latency-760.jsonl.gz \
        --run-ids <id> <id> ... --repeat 3 --out probe.json

Without the causal-conv1d package, transformers runs the Gated DeltaNet convolution as a depthwise ``conv1d``,
which cuDNN serves on CUDA. Modes: ``default``; ``nocudnn`` (``torch.backends.cudnn.enabled = False``);
``benchmark`` (``torch.backends.cudnn.benchmark = True``). The first pass of a mode includes one-time costs,
later passes show the steady state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from d25.vega.release.sample import read_rows

MODES = ("default", "nocudnn", "benchmark")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--runtime-source",
        choices=("package", "tree"),
        default="package",
        help="tree: this source tree's package/ runtime instead of the one in the package",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    import shutil
    import tempfile

    import torch

    source = args.package
    if args.runtime_source == "tree":
        here = Path(__file__).resolve().parent
        source = Path(tempfile.mkdtemp(prefix="runtime-"))
        shutil.copyfile(
            here / "package" / "decision25_runtime.py", source / "decision25_runtime.py"
        )
        shutil.copyfile(
            here.parent / "common" / "decision_format.py",
            source / "decision25_format.py",
        )
    sys.path.insert(0, str(source))
    import decision25_runtime as rt

    rows = {r["_evaluation"]["run_id"]: r for r in read_rows(args.rows)}
    model = rt.Decision25.from_pretrained(args.package, device=args.device)
    warmup = model.warmup()
    report = {"warmup_seconds": round(warmup, 1), "kernels": model.kernels, "modes": {}}
    for mode in args.modes:
        torch.backends.cudnn.enabled = mode != "nocudnn"
        torch.backends.cudnn.benchmark = mode == "benchmark"
        timings = {}
        for rid in args.run_ids:
            row = rows[rid]
            times = []
            for _ in range(args.repeat):
                model.synchronize()
                started = time.perf_counter()
                model.system_one(state=row["state"], questions=row["questions"])
                model.synchronize()
                times.append(round((time.perf_counter() - started) * 1000, 1))
            timings[rid] = times
            print(json.dumps({"mode": mode, "run_id": rid, "ms": times}), flush=True)
        report["modes"][mode] = timings
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
