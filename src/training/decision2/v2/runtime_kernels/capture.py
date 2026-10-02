"""Real-activation fidelity and end-to-end answers of the fused kernels on a released package.

    python3 -m v2.runtime_kernels.capture --out RUN --package PKG [--base-path BASE]
        --prompts PANEL.prompts.jsonl [--count N] [--modes replay,shadow,fused] [--timed]

Loads the package through its own runtime (``decision2.Decision2``), answers each prompt
with the shipped forward, then again under ``integrate.FusedLayers`` in each mode:
``replay`` must give bit-identical answers (it validates the reference replay), ``shadow``
records per-kernel fidelity on the real activations of every layer, ``fused`` measures the
end-to-end effect with the release parity definitions (answer changes, largest drift of any
reported probability). ``--timed`` records per-request latency (eager, synchronised) for the
shipped forward and the fused one. Prompt text stays on the node; RUN/capture.json holds
numbers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--base-path")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--modes", default="replay,shadow,fused")
    ap.add_argument("--timed", action="store_true")
    args = ap.parse_args()

    import torch

    from v2.release.examples import category, compare_answers, numbers  # noqa: F401

    from .integrate import FusedLayers, Stats

    sys.path.insert(0, str(args.package.resolve()))
    from decision2 import Decision2

    rows = [
        json.loads(line)
        for line in args.prompts.read_text().splitlines()
        if line.strip()
    ]
    rows = rows[args.offset : args.offset + args.count]
    t0 = time.perf_counter()
    model = Decision2.from_pretrained(
        args.package, device="cuda:0", base_path=args.base_path, threads=4
    )
    load_s = time.perf_counter() - t0
    backbone = model.backend.model.backbone

    def answer(p: dict[str, Any]) -> dict[str, Any]:
        return model.system_one(state=p["state"], questions=p["questions"])["answers"]

    def run(label: str) -> tuple[list[Any], list[float]]:
        outs, lat = [], []
        for p in rows:
            torch.cuda.synchronize()
            s = time.perf_counter()
            outs.append(answer(p))
            torch.cuda.synchronize()
            lat.append(1000 * (time.perf_counter() - s))
        return outs, lat

    report: dict[str, Any] = {
        "package_manifest_sha256": hashlib.sha256(
            (args.package / "MODEL_MANIFEST.json").read_bytes()
        ).hexdigest(),
        "model_name": model.model_name,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "count": len(rows),
        "load_seconds": load_s,
        "modes": {},
    }
    with torch.inference_mode():
        run("warm")
        ref_out, ref_lat = run("ref")
        if args.timed:
            ref_out2, ref_lat = run("ref")
            report["ref_repeat_identical"] = canonical(ref_out2) == canonical(ref_out)
        report["ref_latency_ms"] = {
            "median": statistics.median(ref_lat),
            "mean": statistics.mean(ref_lat),
        }
        for mode in [m for m in args.modes.split(",") if m]:
            stats = Stats()
            with FusedLayers(backbone, mode, stats):
                if mode == "fused" and args.timed:
                    run("warm")
                outs, lat = run(mode)
            tot = {
                "slots": 0,
                "category_changes": 0,
                "missing": 0,
                "max_abs_drift": 0.0,
            }
            identical = 0
            drifts = []
            for left, right in zip(ref_out, outs):
                c = compare_answers(left, right)
                identical += canonical(left) == canonical(right)
                for k in ("slots", "category_changes", "missing"):
                    tot[k] += c[k]
                tot["max_abs_drift"] = max(tot["max_abs_drift"], c["max_abs_drift"])
                drifts.append(c["max_abs_drift"])
            rec = {
                "vs_ref": {
                    **tot,
                    "bit_identical_requests": identical,
                    "requests": len(rows),
                },
                "drift_p50": statistics.median(drifts) if drifts else None,
                "drift_p95": (
                    sorted(drifts)[int(0.95 * (len(drifts) - 1))] if drifts else None
                ),
                "latency_ms": {
                    "median": statistics.median(lat),
                    "mean": statistics.mean(lat),
                },
            }
            if mode == "shadow":
                rec["kernels"] = stats.summary()
            report["modes"][mode] = rec
            print(
                json.dumps(
                    {"mode": mode, **rec["vs_ref"], "lat_ms": rec["latency_ms"]}
                ),
                flush=True,
            )
            (args.out / "capture.json").write_text(
                json.dumps(report, indent=1, sort_keys=True)
            )
    (args.out / "capture.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"done": str(args.out / "capture.json")}))


if __name__ == "__main__":
    main()
