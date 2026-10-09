"""Markdown summary of scored public-suite runs (index, areas, per-benchmark skill and coverage, cost).

    python -m d25.vega.eval.results_table --out public-results.md LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Each RUN_DIR holds the kit's ``scores.json`` and, for batched runs, ``status.json`` (wall time, GPU-hours).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AREAS = ("knowledge", "language", "retrieval", "tools", "arts")


def load(run_dir: Path) -> dict:
    scores = json.loads((run_dir / "scores.json").read_text())
    status = (
        json.loads((run_dir / "status.json").read_text())
        if (run_dir / "status.json").exists()
        else {}
    )
    meta = (
        json.loads((run_dir / "run-meta.json").read_text())
        if (run_dir / "run-meta.json").exists()
        else {}
    )
    return {"scores": scores, "status": status, "meta": meta}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--title", default="Public index 0.3, local reproduction (kit-scored)"
    )
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    a = ap.parse_args(argv)
    runs = []
    for item in a.runs:
        label, _, path = item.partition("=")
        runs.append((label, load(Path(path))))
    lines = [
        f"# {a.title}",
        "",
        "Skill is chance-corrected and coverage-adjusted, x100. Coverage = answered / requests over the index "
        "benchmarks. Wall = worker wall time of the batched run (or as noted); GPU-h includes model loading.",
        "",
        "| Model | Public index | Knowledge | Language | Retrieval | Tools | Arts | Coverage | Complete | Wall (h) | GPUs | GPU-h | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---|",
    ]
    bench_ids = None
    for label, r in runs:
        s, st, meta = r["scores"], r["status"], r["meta"]
        areas = {x["id"]: x for x in s["areas"]}
        idx = {k: v for k, v in s["index_benchmarks"].items() if v.get("in_index")}
        bench_ids = bench_ids or sorted(idx, key=int)
        req = sum(s["benchmarks"].get(k, {}).get("requests", 0) or 0 for k in idx)
        ans = sum(
            (v["coverage"] * (s["benchmarks"].get(k, {}).get("requests", 0) or 0))
            for k, v in idx.items()
        )
        wall = meta.get(
            "wall_hours",
            round(st.get("worker_wall_seconds", 0) / 3600, 2) if st else None,
        )
        gpus = meta.get("gpus", st.get("gpus"))
        gpuh = meta.get("gpu_hours", st.get("gpu_hours"))
        lines.append(
            f"| {label} | {s['decision_index']:.2f} | "
            + " | ".join(f"{100 * areas[x]['skill']:.2f}" for x in AREAS)
            + f" | {ans / req:.4f} | {'yes' if s.get('complete') else 'no'} | {wall if wall is not None else ''} | "
            f"{gpus if gpus is not None else ''} | {gpuh if gpuh is not None else ''} | {meta.get('notes', '')} |"
        )
    lines += [
        "",
        "Per-benchmark skill (coverage in parentheses when below 1):",
        "",
        "| # | Benchmark | " + " | ".join(label for label, _ in runs) + " |",
        "|---:|---|" + "---:|" * len(runs),
    ]
    names = {}
    for _, r in runs:
        for k, v in r["scores"]["benchmarks"].items():
            names.setdefault(k, v.get("dataset", k))
    for k in bench_ids or []:
        cells = []
        for _, r in runs:
            v = r["scores"]["index_benchmarks"].get(k)
            if v is None:
                cells.append("")
                continue
            cov = "" if v["coverage"] >= 0.99995 else f" ({v['coverage']:.4f})"
            cells.append(f"{100 * v['skill']:.2f}{cov}")
        lines.append(f"| {k} | {names.get(k, k)} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
