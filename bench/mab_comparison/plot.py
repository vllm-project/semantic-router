"""Plotting (matplotlib required) for the MAB comparison harness.

Reads runner.py JSONL output and writes:

  cumulative_regret_<workload>.png   curves with seed-aggregated mean ± std
  optimal_arm_rate_<workload>.png    same shape
  recovery_drift_at_t5000.png        bar chart of recovery_time per algorithm

Plotting is OPTIONAL — the runner and compare scripts are matplotlib-free.
This module imports matplotlib lazily so the rest of the package stays
importable on hosts without it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aggregate_curves(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    if not records:
        return [], []
    horizon = len(records[0][key])
    by_step: list[list[float]] = [[] for _ in range(horizon)]
    for r in records:
        curve = r[key]
        for t, v in enumerate(curve):
            by_step[t].append(v)
    means = [mean(col) for col in by_step]
    stds = [stdev(col) if len(col) > 1 else 0.0 for col in by_step]
    return means, stds


def plot_workload(
    workload: str,
    algorithms: list[str],
    results_dir: Path,
    out_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt  # lazy import

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for metric_key, label, fname in (
        ("cumulative_regret_curve", "Cumulative regret", "cumulative_regret"),
        ("optimal_arm_rate_curve", "Optimal arm rate", "optimal_arm_rate"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for alg in algorithms:
            records = _load_jsonl(results_dir / workload / f"{alg}.jsonl")
            if not records:
                continue
            means, stds = _aggregate_curves(records, metric_key)
            xs = list(range(len(means)))
            ax.plot(xs, means, label=alg)
            lower = [m - s for m, s in zip(means, stds, strict=True)]
            upper = [m + s for m, s in zip(means, stds, strict=True)]
            ax.fill_between(xs, lower, upper, alpha=0.15)
        ax.set_xlabel("downsampled round")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — {workload}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{fname}_{workload}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written


def plot_recovery(
    algorithms: list[str],
    results_dir: Path,
    out_dir: Path,
    workload: str = "drift_at_t5000",
) -> Path | None:
    import matplotlib.pyplot as plt  # lazy import

    out_dir.mkdir(parents=True, exist_ok=True)
    means: list[float] = []
    labels: list[str] = []
    for alg in algorithms:
        records = _load_jsonl(results_dir / workload / f"{alg}.jsonl")
        if not records:
            continue
        recoveries = [r["recovery_time"] for r in records if r.get("recovery_time") is not None]
        if not recoveries:
            continue
        labels.append(alg)
        means.append(mean(recoveries))

    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means)
    ax.set_ylabel("recovery time (rounds, lower is better)")
    ax.set_title(f"Recovery after drift — {workload}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"recovery_{workload}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---- CLI -------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--algorithms", default="ucb1,epsilon_greedy,routing_sampling_py")
    parser.add_argument(
        "--workloads",
        default=",".join(
            [
                "stationary_3arm",
                "stationary_5arm_close",
                "drift_at_t5000",
                "gradual_drift",
                "contextual_2cluster",
                "cold_start_with_prior",
            ]
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    wls = [w.strip() for w in args.workloads.split(",") if w.strip()]
    written: list[Path] = []
    for wl in wls:
        written.extend(plot_workload(wl, algos, args.results_dir, args.out_dir))
    rec = plot_recovery(algos, args.results_dir, args.out_dir)
    if rec is not None:
        written.append(rec)
    print(f"Wrote {len(written)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
