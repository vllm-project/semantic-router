"""Pairwise comparison + KEEP/KILL verdict for MAB algorithms.

Reads JSONL files written by runner.py and produces:

  1. paired bootstrap CIs on cumulative_regret and optimal_arm_rate deltas
     (one comparison per (algorithm_x, algorithm_y, workload) triple)
  2. a typed verdict per (algorithm_x, algorithm_y) over all workloads

VERDICTS (pre-registered, applied in order):

  KEEP_X
      X has lower cumulative_regret AND higher optimal_arm_rate than Y
      across ALL workloads where the comparison is paired-feasible
      (CI excludes 0 in X's favor on every workload).

  KEEP_X_CONTEXTUAL_ONLY
      KEEP_X conditions hold ONLY on contextual_2cluster; ties on
      stationary workloads. Suitable when X is a contextual algorithm
      that tradesoff zero benefit against contextual-only signal.

  KILL_X
      X is significantly WORSE than Y on any workload (CI excludes 0
      against X), with no compensating contextual win.

  INCONCLUSIVE
      Otherwise — typically n_seeds is too small to resolve the comparison.

The protocol is identical-in-spirit to bench/grounded_fusion/compare_multiarm.py,
adapted for MAB metrics (lower-is-better regret, higher-is-better arm rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_seed(records: list[dict]) -> dict[int, dict]:
    return {r["seed"]: r for r in records}


def compare_pair(
    records_x: list[dict],
    records_y: list[dict],
    metric_key: str,
    *,
    lower_is_better: bool,
) -> dict:
    """Paired bootstrap CI for X - Y on a single metric across shared seeds."""
    bx = _by_seed(records_x)
    by = _by_seed(records_y)
    shared = sorted(set(bx) & set(by))
    if not shared:
        return {"n_paired": 0, "favorable_to_x": False, "significant": False}
    deltas = [bx[s][metric_key] - by[s][metric_key] for s in shared]
    mean, lo, hi = metrics.paired_bootstrap_ci(deltas)
    favorable_to_x = hi < 0 if lower_is_better else lo > 0
    return {
        "n_paired": len(shared),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "significant": (lo > 0) or (hi < 0),
        "favorable_to_x": favorable_to_x,
    }


def compare_all(
    results_dir: Path,
    algorithms: list[str],
    workloads: list[str],
) -> dict:
    """Build the full comparison table over (alg_x, alg_y, workload) triples."""
    loaded: dict[tuple[str, str], list[dict]] = {}
    for wl in workloads:
        for alg in algorithms:
            loaded[(alg, wl)] = _load_jsonl(results_dir / wl / f"{alg}.jsonl")

    comparisons: dict[str, dict] = {}
    for i, alg_x in enumerate(algorithms):
        for alg_y in algorithms[i + 1 :]:
            key = f"{alg_x}_vs_{alg_y}"
            comparisons[key] = {}
            for wl in workloads:
                rx = loaded[(alg_x, wl)]
                ry = loaded[(alg_y, wl)]
                if not rx or not ry:
                    comparisons[key][wl] = {"n_paired": 0}
                    continue
                comparisons[key][wl] = {
                    "cumulative_regret": compare_pair(
                        rx, ry, "cumulative_regret_final", lower_is_better=True
                    ),
                    "optimal_arm_rate": compare_pair(
                        rx, ry, "optimal_arm_rate_final", lower_is_better=False
                    ),
                }
    return comparisons


def verdict_for(
    comparisons: dict,
    alg_x: str,
    alg_y: str,
    workloads: list[str],
) -> tuple[str, str]:
    """Apply the pre-registered KEEP/KILL rule to one pair of algorithms."""
    pair_key = f"{alg_x}_vs_{alg_y}"
    pair = comparisons.get(pair_key)
    if pair is None:
        # Try the reverse direction; we don't auto-flip because the verdict
        # speaks about alg_x specifically.
        return ("INCONCLUSIVE", f"no comparison data for {alg_x} vs {alg_y}")

    # Per-workload verdicts: does X dominate, lose, or tie Y on this workload?
    per_wl: dict[str, str] = {}
    for wl in workloads:
        block = pair.get(wl, {})
        regret = block.get("cumulative_regret", {})
        rate = block.get("optimal_arm_rate", {})
        if not regret or regret.get("n_paired", 0) == 0:
            per_wl[wl] = "no_data"
            continue
        x_wins_regret = regret.get("favorable_to_x", False)
        y_wins_regret = regret.get("significant", False) and not x_wins_regret
        x_wins_rate = rate.get("favorable_to_x", False)
        y_wins_rate = rate.get("significant", False) and not x_wins_rate

        if x_wins_regret and x_wins_rate:
            per_wl[wl] = "x_wins"
        elif y_wins_regret or y_wins_rate:
            per_wl[wl] = "y_wins"
        else:
            per_wl[wl] = "tie"

    wl_results = list(per_wl.values())
    has_y_win = "y_wins" in wl_results
    contextual_x_win = per_wl.get("contextual_2cluster") == "x_wins"
    non_contextual_results = [
        v for k, v in per_wl.items() if k != "contextual_2cluster"
    ]
    only_contextual_win = contextual_x_win and all(
        v in ("tie", "no_data") for v in non_contextual_results
    )

    if has_y_win and not only_contextual_win:
        details = ", ".join(f"{wl}={v}" for wl, v in per_wl.items())
        return ("KILL_" + alg_x.upper(), details)
    if all(v == "x_wins" for v in wl_results):
        return ("KEEP_" + alg_x.upper(), f"x dominates on all workloads ({wl_results})")
    if only_contextual_win:
        return (
            "KEEP_" + alg_x.upper() + "_CONTEXTUAL_ONLY",
            f"x wins only on contextual_2cluster; ties elsewhere ({per_wl})",
        )
    return ("INCONCLUSIVE", f"per-workload: {per_wl}")


def render_summary(
    algorithms: list[str],
    workloads: list[str],
    comparisons: dict,
) -> dict:
    verdicts: dict[str, dict] = {}
    for i, alg_x in enumerate(algorithms):
        for alg_y in algorithms[i + 1 :]:
            verdict, reason = verdict_for(comparisons, alg_x, alg_y, workloads)
            verdicts[f"{alg_x}_vs_{alg_y}"] = {
                "verdict": verdict,
                "reason": reason,
            }
    return {
        "algorithms": algorithms,
        "workloads": workloads,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }


# ---- CLI -------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithms", default="ucb1,epsilon_greedy,routing_sampling_py"
    )
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
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    wls = [w.strip() for w in args.workloads.split(",") if w.strip()]
    comparisons = compare_all(args.results_dir, algos, wls)
    summary = render_summary(algos, wls, comparisons)
    text = json.dumps(summary, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
        print(f"Wrote verdict to {args.json_out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
