"""Runner: execute (algorithm × workload × seed) trajectories and write JSONL.

For each (algorithm, workload, seed) triple:
  1. generate the workload (deterministic given seed)
  2. reset the algorithm with n_arms / context_dim
  3. for t in [0, horizon): select -> observe reward -> update
  4. compute trajectory metrics (cumulative regret, optimal-arm rate, recovery)
  5. emit one JSONL line under results/{workload}/{algorithm}.jsonl

Output format (one JSON object per seed):

    {
      "algorithm": "ucb1",
      "workload":  "stationary_3arm",
      "seed": 0,
      "n_arms": 3,
      "horizon": 10000,
      "context_dim": null,
      "workload_sha256": "abc...",
      "cumulative_regret_final": 42.0,
      "cumulative_regret_curve": [...],          // downsampled to 200 points
      "optimal_arm_rate_final":  0.93,
      "optimal_arm_rate_curve":  [...],          // downsampled to 200 points
      "arm_pull_counts": {"0": 9300, ...},
      "select_p95_us": 12.3,
      "recovery_time": null                      // only for drift_at_t5000
    }

The downsampled curves keep file size linear in seeds (not in horizon).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import algorithms, metrics, workloads

CURVE_POINTS = 200
DRIFT_WORKLOADS = {"drift_at_t5000": 5000}


def run_one(
    algorithm_name: str,
    workload_name: str,
    seed: int,
    algorithm_kwargs: dict | None = None,
) -> dict:
    """Run one trajectory. Returns the JSON record (not yet written)."""
    workload = workloads.generate(workload_name, seed)
    kwargs = dict(algorithm_kwargs or {})
    # Pass cold-start prior hints if the workload calls for it and the algorithm supports them.
    if (
        workload_name == "cold_start_with_prior"
        and algorithm_name == "routing_sampling_py"
    ):
        kwargs.setdefault("seed_weight", 2.0)
        # Same "moderately strong prior" the Go runtime applies via QualityScore
        # (see router_learning_adaptation.go:332-335).
        kwargs.setdefault("quality_seeds", [0.85, 0.50, 0.15])
    algorithm = algorithms.build(algorithm_name, seed=seed, **kwargs)
    algorithm.reset(workload.n_arms, workload.context_dim)

    chosen: list[int] = [0] * workload.horizon
    select_durations_ns: list[int] = []
    for t in range(workload.horizon):
        ctx = workload.context_at(t)
        t0 = time.perf_counter_ns()
        arm = algorithm.select(t, ctx)
        select_durations_ns.append(time.perf_counter_ns() - t0)
        reward = workload.reward_at(t, arm)
        algorithm.update(arm, reward, ctx)
        chosen[t] = arm

    regret_curve = metrics.cumulative_regret(
        workload.rewards, workload.optimal_arms, chosen
    )
    rate_curve = metrics.optimal_arm_rate(workload.optimal_arms, chosen)

    record: dict = {
        "algorithm": algorithm_name,
        "workload": workload_name,
        "seed": seed,
        "n_arms": workload.n_arms,
        "horizon": workload.horizon,
        "context_dim": workload.context_dim,
        "workload_sha256": workload.sha256,
        "cumulative_regret_final": regret_curve[-1],
        "cumulative_regret_curve": _downsample(regret_curve, CURVE_POINTS),
        "optimal_arm_rate_final": rate_curve[-1],
        "optimal_arm_rate_curve": _downsample(rate_curve, CURVE_POINTS),
        "arm_pull_counts": _arm_pull_counts(chosen, workload.n_arms),
        "select_p95_us": _percentile_us(select_durations_ns, 0.95),
    }
    if workload_name in DRIFT_WORKLOADS:
        record["recovery_time"] = metrics.recovery_time(
            workload.optimal_arms,
            chosen,
            drift_round=DRIFT_WORKLOADS[workload_name],
        )
    return record


def run_matrix(
    algorithms_list: list[str],
    workloads_list: list[str],
    n_seeds: int,
    out_dir: Path,
    algorithm_kwargs: dict[str, dict] | None = None,
) -> Path:
    """Run the full (algorithm × workload × seed) matrix; write one JSONL per pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    algorithm_kwargs = algorithm_kwargs or {}
    for workload in workloads_list:
        wl_dir = out_dir / workload
        wl_dir.mkdir(parents=True, exist_ok=True)
        for algorithm in algorithms_list:
            jsonl = wl_dir / f"{algorithm}.jsonl"
            with jsonl.open("w") as fh:
                for seed in range(n_seeds):
                    record = run_one(
                        algorithm,
                        workload,
                        seed,
                        algorithm_kwargs=algorithm_kwargs.get(algorithm),
                    )
                    fh.write(json.dumps(record) + "\n")
    return out_dir


# ---- helpers ---------------------------------------------------------------


def _downsample(curve: list[float], n: int) -> list[float]:
    if len(curve) <= n:
        return curve
    step = len(curve) / n
    return [curve[min(int(i * step), len(curve) - 1)] for i in range(n)]


def _arm_pull_counts(chosen: list[int], n_arms: int) -> dict[str, int]:
    counts = [0] * n_arms
    for arm in chosen:
        counts[arm] += 1
    return {str(a): counts[a] for a in range(n_arms)}


def _percentile_us(durations_ns: list[int], pct: float) -> float:
    if not durations_ns:
        return 0.0
    sorted_ns = sorted(durations_ns)
    idx = int(pct * (len(sorted_ns) - 1))
    return sorted_ns[idx] / 1000.0


# ---- CLI -------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--algorithms",
        default=",".join(algorithms.names()),
        help="comma-separated algorithm names",
    )
    parser.add_argument(
        "--workloads",
        default=",".join(workloads.names()),
        help="comma-separated workload names",
    )
    parser.add_argument(
        "--seeds", type=int, default=30, help="number of seeds (default: 30)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".agent-harness/experiments/mab-comparison"),
        help="output directory for JSONL results",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    wls = [w.strip() for w in args.workloads.split(",") if w.strip()]
    out = run_matrix(algos, wls, args.seeds, args.out_dir)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
