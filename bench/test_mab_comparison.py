"""Unit tests for bench.mab_comparison.

Standard library + pytest only; no router runtime imports. Mirrors the
testing posture of bench/test_agentic_routing_experiment.py.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# Make sure the package is importable when running from repo root via pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

algorithms = importlib.import_module("bench.mab_comparison.algorithms")
compare = importlib.import_module("bench.mab_comparison.compare")
metrics = importlib.import_module("bench.mab_comparison.metrics")
runner = importlib.import_module("bench.mab_comparison.runner")
workloads = importlib.import_module("bench.mab_comparison.workloads")


# ---- workload determinism --------------------------------------------------


def test_workload_determinism_same_seed():
    a = workloads.generate("stationary_3arm", seed=42)
    b = workloads.generate("stationary_3arm", seed=42)
    assert a.sha256 == b.sha256
    assert a.rewards == b.rewards
    assert a.optimal_arms == b.optimal_arms


def test_workload_determinism_different_seeds_diverge():
    a = workloads.generate("stationary_3arm", seed=42)
    b = workloads.generate("stationary_3arm", seed=43)
    assert a.sha256 != b.sha256


def test_drift_workload_optimal_changes_at_t5000():
    w = workloads.generate("drift_at_t5000", seed=0)
    # Before drift: arm 0 is optimal (mean 0.9 vs 0.5/0.1)
    assert w.optimal_arms[0] == 0
    # After drift: arm 2 is optimal (mean 0.9 vs 0.5/0.1)
    assert w.optimal_arms[5001] == 2


def test_contextual_workload_carries_context_dim():
    w = workloads.generate("contextual_2cluster", seed=0)
    assert w.context_dim == 2
    assert w.context_at(0) is not None
    assert len(w.context_at(0)) == 2


# ---- UCB1 convergence (paper-benchmark-style) ------------------------------


def test_ucb1_converges_on_stationary_3arm():
    """Mirrors paper_benchmark_test.go TestRouterR1_ThompsonSamplingConvergence:
    a 3-arm bandit with true means [0.9, 0.5, 0.1] should be solved by UCB1.
    """
    record = runner.run_one("ucb1", "stationary_3arm", seed=0)
    # After 10000 rounds, the optimal-arm rate should be high.
    assert record["optimal_arm_rate_final"] > 0.95
    # Arm 0 must be pulled the most.
    counts = record["arm_pull_counts"]
    pulls = [counts[str(i)] for i in range(3)]
    assert pulls[0] == max(pulls)


def test_routing_sampling_py_converges_on_stationary_3arm():
    record = runner.run_one("routing_sampling_py", "stationary_3arm", seed=0)
    # Routing sampling (Beta-TS) should also identify arm 0 as optimal.
    assert record["optimal_arm_rate_final"] > 0.90
    counts = record["arm_pull_counts"]
    pulls = [counts[str(i)] for i in range(3)]
    assert pulls[0] == max(pulls)


# ---- routing_sampling_py score formula parity ------------------------------


def test_routing_sampling_py_score_no_signals_equals_mean():
    """With zero EWMAs, no costs, no observations -> score equals Beta mean."""
    alg = algorithms.RoutingSamplingPy(use_sampling=False)
    alg.reset(n_arms=3, context_dim=None)
    # Initial state: alpha=1, beta=1 -> mean=0.5 for every arm.
    for arm in range(3):
        score = alg._score(arm)
        assert abs(score - 0.5) < 1e-12


def test_routing_sampling_py_score_after_winners():
    """After 10 wins on arm 0 and 10 losses on arm 1 -> arm 0 score > arm 1."""
    alg = algorithms.RoutingSamplingPy(use_sampling=False)
    alg.reset(n_arms=2, context_dim=None)
    for _ in range(10):
        alg.update(0, 1.0, None)
        alg.update(1, 0.0, None)
    s0 = alg._score(0)
    s1 = alg._score(1)
    assert s0 > s1, f"expected arm 0 > arm 1; got {s0} vs {s1}"


def test_routing_sampling_py_seed_weight_warm_start():
    """Setting seed_weight=10 with quality_seeds shifts initial mean strongly."""
    alg = algorithms.RoutingSamplingPy(
        use_sampling=False, seed_weight=10, quality_seeds=[0.9, 0.1]
    )
    alg.reset(n_arms=2, context_dim=None)
    s0 = alg._score(0)
    s1 = alg._score(1)
    # alpha_0 = 10*0.9 + 0 + 1 = 10; beta_0 = 10*0.1 + 0 + 1 = 2; mean = 10/12 ≈ 0.833
    # alpha_1 = 10*0.1 + 0 + 1 = 2;  beta_1 = 10*0.9 + 0 + 1 = 10; mean = 2/12 ≈ 0.167
    assert abs(s0 - (10 / 12)) < 1e-9
    assert abs(s1 - (2 / 12)) < 1e-9


# ---- bootstrap CI ----------------------------------------------------------


def test_paired_bootstrap_ci_zero_mean_includes_zero():
    deltas = [-1.0, 1.0, -0.5, 0.5] * 10
    mean, lo, hi = metrics.paired_bootstrap_ci(deltas, iters=2000, seed=0)
    assert lo < 0 < hi
    assert abs(mean) < 0.2


def test_paired_bootstrap_ci_positive_mean_excludes_zero():
    deltas = [1.0] * 100
    mean, lo, hi = metrics.paired_bootstrap_ci(deltas, iters=2000, seed=0)
    assert lo > 0
    assert abs(mean - 1.0) < 1e-9


# ---- recovery time ---------------------------------------------------------


def test_recovery_time_returns_positive_int_for_drift():
    record = runner.run_one("ucb1", "drift_at_t5000", seed=0)
    # UCB1 should eventually recover after drift; result is None only if it fails
    # to reach 80% optimal-arm rate by horizon.
    rec = record["recovery_time"]
    assert rec is None or rec > 0


# ---- comparison + verdict --------------------------------------------------


def test_compare_pair_paired_seeds_only(tmp_path: Path):
    # Construct minimal JSONL fixtures.
    rx = [{"seed": 0, "cumulative_regret_final": 10.0}, {"seed": 1, "cumulative_regret_final": 12.0}]
    ry = [{"seed": 0, "cumulative_regret_final": 20.0}, {"seed": 1, "cumulative_regret_final": 22.0}]
    out = compare.compare_pair(rx, ry, "cumulative_regret_final", lower_is_better=True)
    assert out["n_paired"] == 2
    assert out["mean_delta"] < 0  # X is lower -> favorable
    assert out["favorable_to_x"] is True


def test_verdict_keep_when_x_dominates_all_workloads():
    # Forge a comparison table where alg_x dominates on every workload.
    comparisons = {
        "ucb1_vs_routing_sampling_py": {
            wl: {
                "cumulative_regret": {
                    "n_paired": 30,
                    "mean_delta": -10.0,
                    "ci95": [-15.0, -5.0],
                    "significant": True,
                    "favorable_to_x": True,
                },
                "optimal_arm_rate": {
                    "n_paired": 30,
                    "mean_delta": 0.05,
                    "ci95": [0.02, 0.08],
                    "significant": True,
                    "favorable_to_x": True,
                },
            }
            for wl in ["stationary_3arm", "drift_at_t5000", "contextual_2cluster"]
        }
    }
    verdict, _ = compare.verdict_for(
        comparisons,
        "ucb1",
        "routing_sampling_py",
        ["stationary_3arm", "drift_at_t5000", "contextual_2cluster"],
    )
    assert verdict == "KEEP_UCB1"


def test_verdict_kill_when_x_loses_any_workload():
    comparisons = {
        "ucb1_vs_routing_sampling_py": {
            "stationary_3arm": {
                "cumulative_regret": {
                    "n_paired": 30,
                    "mean_delta": 5.0,
                    "ci95": [3.0, 7.0],
                    "significant": True,
                    "favorable_to_x": False,
                },
                "optimal_arm_rate": {
                    "n_paired": 30,
                    "mean_delta": -0.05,
                    "ci95": [-0.08, -0.02],
                    "significant": True,
                    "favorable_to_x": False,
                },
            }
        }
    }
    verdict, _ = compare.verdict_for(
        comparisons,
        "ucb1",
        "routing_sampling_py",
        ["stationary_3arm"],
    )
    assert verdict == "KILL_UCB1"


def test_verdict_inconclusive_when_ties():
    comparisons = {
        "ucb1_vs_routing_sampling_py": {
            "stationary_3arm": {
                "cumulative_regret": {
                    "n_paired": 30,
                    "mean_delta": 0.5,
                    "ci95": [-2.0, 3.0],
                    "significant": False,
                    "favorable_to_x": False,
                },
                "optimal_arm_rate": {
                    "n_paired": 30,
                    "mean_delta": 0.0,
                    "ci95": [-0.02, 0.02],
                    "significant": False,
                    "favorable_to_x": False,
                },
            }
        }
    }
    verdict, _ = compare.verdict_for(
        comparisons,
        "ucb1",
        "routing_sampling_py",
        ["stationary_3arm"],
    )
    assert verdict == "INCONCLUSIVE"


# ---- end-to-end smoke ------------------------------------------------------


def test_end_to_end_smoke(tmp_path: Path):
    """One algorithm × one workload × 3 seeds; full runner+compare path."""
    out = runner.run_matrix(
        algorithms_list=["ucb1"],
        workloads_list=["stationary_3arm"],
        n_seeds=3,
        out_dir=tmp_path,
    )
    jsonl = out / "stationary_3arm" / "ucb1.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["algorithm"] == "ucb1"
        assert record["workload"] == "stationary_3arm"
        assert record["n_arms"] == 3
        assert record["horizon"] == 10000
        assert isinstance(record["cumulative_regret_final"], float)
