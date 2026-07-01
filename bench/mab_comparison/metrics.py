"""Metrics for MAB algorithm comparison.

- cumulative_regret(rewards, optimal_arms, chosen_arms): cumulative best-arm gap
- optimal_arm_rate(optimal_arms, chosen_arms): running fraction of optimal pulls
- recovery_time(...): rounds to re-discover optimal after a drift event
- paired_bootstrap_ci(deltas): identical to bench/grounded_fusion/metrics.py

The bootstrap CI implementation matches grounded_fusion/metrics.py to keep
PR #2269 and this harness comparable; copied (not imported) so this package
remains self-contained and can run without the grounding bench installed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def cumulative_regret(
    rewards: Sequence[Sequence[float]],
    optimal_arms: Sequence[int],
    chosen_arms: Sequence[int],
) -> list[float]:
    """Per-round cumulative regret, length T.

    regret_t = sum_{s<=t} (rewards[s][optimal_s] - rewards[s][chosen_s])

    Uses the FULL-INFORMATION reward table from the workload, so regret
    measures the algorithm's choice quality independent of reward noise on
    unchosen arms. This is the standard practice for synthetic MAB eval.
    """
    horizon = len(chosen_arms)
    if len(rewards) != horizon or len(optimal_arms) != horizon:
        raise ValueError("rewards / optimal_arms / chosen_arms length mismatch")
    cum = 0.0
    out: list[float] = []
    for t in range(horizon):
        gap = rewards[t][optimal_arms[t]] - rewards[t][chosen_arms[t]]
        cum += gap
        out.append(cum)
    return out


def optimal_arm_rate(
    optimal_arms: Sequence[int],
    chosen_arms: Sequence[int],
) -> list[float]:
    """Per-round running fraction of pulls that selected the optimal arm.

    rate_t = (count of s in [0..t] where chosen_s == optimal_s) / (t + 1)
    """
    horizon = len(chosen_arms)
    if len(optimal_arms) != horizon:
        raise ValueError("optimal_arms / chosen_arms length mismatch")
    hits = 0
    out: list[float] = []
    for t in range(horizon):
        if chosen_arms[t] == optimal_arms[t]:
            hits += 1
        out.append(hits / (t + 1))
    return out


def recovery_time(
    optimal_arms: Sequence[int],
    chosen_arms: Sequence[int],
    drift_round: int,
    window: int = 100,
    threshold: float = 0.8,
) -> int | None:
    """Rounds after `drift_round` until optimal-arm rate in a sliding window
    crosses `threshold`. Returns None if never recovered.

    Convention: returned value is `t - drift_round` where t is the first
    round at which the rolling optimal-arm rate over the trailing `window`
    rounds reaches `threshold`.
    """
    horizon = len(chosen_arms)
    if drift_round < 0 or drift_round >= horizon:
        raise ValueError("drift_round out of range")
    for t in range(drift_round + window, horizon):
        recent_hits = sum(
            1 for s in range(t - window + 1, t + 1) if chosen_arms[s] == optimal_arms[s]
        )
        if recent_hits / window >= threshold:
            return t - drift_round
    return None


def paired_bootstrap_ci(
    deltas: Sequence[float],
    iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean_delta, lo, hi) percentile bootstrap CI over per-pair deltas.

    Identical formula to bench/grounded_fusion/metrics.py — kept here as a copy
    so this package is import-self-contained.
    """
    deltas = [float(d) for d in deltas]
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(deltas) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (mean, lo, hi)
