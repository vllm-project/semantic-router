"""Summary statistics and 95% t-intervals for repeated A/B measurements."""

from __future__ import annotations

import statistics
from typing import Any

T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}

MIN_INTERVAL_ROUNDS = 2
MIN_ACCEPTANCE_ROUNDS = 5
HIGHER_IS_BETTER = ("requests_per_second", "throughput_rps")


def t_interval(diffs: list[float]) -> dict[str, Any]:
    """Mean of ``diffs`` and, from two rounds, a 95% Student-t interval."""
    if not diffs:
        raise ValueError("at least one measurement round is required")
    mean = statistics.fmean(diffs)
    n = len(diffs)
    if n < MIN_INTERVAL_ROUNDS:
        return {
            "diff": mean,
            "low": None,
            "high": None,
            "rounds": n,
            "verdict": "Insufficient",
        }
    df = n - 1
    t = T95[max(k for k in T95 if k <= df)]
    half = t * statistics.stdev(diffs) / n**0.5
    return {
        "diff": mean,
        "low": mean - half,
        "high": mean + half,
        "rounds": n,
    }


def verdict(interval: dict[str, Any]) -> str:
    """Better/Worse/Level only from five rounds with a real interval."""
    n = int(interval.get("rounds") or 0)
    if n < MIN_ACCEPTANCE_ROUNDS or interval.get("low") is None:
        return "Insufficient"
    if interval["high"] < 0:
        return "Better"
    if interval["low"] > 0:
        return "Worse"
    return "Level"


def inconclusive_worse(interval: dict[str, Any]) -> bool:
    """Point estimate is worse and a five-round interval still straddles zero."""
    return (
        verdict(interval) == "Level"
        and interval["diff"] > 0
        and interval.get("low") is not None
    )


def minus_baseline(
    side_rounds: list[dict[str, float]],
    baseline_rounds: list[dict[str, float]],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Per metric, side minus baseline over aligned rounds, with verdicts."""
    if len(side_rounds) != len(baseline_rounds):
        raise ValueError("benchmark sides must have the same number of rounds")
    n = len(side_rounds)
    if not n:
        raise ValueError("at least one measurement round is required")
    out: dict[str, Any] = {"rounds": n}
    for metric in metrics:
        diffs = [side_rounds[i][metric] - baseline_rounds[i][metric] for i in range(n)]
        entry = t_interval(diffs)
        entry["baseline"] = statistics.fmean(
            baseline_rounds[i][metric] for i in range(n)
        )
        label = verdict(entry)
        if metric in HIGHER_IS_BETTER and label in {"Better", "Worse"}:
            label = {"Better": "Worse", "Worse": "Better"}[label]
        entry["verdict"] = label
        rounded: dict[str, Any] = {}
        for key, value in entry.items():
            if isinstance(value, float):
                rounded[key] = round(value, 6)
            else:
                rounded[key] = value
        rounded["rounds"] = n
        out[metric] = rounded
    return out


def should_extend(intervals: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """Add rounds when evidence is short of five, or worse-at-the-point and Level."""
    for key in keys:
        if key not in intervals:
            continue
        entry = intervals[key]
        if entry.get("verdict") == "Insufficient":
            return True
        direction = -1 if key in HIGHER_IS_BETTER else 1
        if verdict(entry) == "Level" and direction * entry["diff"] > 0:
            return True
    return False


def summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "requests": len(ordered),
        "p50_ms": 1000 * statistics.median(ordered),
        "p95_ms": 1000
        * ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "mean_ms": 1000 * statistics.fmean(ordered),
        "requests_per_second": len(ordered) / sum(ordered),
    }


def paired_difference(path: list[float], base: list[float]) -> dict[str, float]:
    """Median and mean of the per-request differences ``path - base``, in ms."""
    differences = [1000 * (a - b) for a, b in zip(path, base, strict=True)]
    return {
        "p50": statistics.median(differences),
        "mean": statistics.fmean(differences),
    }
