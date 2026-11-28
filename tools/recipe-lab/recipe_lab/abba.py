"""AB/BA alternating schedule for paired live measurements (Sub-issue 3).

Alternating the label order per repetition spreads time-dependent drift
(warm-up, cache state, thermals) evenly across baseline and candidate
instead of systematically favouring whoever runs second. Aggregation is
descriptive only: medians and ranges, with an explicit honesty note - the
verdict does not claim statistical significance.
"""

from __future__ import annotations

import statistics
from typing import Any

BASELINE = "baseline"
CANDIDATE = "candidate"
HONESTY_NOTE = (
    "Alternating AB/BA paired trials; medians and ranges are descriptive "
    "only; statistical significance is not established."
)


def schedule(repetitions: int) -> list[tuple[str, int]]:
    """Ordered trial labels: AB, BA, AB, BA, ... (label, repetition index)."""
    if repetitions < 1:
        return []
    trials: list[tuple[str, int]] = []
    for rep in range(repetitions):
        order = (BASELINE, CANDIDATE) if rep % 2 == 0 else (CANDIDATE, BASELINE)
        trials.extend((label, rep) for label in order)
    return trials


def paired_summary(
    results: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, Any]:
    """Summarize per-label trial metrics (descriptive medians and ranges).

    `results` rows: {"label": "baseline"|"candidate", "metrics": {...}}.
    Rows missing the metric are reported as missing - the verdict layer
    turns that into INCONCLUSIVE, never into a bad number.
    """
    per_label: dict[str, list[float]] = {BASELINE: [], CANDIDATE: []}
    missing = {BASELINE: 0, CANDIDATE: 0}
    for row in results:
        label = row.get("label")
        if label not in per_label:
            continue
        value = (row.get("metrics") or {}).get(metric_key)
        if isinstance(value, (int, float)):
            per_label[label].append(float(value))
        else:
            missing[label] += 1
    summary: dict[str, Any] = {
        "metric_key": metric_key,
        "measurement_method": HONESTY_NOTE,
        "trials": {label: len(values) for label, values in per_label.items()},
        "missing": missing,
    }
    for label, values in per_label.items():
        if values:
            summary[label] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        else:
            summary[label] = None
    return summary
