"""Shared deterministic metric primitives and coverage intervals."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.reporting import EvaluationCoverage, EvaluationMetric


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: Iterable[float]) -> tuple[float | None, int]:
    rows = list(values)
    if not rows:
        return None, 0
    return sum(rows) / len(rows), len(rows)


def _sum_available(values: Iterable[float | None]) -> float | None:
    rows = [value for value in values if value is not None]
    return sum(rows) if rows else None


def _metric(
    metric_id: str,
    name: str,
    track_id: str,
    value: float | None,
    unit: str,
    direction: str,
    sample_count: int,
) -> EvaluationMetric:
    return EvaluationMetric(
        id=metric_id,
        name=name,
        track_id=track_id,
        value=value,
        unit=unit,
        direction=direction,
        sample_count=sample_count,
    )


def coverage(records: list[ExecutionRecord], total_cases: int) -> EvaluationCoverage:
    case_ids = {row.case_id for row in records if row.status != "unavailable"}
    evaluated = min(len(case_ids), total_cases)
    return _coverage_counts(evaluated, total_cases)


def aggregate_track_coverage(
    records: list[ExecutionRecord], totals: dict[str, int]
) -> EvaluationCoverage:
    """Count one evidence cell per selected track and applicable case."""

    evaluated = 0
    for track_id, total in totals.items():
        case_ids = {
            row.case_id
            for row in records
            if row.track_id == track_id and row.status != "unavailable"
        }
        evaluated += min(len(case_ids), total)
    return _coverage_counts(evaluated, sum(totals.values()))


def _coverage_counts(evaluated: int, total: int) -> EvaluationCoverage:
    fraction = evaluated / total if total else 0.0
    interval = _wilson(evaluated, total) if total else None
    return EvaluationCoverage(
        evaluated=evaluated,
        total=total,
        fraction=fraction,
        unavailable=max(total - evaluated, 0),
        confidence_level=0.95 if interval else None,
        confidence_interval=interval,
    )


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (successes / total + z * z / (2 * total)) / denominator
    margin = (
        z
        * sqrt(
            (successes / total * (1 - successes / total) + z * z / (4 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
