"""Shared deterministic metric primitives and coverage intervals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.reporting import (
    METRIC_ANALYSIS_CONTRACT_VERSION,
    EvaluationCoverage,
    EvaluationMetric,
    MetricAnalysisProvenance,
    metric_analysis_specification,
)


def canonical_ordered_float_sum(values: Iterable[float]) -> float:
    """Sum binary64 values in evidence order for cross-runtime attestation."""
    total = 0.0
    for value in values:
        total += value
    return total


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


def mean_with_count(values: Iterable[float]) -> tuple[float | None, int]:
    rows = list(values)
    if not rows:
        return None, 0
    return sum(rows) / len(rows), len(rows)


def _sum_available(values: Iterable[float | None]) -> float | None:
    rows = [value for value in values if value is not None]
    return canonical_ordered_float_sum(rows) if rows else None


def complete_sum(values: Iterable[float | None]) -> float | None:
    """Sum a ledger only when every observation carries an explicit value."""

    rows = list(values)
    if not rows or any(value is None for value in rows):
        return None
    return canonical_ordered_float_sum(value for value in rows if value is not None)


def metric_analysis_provenance(
    metric_id: str, *, observed_exclusions: int
) -> MetricAnalysisProvenance:
    """Return the registered analysis plan for one metric identifier.

    Unknown identifiers intentionally fail closed: only cataloged estimators can
    be published, and sealing compares workers to this same registry.
    """

    spec = metric_analysis_specification(metric_id)

    return MetricAnalysisProvenance(
        contract_version=METRIC_ANALYSIS_CONTRACT_VERSION,
        estimator_id=spec.estimator_id,
        estimator_version=spec.estimator_version,
        analysis_unit=spec.analysis_unit,
        cluster_unit=spec.cluster_unit,
        weighting=spec.weighting,
        missingness=spec.missingness,
        exclusion_policy=spec.exclusion_policy,
        observed_exclusions=observed_exclusions,
    )


@dataclass(frozen=True)
class MetricDraft:
    """Internal unreleased reduction, before evidence exclusions are bound.

    A draft deliberately cannot be serialized as an ``EvaluationMetric``.  The
    report dispatcher binds it to the complete normalized evidence population
    before it becomes a publishable metric, preventing a reducer from silently
    manufacturing an ``observed_exclusions=0`` claim.
    """

    id: str
    name: str
    track_id: str
    value: float | None
    unit: str
    direction: str
    sample_count: int
    confidence_interval: tuple[float, float] | None = None
    planned_analysis_units: int | None = None
    # Only the sealed model-pool reducer owns this: generic reducers must let
    # the publication boundary derive exclusions from their planned population.
    model_pool_observed_exclusions: int | None = None

    def model_copy(self, *, update: dict[str, Any]) -> MetricDraft:
        return replace(self, **update)

    def publish(self, *, unavailable_analysis_units: int) -> EvaluationMetric:
        planned = (
            self.sample_count
            if self.planned_analysis_units is None
            else self.planned_analysis_units
        )
        if planned < self.sample_count:
            raise ValueError(
                f"metric {self.id} has {self.sample_count} observed units but only "
                f"{planned} planned analysis units"
            )
        if unavailable_analysis_units < 0:
            raise ValueError(f"metric {self.id} has a negative unavailable-unit count")
        if self.model_pool_observed_exclusions is not None:
            if (
                not self.id.startswith("model_pool.")
                or self.model_pool_observed_exclusions < 0
            ):
                raise ValueError(
                    f"metric {self.id} has an invalid sealed model-pool exclusion count"
                )
            observed_exclusions = self.model_pool_observed_exclusions
        else:
            observed_exclusions = (
                planned - self.sample_count
            ) + unavailable_analysis_units
        return EvaluationMetric(
            id=self.id,
            name=self.name,
            track_id=self.track_id,
            value=self.value,
            unit=self.unit,
            direction=self.direction,
            sample_count=self.sample_count,
            confidence_interval=self.confidence_interval,
            analysis_provenance=metric_analysis_provenance(
                self.id, observed_exclusions=observed_exclusions
            ),
        )


def build_metric(
    metric_id: str,
    name: str,
    track_id: str,
    value: float | None,
    unit: str,
    direction: str,
    sample_count: int,
    *,
    planned_analysis_units: int | None = None,
) -> MetricDraft:
    return MetricDraft(
        id=metric_id,
        name=name,
        track_id=track_id,
        value=value,
        unit=unit,
        direction=direction,
        sample_count=sample_count,
        planned_analysis_units=planned_analysis_units,
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
    interval = wilson_interval(evaluated, total) if total else None
    return EvaluationCoverage(
        evaluated=evaluated,
        total=total,
        fraction=fraction,
        unavailable=max(total - evaluated, 0),
        confidence_level=0.95 if interval else None,
        confidence_interval=interval,
    )


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
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
