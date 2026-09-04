"""Deterministic confidence intervals over normalized execution evidence."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import _wilson, percentile
from cli.evaluation.reporting import EvaluationMetric

_BINOMIAL_METRICS = {
    "routing.coverage",
    "routing.abstention_rate",
    "routing.accuracy",
    "routing.fallback_rate",
    "model_pool.unique_win_rate",
    "joint.reliability",
    "agentic.success_rate",
    "multimodal.support_rate",
    "preference.agreement",
    "preference.propensity_coverage",
    "safety.violation_case_rate",
    "safety.block_accuracy",
    "capacity.success_rate",
    "capacity.error_rate",
}
_LOWER_CONFIDENCE_QUANTILE = 0.025
_UPPER_CONFIDENCE_QUANTILE = 0.975
_MIN_BOOTSTRAP_SAMPLE_SIZE = 2
_MIN_BOOTSTRAP_RESAMPLES = 100
_LATENCY_METRICS: dict[str, tuple[str, float]] = {
    "routing.latency_p50_ms": ("routing", 0.50),
    "routing.latency_p95_ms": ("routing", 0.95),
    "joint.latency_p95_ms": ("joint", 0.95),
    "capacity.latency_p95_ms": ("capacity", 0.95),
    "capacity.latency_p99_ms": ("capacity", 0.99),
}
_FIELD_METRICS = {
    "joint.realized_quality": ("joint", "quality"),
    "agentic.task_score": ("agentic", "quality"),
    "multimodal.quality": ("multimodal", "quality"),
}


def _quantile(values: list[float], quantile: float) -> float:
    value = percentile(values, quantile)
    if value is None:
        raise ValueError("quantile requires at least one value")
    return value


def bootstrap_interval(
    values: Iterable[float],
    statistic: Callable[[list[float]], float],
    *,
    seed: int,
    resamples: int = 1000,
) -> tuple[float, float] | None:
    """Return a deterministic percentile bootstrap interval.

    A singleton has no empirical sampling distribution and therefore does not
    receive an interval. Callers must retain its sample count explicitly.
    """

    rows = list(values)
    if len(rows) < _MIN_BOOTSTRAP_SAMPLE_SIZE or resamples < _MIN_BOOTSTRAP_RESAMPLES:
        return None
    generator = random.Random(seed)
    estimates = [
        statistic([rows[generator.randrange(len(rows))] for _ in rows])
        for _ in range(resamples)
    ]
    estimates.sort()
    return (
        _quantile(estimates, _LOWER_CONFIDENCE_QUANTILE),
        _quantile(estimates, _UPPER_CONFIDENCE_QUANTILE),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _record_values(
    records: list[ExecutionRecord], metric_id: str
) -> tuple[list[float], Callable[[list[float]], float]] | None:
    latency = _LATENCY_METRICS.get(metric_id)
    if latency is not None:
        track_id, quantile = latency
        return _latency(records, track_id), lambda rows: _quantile(rows, quantile)
    selection = _FIELD_METRICS.get(metric_id)
    if selection is not None:
        track_id, field = selection
        values = [
            float(value)
            for row in records
            if row.track_id == track_id
            and row.status != "unavailable"
            and (value := getattr(row, field)) is not None
        ]
        return values, _mean
    if metric_id == "joint.oracle_regret":
        return _joint_regrets(records, normalized=False), _mean
    if metric_id == "joint.normalized_regret":
        return _joint_regrets(records, normalized=True), _mean
    if metric_id.startswith("model_pool.arm.") and metric_id.endswith(".quality"):
        arm_id = metric_id.removeprefix("model_pool.arm.").removesuffix(".quality")
        values = [
            row.quality
            for row in records
            if row.track_id == "model_pool"
            and row.arm_id == arm_id
            and row.status != "unavailable"
            and row.quality is not None
        ]
        return values, _mean
    return None


def _latency(records: list[ExecutionRecord], track_id: str) -> list[float]:
    return [
        row.latency_ms
        for row in records
        if row.track_id == track_id
        and row.status != "unavailable"
        and row.latency_ms is not None
    ]


def _joint_regrets(records: list[ExecutionRecord], *, normalized: bool) -> list[float]:
    pool: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if (
            row.track_id == "model_pool"
            and row.status != "unavailable"
            and row.success
            and row.quality is not None
        ):
            pool[row.case_id].append(row.quality)
    values: list[float] = []
    for row in records:
        if (
            row.track_id != "joint"
            or row.status == "unavailable"
            or row.quality is None
            or not pool.get(row.case_id)
        ):
            continue
        oracle = max(pool[row.case_id])
        regret = oracle - row.quality
        if normalized:
            if oracle <= 0:
                continue
            regret /= oracle
        values.append(regret)
    return values


def attach_confidence_intervals(
    metrics: list[EvaluationMetric],
    records: list[ExecutionRecord],
    *,
    seed: int,
    resamples: int = 1000,
) -> list[EvaluationMetric]:
    """Attach only intervals whose analysis unit is represented in evidence."""

    decorated: list[EvaluationMetric] = []
    for index, metric in enumerate(metrics):
        interval: tuple[float, float] | None = None
        if (
            metric.id in _BINOMIAL_METRICS
            or (
                metric.id.startswith("model_pool.arm.")
                and metric.id.endswith(".success_rate")
            )
            or (
                metric.id.startswith("capacity.level.")
                and (
                    metric.id.endswith(".success_rate")
                    or metric.id.endswith(".error_rate")
                )
            )
        ):
            if (
                metric.value is not None
                and metric.sample_count
                and 0 <= metric.value <= 1
            ):
                successes = min(
                    metric.sample_count,
                    max(0, round(metric.value * metric.sample_count)),
                )
                interval = _wilson(successes, metric.sample_count)
        else:
            sample = _record_values(records, metric.id)
            if sample is not None:
                values, statistic = sample
                interval = bootstrap_interval(
                    values,
                    statistic,
                    seed=seed + index * 104729,
                    resamples=resamples,
                )
        decorated.append(metric.model_copy(update={"confidence_interval": interval}))
    return decorated
