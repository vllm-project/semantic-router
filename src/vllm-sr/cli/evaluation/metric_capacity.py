"""Metrics for bounded multi-level live capacity sweeps."""

from __future__ import annotations

from collections import defaultdict

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import _mean, _metric, _sum_available, percentile
from cli.evaluation.reporting import EvaluationMetric

_SATURATION_THROUGHPUT_RATIO = 0.95
_SATURATION_ERROR_RATE = 0.05


def _level_metrics(
    by_level: dict[int, list[ExecutionRecord]],
) -> list[EvaluationMetric]:
    metrics: list[EvaluationMetric] = []
    for concurrency, rows in sorted(by_level.items()):
        prefix = f"capacity.level.{concurrency}"
        latency = [row.latency_ms for row in rows if row.latency_ms is not None]
        success, count = _mean(
            float(bool(row.success)) for row in rows if row.success is not None
        )
        throughput = max(
            (row.throughput_rps for row in rows if row.throughput_rps is not None),
            default=None,
        )
        elapsed = max(
            (
                row.load_elapsed_seconds
                for row in rows
                if row.load_elapsed_seconds is not None
            ),
            default=None,
        )
        values = (
            (
                "throughput_rps",
                "Throughput",
                throughput,
                "requests/s",
                "higher_is_better",
            ),
            (
                "latency_p95_ms",
                "Latency p95",
                percentile(latency, 0.95),
                "ms",
                "lower_is_better",
            ),
            (
                "latency_p99_ms",
                "Latency p99",
                percentile(latency, 0.99),
                "ms",
                "lower_is_better",
            ),
            ("success_rate", "Success rate", success, "fraction", "higher_is_better"),
            (
                "error_rate",
                "Error rate",
                1 - success if success is not None else None,
                "fraction",
                "lower_is_better",
            ),
            (
                "elapsed_seconds",
                "Elapsed wall time",
                elapsed,
                "seconds",
                "lower_is_better",
            ),
            ("request_count", "Request count", float(len(rows)), "requests", "target"),
            (
                "runtime_cost_usd",
                "Runtime cost",
                _sum_available(row.runtime_cost for row in rows),
                "USD",
                "lower_is_better",
            ),
        )
        metrics.extend(
            _metric(
                f"{prefix}.{suffix}",
                f"Concurrency {concurrency} {name}",
                "capacity",
                value,
                unit,
                direction,
                count,
            )
            for suffix, name, value, unit, direction in values
        )
    return metrics


def _saturation_concurrency(
    by_level: dict[int, list[ExecutionRecord]],
) -> float | None:
    peak = 0.0
    for concurrency, rows in sorted(by_level.items()):
        throughput = max((row.throughput_rps or 0 for row in rows), default=0)
        errors = sum(row.success is False for row in rows) / len(rows) if rows else 0
        if peak and (
            throughput < peak * _SATURATION_THROUGHPUT_RATIO
            or errors > _SATURATION_ERROR_RATE
        ):
            return float(concurrency)
        peak = max(peak, throughput)
    return None


def _summary_metric_specifications(
    throughput: list[float],
    latency: list[float],
    success: float | None,
    sample_count: int,
    saturation: float | None,
    tested_upper_bound: float | None,
    level_count: int,
) -> tuple[tuple[str, str, float | None, str, str, int], ...]:
    return (
        (
            "capacity.throughput_rps",
            "Peak bounded-sweep throughput",
            max(throughput) if throughput else None,
            "requests/s",
            "higher_is_better",
            len(throughput),
        ),
        (
            "capacity.latency_p95_ms",
            "Sweep latency p95",
            percentile(latency, 0.95),
            "ms",
            "lower_is_better",
            len(latency),
        ),
        (
            "capacity.latency_p99_ms",
            "Sweep latency p99",
            percentile(latency, 0.99),
            "ms",
            "lower_is_better",
            len(latency),
        ),
        (
            "capacity.success_rate",
            "Sweep success rate",
            success,
            "fraction",
            "higher_is_better",
            sample_count,
        ),
        (
            "capacity.error_rate",
            "Sweep error rate",
            1 - success if success is not None else None,
            "fraction",
            "lower_is_better",
            sample_count,
        ),
        (
            "capacity.saturation_concurrency",
            "Observed saturation concurrency",
            saturation,
            "concurrency",
            "higher_is_better",
            level_count,
        ),
        (
            "capacity.saturation_concurrency_lower_bound",
            "Saturation concurrency lower bound",
            saturation if saturation is not None else tested_upper_bound,
            "concurrency",
            "higher_is_better",
            level_count,
        ),
        (
            "capacity.saturation_observed",
            "Saturation observed in bounded sweep",
            1.0 if saturation is not None else 0.0,
            "boolean",
            "target",
            level_count,
        ),
        (
            "capacity.slo_headroom",
            "SLO headroom",
            None,
            "concurrency",
            "higher_is_better",
            0,
        ),
    )


def _summary_metrics(
    records: list[ExecutionRecord], by_level: dict[int, list[ExecutionRecord]]
) -> list[EvaluationMetric]:
    throughput = [
        row.throughput_rps for row in records if row.throughput_rps is not None
    ]
    latency = [row.latency_ms for row in records if row.latency_ms is not None]
    success, count = _mean(
        float(bool(row.success)) for row in records if row.success is not None
    )
    successful = sum(bool(row.success) for row in records)
    capacity_cost = _sum_available(row.capacity_tco for row in records)
    if capacity_cost is None:
        capacity_cost = _sum_available(row.runtime_cost for row in records)
    saturation = _saturation_concurrency(by_level)
    tested_upper_bound = float(max(by_level)) if by_level else None
    specifications = _summary_metric_specifications(
        throughput,
        latency,
        success,
        count,
        saturation,
        tested_upper_bound,
        len(by_level),
    )
    metrics = [
        _metric(metric_id, name, "capacity", value, unit, direction, sample_count)
        for metric_id, name, value, unit, direction, sample_count in specifications
    ]
    metrics.append(
        _metric(
            "capacity.cost_per_successful_request",
            "Capacity cost per successful request",
            "capacity",
            (
                capacity_cost / successful
                if capacity_cost is not None and successful
                else None
            ),
            "USD/request",
            "lower_is_better",
            successful,
        )
    )
    return metrics


def _capacity(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    by_level: dict[int, list[ExecutionRecord]] = defaultdict(list)
    for row in records:
        if row.concurrency is not None:
            by_level[row.concurrency].append(row)
    return _summary_metrics(records, dict(by_level)) + _level_metrics(dict(by_level))
