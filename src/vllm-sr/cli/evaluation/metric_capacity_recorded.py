"""Diagnostic metric projection for raw recorded capacity observations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_capacity_support import MetricSpec, measurement_cost
from cli.evaluation.metric_core import (
    MetricDraft,
    build_metric,
    complete_sum,
    percentile,
)

_SATURATION_THROUGHPUT_RATIO = 0.95
_SATURATION_ERROR_RATE = 0.05


def _saturation(by_level: dict[int, list[ExecutionRecord]]) -> int | None:
    peak = 0.0
    for concurrency, rows in sorted(by_level.items()):
        throughput = max((row.throughput_rps or 0 for row in rows), default=0)
        errors = sum(row.success is False for row in rows) / len(rows) if rows else 0
        if peak and (
            throughput < peak * _SATURATION_THROUGHPUT_RATIO
            or errors > _SATURATION_ERROR_RATE
        ):
            return concurrency
        peak = max(peak, throughput)
    return None


def _level_performance_specs(
    rows: list[ExecutionRecord],
) -> tuple[MetricSpec, ...]:
    latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
    throughput = max(
        (row.throughput_rps for row in rows if row.throughput_rps is not None),
        default=None,
    )
    return (
        (
            "throughput_rps",
            "Recorded throughput",
            throughput,
            "requests/s",
            "higher_is_better",
            len(rows),
        ),
        (
            "latency_p95_ms",
            "Recorded latency p95",
            percentile(latencies, 0.95),
            "ms",
            "lower_is_better",
            len(latencies),
        ),
        (
            "latency_p99_ms",
            "Recorded latency p99",
            percentile(latencies, 0.99),
            "ms",
            "lower_is_better",
            len(latencies),
        ),
        (
            "success_rate",
            "Mean independent-cluster success rate",
            None,
            "fraction",
            "higher_is_better",
            0,
        ),
        (
            "error_rate",
            "Mean independent-cluster error rate",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
        (
            "error_rate_upper_bound",
            "Worst independent-cluster one-sided 95% error-rate upper bound",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
        (
            "error_rate_cluster_range",
            "Independent-cluster error-rate range",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
    )


def _level_accounting_specs(
    rows: list[ExecutionRecord],
) -> tuple[MetricSpec, ...]:
    elapsed = max(
        (
            row.load_elapsed_seconds
            for row in rows
            if row.load_elapsed_seconds is not None
        ),
        default=None,
    )
    return (
        (
            "elapsed_seconds",
            "Recorded elapsed time",
            elapsed,
            "seconds",
            "lower_is_better",
            len(rows),
        ),
        (
            "measurement_cluster_count",
            "Independent measurement clusters",
            None,
            "clusters",
            "target",
            0,
        ),
        (
            "measurement_request_count",
            "Recorded observations",
            float(len(rows)),
            "requests",
            "target",
            len(rows),
        ),
        (
            "runtime_cost_usd",
            "Recorded runtime cost",
            complete_sum(row.runtime_cost for row in rows),
            "USD",
            "lower_is_better",
            len(rows),
        ),
    )


def _level_metrics(concurrency: int, rows: list[ExecutionRecord]) -> list[MetricDraft]:
    prefix = f"capacity.level.{concurrency}"
    values = (*_level_performance_specs(rows), *_level_accounting_specs(rows))
    return [
        build_metric(
            f"{prefix}.{suffix}",
            f"Concurrency {concurrency} {name}",
            "capacity",
            value,
            unit,
            direction,
            sample_count,
        )
        for suffix, name, value, unit, direction, sample_count in values
    ]


@dataclass(frozen=True)
class _RecordedSummaryStats:
    by_level: dict[int, list[ExecutionRecord]]
    latencies: list[float]
    throughputs: list[float]
    successes: int
    saturation: int | None
    successful_levels: list[int]
    cost: float | None


def _summary_stats(records: list[ExecutionRecord]) -> _RecordedSummaryStats:
    by_level: dict[int, list[ExecutionRecord]] = defaultdict(list)
    for row in records:
        if row.concurrency is not None:
            by_level[row.concurrency].append(row)
    successes = sum(row.success is True for row in records)
    return _RecordedSummaryStats(
        by_level=dict(by_level),
        latencies=[row.latency_ms for row in records if row.latency_ms is not None],
        throughputs=[
            row.throughput_rps for row in records if row.throughput_rps is not None
        ],
        successes=successes,
        saturation=_saturation(dict(by_level)),
        successful_levels=[
            concurrency
            for concurrency, rows in by_level.items()
            if rows
            and sum(row.success is True for row in rows) / len(rows)
            >= 1 - _SATURATION_ERROR_RATE
        ],
        cost=measurement_cost(records),
    )


def _performance_specs(stats: _RecordedSummaryStats) -> tuple[MetricSpec, ...]:
    return (
        (
            "capacity.throughput_rps",
            "Peak recorded-source throughput",
            max(stats.throughputs) if stats.throughputs else None,
            "requests/s",
            "higher_is_better",
            len(stats.throughputs),
        ),
        (
            "capacity.latency_p95_ms",
            "Recorded-source latency p95",
            percentile(stats.latencies, 0.95),
            "ms",
            "lower_is_better",
            len(stats.latencies),
        ),
        (
            "capacity.latency_p99_ms",
            "Recorded-source latency p99",
            percentile(stats.latencies, 0.99),
            "ms",
            "lower_is_better",
            len(stats.latencies),
        ),
        (
            "capacity.success_rate",
            "Mean independent-cluster success rate",
            None,
            "fraction",
            "higher_is_better",
            0,
        ),
        (
            "capacity.error_rate",
            "Mean independent-cluster error rate",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
        (
            "capacity.error_rate_upper_bound",
            "Worst independent-cluster one-sided 95% error-rate upper bound",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
        (
            "capacity.error_rate_cluster_range_max",
            "Worst independent-cluster error-rate range",
            None,
            "fraction",
            "lower_is_better",
            0,
        ),
        (
            "capacity.throughput_stability_cv_max",
            "Repeated-window throughput stability unavailable",
            None,
            "ratio",
            "lower_is_better",
            0,
        ),
        (
            "capacity.latency_p95_stability_cv_max",
            "Repeated-window latency stability unavailable",
            None,
            "ratio",
            "lower_is_better",
            0,
        ),
    )


def _envelope_specs(
    records: list[ExecutionRecord], stats: _RecordedSummaryStats
) -> tuple[MetricSpec, ...]:
    return (
        (
            "capacity.measurement_cluster_count_min",
            "Minimum independent clusters per concurrency level",
            None,
            "clusters",
            "target",
            0,
        ),
        (
            "capacity.measurement_request_count",
            "Recorded capacity observations",
            float(len(records)),
            "requests",
            "target",
            len(records),
        ),
        (
            "capacity.warmup_error_count",
            "Warmup evidence unavailable for recorded source",
            None,
            "errors",
            "lower_is_better",
            0,
        ),
        (
            "capacity.saturation_concurrency",
            "Recorded-source saturation indicator",
            float(stats.saturation) if stats.saturation is not None else None,
            "concurrency",
            "higher_is_better",
            len(stats.by_level),
        ),
        (
            "capacity.saturation_concurrency_lower_bound",
            "Highest tested recorded-source concurrency",
            float(stats.saturation or max(stats.by_level)) if stats.by_level else None,
            "concurrency",
            "higher_is_better",
            len(stats.by_level),
        ),
        (
            "capacity.saturation_observed",
            "Recorded-source saturation observed",
            1.0 if stats.saturation is not None else 0.0,
            "boolean",
            "target",
            len(stats.by_level),
        ),
        (
            "capacity.slo_headroom",
            "Qualified concurrency above the frozen SLO requirement",
            None,
            "concurrency",
            "higher_is_better",
            0,
        ),
        (
            "capacity.success_concurrency_upper_bound",
            "Highest recorded concurrency below the diagnostic error threshold",
            float(max(stats.successful_levels)) if stats.successful_levels else None,
            "concurrency",
            "higher_is_better",
            len(stats.by_level),
        ),
        (
            "capacity.cost_per_successful_request",
            "Recorded cost per successful request",
            (
                stats.cost / stats.successes
                if stats.cost is not None and stats.successes
                else None
            ),
            "USD/request",
            "lower_is_better",
            stats.successes,
        ),
    )


def recorded_capacity_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    stats = _summary_stats(records)
    specifications = (*_performance_specs(stats), *_envelope_specs(records, stats))
    metrics = [
        build_metric(metric_id, name, "capacity", value, unit, direction, sample_count)
        for metric_id, name, value, unit, direction, sample_count in specifications
    ]
    for concurrency, rows in sorted(stats.by_level.items()):
        metrics.extend(_level_metrics(concurrency, rows))
    return metrics
