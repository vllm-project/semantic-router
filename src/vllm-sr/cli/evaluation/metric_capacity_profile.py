"""Metric projection for attested closed-loop capacity profiles."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.capacity_profile import CapacityProfile, CapacityProfileLevel
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_capacity_support import MetricSpec, measurement_cost
from cli.evaluation.metric_core import MetricDraft, build_metric


def _level_performance_specs(level: CapacityProfileLevel) -> tuple[MetricSpec, ...]:
    return (
        (
            "throughput_rps",
            "Mean repeated-window throughput",
            level.throughput_rps,
            "requests/s",
            "higher_is_better",
            len(level.repetitions),
        ),
        (
            "throughput_cv",
            "Throughput coefficient of variation",
            level.throughput_cv,
            "ratio",
            "lower_is_better",
            len(level.repetitions),
        ),
        (
            "latency_p95_ms",
            "Measurement latency p95",
            level.latency_p95_ms,
            "ms",
            "lower_is_better",
            level.measurement_requests,
        ),
        (
            "latency_p99_ms",
            "Measurement latency p99",
            level.latency_p99_ms,
            "ms",
            "lower_is_better",
            level.measurement_requests,
        ),
        (
            "latency_p95_cv",
            "Latency p95 coefficient of variation",
            level.latency_p95_cv,
            "ratio",
            "lower_is_better",
            len(level.repetitions),
        ),
        (
            "success_rate",
            "Mean independent-cluster success rate",
            1 - level.error_rate,
            "fraction",
            "higher_is_better",
            level.measurement_cluster_count,
        ),
        (
            "error_rate",
            "Mean independent-cluster error rate",
            level.error_rate,
            "fraction",
            "lower_is_better",
            level.measurement_cluster_count,
        ),
        (
            "error_rate_upper_bound",
            "Worst independent-cluster one-sided 95% error-rate upper bound",
            level.error_rate_upper_bound,
            "fraction",
            "lower_is_better",
            level.measurement_cluster_count,
        ),
        (
            "error_rate_cluster_range",
            "Independent-cluster error-rate range",
            level.error_rate_cluster_range,
            "fraction",
            "lower_is_better",
            level.measurement_cluster_count,
        ),
    )


def _level_protocol_specs(level: CapacityProfileLevel) -> tuple[MetricSpec, ...]:
    return (
        (
            "elapsed_seconds",
            "Total measurement wall time",
            level.elapsed_seconds,
            "seconds",
            "lower_is_better",
            len(level.repetitions),
        ),
        (
            "measurement_cluster_count",
            "Independent measurement clusters",
            float(level.measurement_cluster_count),
            "clusters",
            "target",
            level.measurement_cluster_count,
        ),
        (
            "measurement_request_count",
            "Measurement requests",
            float(level.measurement_requests),
            "requests",
            "target",
            level.measurement_requests,
        ),
        (
            "warmup_request_count",
            "Warmup requests",
            float(level.warmup_requests),
            "requests",
            "target",
            level.warmup_requests,
        ),
        (
            "warmup_error_count",
            "Warmup errors",
            float(level.warmup_errors),
            "errors",
            "lower_is_better",
            level.warmup_requests,
        ),
        (
            "runtime_cost_usd",
            "Measurement runtime cost",
            level.runtime_cost_usd,
            "USD",
            "lower_is_better",
            level.measurement_requests,
        ),
        (
            "throughput_scaling_efficiency",
            "Adjacent-level throughput scaling efficiency",
            level.throughput_scaling_efficiency,
            "ratio",
            "higher_is_better",
            (
                len(level.repetitions)
                if level.throughput_scaling_efficiency is not None
                else 0
            ),
        ),
        (
            "qualified",
            "Frozen SLO level qualification",
            1.0 if level.qualified else 0.0,
            "boolean",
            "target",
            level.measurement_requests,
        ),
    )


def _level_metrics(level: CapacityProfileLevel) -> list[MetricDraft]:
    prefix = f"capacity.level.{level.concurrency}"
    specifications = (*_level_performance_specs(level), *_level_protocol_specs(level))
    return [
        build_metric(
            f"{prefix}.{suffix}",
            f"Concurrency {level.concurrency} {name}",
            "capacity",
            value,
            unit,
            direction,
            sample_count,
        )
        for suffix, name, value, unit, direction, sample_count in specifications
    ]


@dataclass(frozen=True)
class _ProfileSummaryStats:
    total_requests: int
    total_clusters: int
    cluster_error_rates: tuple[float, ...]
    successes: int
    cost: float | None
    saturation: int | None
    qualified_error_levels: list[int]


def _summary_stats(
    records: list[ExecutionRecord], profile: CapacityProfile
) -> _ProfileSummaryStats:
    measurement = [row for row in records if row.load_phase == "measurement"]
    total_requests = sum(level.measurement_requests for level in profile.levels)
    successes = sum(level.successes for level in profile.levels)
    cluster_error_rates = tuple(
        repetition.error_rate
        for level in profile.levels
        for repetition in level.repetitions
    )
    cost = measurement_cost(measurement)
    if cost is None:
        cost = sum(level.runtime_cost_usd for level in profile.levels)
    return _ProfileSummaryStats(
        total_requests=total_requests,
        total_clusters=len(cluster_error_rates),
        cluster_error_rates=cluster_error_rates,
        successes=successes,
        cost=cost,
        saturation=profile.assessment.saturation_concurrency,
        qualified_error_levels=[
            level.concurrency for level in profile.levels if level.error_slo_passed
        ],
    )


def _summary_performance_specs(
    profile: CapacityProfile, stats: _ProfileSummaryStats
) -> tuple[MetricSpec, ...]:
    return (
        (
            "capacity.throughput_rps",
            "Peak repeated-window throughput",
            max(level.throughput_rps for level in profile.levels),
            "requests/s",
            "higher_is_better",
            len(profile.levels),
        ),
        (
            "capacity.latency_p95_ms",
            "Worst measured-level latency p95",
            max(level.latency_p95_ms for level in profile.levels),
            "ms",
            "lower_is_better",
            stats.total_requests,
        ),
        (
            "capacity.latency_p99_ms",
            "Worst measured-level latency p99",
            max(level.latency_p99_ms for level in profile.levels),
            "ms",
            "lower_is_better",
            stats.total_requests,
        ),
        (
            "capacity.success_rate",
            "Mean independent-cluster success rate",
            1 - (sum(stats.cluster_error_rates) / stats.total_clusters),
            "fraction",
            "higher_is_better",
            stats.total_clusters,
        ),
        (
            "capacity.error_rate",
            "Mean independent-cluster error rate",
            sum(stats.cluster_error_rates) / stats.total_clusters,
            "fraction",
            "lower_is_better",
            stats.total_clusters,
        ),
        (
            "capacity.error_rate_upper_bound",
            "Worst independent-cluster one-sided 95% error-rate upper bound",
            max(
                repetition.error_rate_upper_bound
                for level in profile.levels
                for repetition in level.repetitions
            ),
            "fraction",
            "lower_is_better",
            stats.total_clusters,
        ),
        (
            "capacity.error_rate_cluster_range_max",
            "Worst independent-cluster error-rate range",
            max(level.error_rate_cluster_range for level in profile.levels),
            "fraction",
            "lower_is_better",
            len(profile.levels),
        ),
        (
            "capacity.throughput_stability_cv_max",
            "Worst throughput coefficient of variation",
            max(level.throughput_cv for level in profile.levels),
            "ratio",
            "lower_is_better",
            len(profile.levels),
        ),
        (
            "capacity.latency_p95_stability_cv_max",
            "Worst latency p95 coefficient of variation",
            max(level.latency_p95_cv for level in profile.levels),
            "ratio",
            "lower_is_better",
            len(profile.levels),
        ),
    )


def _summary_envelope_specs(
    profile: CapacityProfile, stats: _ProfileSummaryStats
) -> tuple[MetricSpec, ...]:
    return (
        (
            "capacity.measurement_cluster_count_min",
            "Minimum independent clusters per concurrency level",
            float(min(level.measurement_cluster_count for level in profile.levels)),
            "clusters",
            "target",
            len(profile.levels),
        ),
        (
            "capacity.measurement_request_count",
            "Frozen measurement requests",
            float(stats.total_requests),
            "requests",
            "target",
            stats.total_requests,
        ),
        (
            "capacity.warmup_error_count",
            "Warmup errors",
            float(sum(level.warmup_errors for level in profile.levels)),
            "errors",
            "lower_is_better",
            sum(level.warmup_requests for level in profile.levels),
        ),
        (
            "capacity.saturation_concurrency",
            "First unqualified concurrency",
            float(stats.saturation) if stats.saturation is not None else None,
            "concurrency",
            "higher_is_better",
            len(profile.levels),
        ),
        (
            "capacity.saturation_concurrency_lower_bound",
            "Measured saturation lower bound",
            float(stats.saturation or profile.levels[-1].concurrency),
            "concurrency",
            "higher_is_better",
            len(profile.levels),
        ),
        (
            "capacity.saturation_observed",
            "Saturation observed in the frozen load ladder",
            1.0 if stats.saturation is not None else 0.0,
            "boolean",
            "target",
            len(profile.levels),
        ),
        (
            "capacity.slo_headroom",
            "Qualified concurrency above the frozen SLO requirement",
            float(profile.assessment.slo_headroom),
            "concurrency",
            "higher_is_better",
            len(profile.levels),
        ),
        (
            "capacity.success_concurrency_upper_bound",
            "Highest concurrency meeting the one-sided error SLO",
            max(stats.qualified_error_levels) if stats.qualified_error_levels else None,
            "concurrency",
            "higher_is_better",
            len(profile.levels),
        ),
        (
            "capacity.cost_per_successful_request",
            "Measurement cost per successful request",
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


def capacity_profile_metrics(
    records: list[ExecutionRecord], profile: CapacityProfile
) -> list[MetricDraft]:
    stats = _summary_stats(records, profile)
    specifications = (
        *_summary_performance_specs(profile, stats),
        *_summary_envelope_specs(profile, stats),
    )
    metrics = [
        build_metric(metric_id, name, "capacity", value, unit, direction, sample_count)
        for metric_id, name, value, unit, direction, sample_count in specifications
    ]
    for level in profile.levels:
        metrics.extend(_level_metrics(level))
    return metrics
