"""Joint routing-and-pool metric reducers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import (
    MetricDraft,
    build_metric,
    canonical_ordered_float_sum,
    complete_sum,
    mean_with_count,
    percentile,
)
from cli.evaluation.metric_model_pool import outcome_quality


@dataclass(frozen=True)
class _JointReduction:
    realized: float | None
    realized_count: int
    oracle_regret: float | None
    oracle_regret_count: int
    normalized_regret: float | None
    normalized_regret_count: int
    reliability: float | None
    reliability_count: int
    oracle_capture: float | None
    oracle_capture_count: int
    runtime_cost_per_success: float | None
    successful_count: int
    latencies: list[float]


def _reduce_joint(
    records: list[ExecutionRecord], pool_records: list[ExecutionRecord]
) -> _JointReduction:
    realized, count = mean_with_count(
        quality for row in records if (quality := outcome_quality(row)) is not None
    )
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in pool_records:
        if (quality := outcome_quality(row)) is not None:
            by_case[row.case_id].append(quality)
    oracle_by_case = {
        case_id: max(values) for case_id, values in by_case.items() if values
    }
    regrets = [
        max(0.0, oracle_by_case[row.case_id] - quality)
        for row in records
        if (quality := outcome_quality(row)) is not None
        and row.case_id in oracle_by_case
    ]
    normalized_regrets = [
        max(0.0, oracle_by_case[row.case_id] - quality) / oracle_by_case[row.case_id]
        for row in records
        if (quality := outcome_quality(row)) is not None
        and row.case_id in oracle_by_case
        and oracle_by_case[row.case_id] > 0
    ]
    oracle_capture = [
        min(1.0, quality / oracle_by_case[row.case_id])
        for row in records
        if (quality := outcome_quality(row)) is not None
        and row.case_id in oracle_by_case
        and oracle_by_case[row.case_id] > 0
    ]
    reliability, reliability_count = mean_with_count(
        float(bool(row.success)) for row in records if row.success is not None
    )
    latencies = [row.latency_ms for row in records if row.latency_ms is not None]
    successful = sum(row.success is True for row in records)
    runtime_cost = complete_sum(row.runtime_cost for row in records)
    return _JointReduction(
        realized=realized,
        realized_count=count,
        oracle_regret=sum(regrets) / len(regrets) if regrets else None,
        oracle_regret_count=len(regrets),
        normalized_regret=(
            canonical_ordered_float_sum(normalized_regrets) / len(normalized_regrets)
            if normalized_regrets
            else None
        ),
        normalized_regret_count=len(normalized_regrets),
        reliability=reliability,
        reliability_count=reliability_count,
        oracle_capture=(
            canonical_ordered_float_sum(oracle_capture) / len(oracle_capture)
            if oracle_capture
            else None
        ),
        oracle_capture_count=len(oracle_capture),
        runtime_cost_per_success=(
            runtime_cost / successful
            if runtime_cost is not None and successful
            else None
        ),
        successful_count=successful,
        latencies=latencies,
    )


def joint_metrics(
    records: list[ExecutionRecord], pool_records: list[ExecutionRecord]
) -> list[MetricDraft]:
    reduced = _reduce_joint(records, pool_records)
    return [
        build_metric(
            "joint.realized_quality",
            "Realized routing quality",
            "joint",
            reduced.realized,
            "score",
            "higher_is_better",
            reduced.realized_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.oracle_regret",
            "Pool-oracle regret",
            "joint",
            reduced.oracle_regret,
            "score",
            "lower_is_better",
            reduced.oracle_regret_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.normalized_regret",
            "Normalized pool-oracle regret",
            "joint",
            reduced.normalized_regret,
            "fraction",
            "lower_is_better",
            reduced.normalized_regret_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.reliability",
            "End-to-end execution reliability",
            "joint",
            reduced.reliability,
            "fraction",
            "higher_is_better",
            reduced.reliability_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.oracle_capture_ratio",
            "Realized fraction of pool-oracle quality",
            "joint",
            reduced.oracle_capture,
            "fraction",
            "higher_is_better",
            reduced.oracle_capture_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.runtime_cost_per_success",
            "Runtime cost per successful case",
            "joint",
            reduced.runtime_cost_per_success,
            "USD/success",
            "lower_is_better",
            reduced.successful_count,
            planned_analysis_units=len(records),
        ),
        build_metric(
            "joint.latency_p95_ms",
            "End-to-end latency p95",
            "joint",
            percentile(reduced.latencies, 0.95),
            "ms",
            "lower_is_better",
            len(reduced.latencies),
            planned_analysis_units=len(records),
        ),
    ]
