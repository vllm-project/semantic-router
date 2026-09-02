"""Routing metric reducers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import (
    MetricDraft,
    build_metric,
    mean_with_count,
    percentile,
)


@dataclass(frozen=True)
class _RoutingReduction:
    total: int
    covered: int
    selected_count: int
    selected_arm_count: int
    selection_entropy: float | None
    success: float | None
    success_count: int
    accuracy: float | None
    accuracy_count: int
    fallback_rate: float | None
    latencies: list[float]


def _reduce_routing(records: list[ExecutionRecord]) -> _RoutingReduction:
    selected = [record.selected_arm_id for record in records if record.selected_arm_id]
    selection_counts = dict(Counter(selected))
    selection_entropy = (
        -sum(
            (count / len(selected)) * log2(count / len(selected))
            for count in selection_counts.values()
        )
        if selected
        else None
    )
    success, success_count = mean_with_count(
        float(record.success) for record in records if record.success is not None
    )
    graded = [record.quality for record in records if record.quality is not None]
    latencies = [
        record.latency_ms for record in records if record.latency_ms is not None
    ]
    total = len(records)
    return _RoutingReduction(
        total=total,
        covered=sum(record.selected_arm_id is not None for record in records),
        selected_count=len(selected),
        selected_arm_count=len(selection_counts),
        selection_entropy=selection_entropy,
        success=success,
        success_count=success_count,
        accuracy=sum(graded) / len(graded) if graded else None,
        accuracy_count=len(graded),
        fallback_rate=(
            sum(bool(row.fallback) for row in records) / total if total else None
        ),
        latencies=latencies,
    )


def routing_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    reduced = _reduce_routing(records)
    return [
        build_metric(
            "routing.coverage",
            "Routing coverage",
            "routing",
            reduced.covered / reduced.total if reduced.total else None,
            "fraction",
            "higher_is_better",
            reduced.total,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.abstention_rate",
            "Routing abstention rate",
            "routing",
            (
                (reduced.total - reduced.covered) / reduced.total
                if reduced.total
                else None
            ),
            "fraction",
            "lower_is_better",
            reduced.total,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.accuracy",
            "Routing accuracy",
            "routing",
            reduced.accuracy,
            "fraction",
            "higher_is_better",
            reduced.accuracy_count,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.fallback_rate",
            "Fallback rate",
            "routing",
            reduced.fallback_rate,
            "fraction",
            "lower_is_better",
            reduced.total,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.success_rate",
            "Routing execution success rate",
            "routing",
            reduced.success,
            "fraction",
            "higher_is_better",
            reduced.success_count,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.selection_entropy_bits",
            "Selected-arm entropy",
            "routing",
            reduced.selection_entropy,
            "bits",
            "target",
            reduced.selected_count,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.selected_arm_count",
            "Selected logical arms",
            "routing",
            float(reduced.selected_arm_count) if reduced.selected_count else None,
            "arms",
            "target",
            reduced.selected_count,
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.latency_p50_ms",
            "Route latency p50",
            "routing",
            percentile(reduced.latencies, 0.50),
            "ms",
            "lower_is_better",
            len(reduced.latencies),
            planned_analysis_units=reduced.total,
        ),
        build_metric(
            "routing.latency_p95_ms",
            "Route latency p95",
            "routing",
            percentile(reduced.latencies, 0.95),
            "ms",
            "lower_is_better",
            len(reduced.latencies),
            planned_analysis_units=reduced.total,
        ),
    ]
