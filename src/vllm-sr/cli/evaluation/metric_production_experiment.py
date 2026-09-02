"""Production A/B assignment, safety-control, and causal preference reducers."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, build_metric
from cli.evaluation.production_experiment_metric_specs import (
    production_experiment_metric_specs,
)
from cli.evaluation.production_experiment_reducer import reduce_production_experiment


def _experiment_metric(
    metric_id: str,
    name: str,
    value: float | None,
    unit: str,
    direction: str,
    sample_count: int,
    confidence_interval: tuple[float, float] | None = None,
) -> MetricDraft:
    metric = build_metric(
        metric_id,
        name,
        "preference",
        value,
        unit,
        direction,
        sample_count,
    )
    if confidence_interval is None:
        return metric
    return metric.model_copy(update={"confidence_interval": confidence_interval})


def production_experiment_metrics(
    records: list[ExecutionRecord],
) -> list[MetricDraft]:
    reduced = reduce_production_experiment(records)
    count = reduced.assignment_count
    return [
        _experiment_metric(
            metric_id,
            name,
            value,
            unit,
            direction,
            count,
            confidence,
        )
        for metric_id, name, value, unit, direction, confidence in (
            production_experiment_metric_specs(reduced)
        )
    ]
