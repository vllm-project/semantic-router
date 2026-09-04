"""Dispatch normalized evidence to narrow track-specific metric reducers."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_capacity import _capacity
from cli.evaluation.metric_core import coverage, percentile
from cli.evaluation.metric_routing_pool import _joint, _model_pool, _routing
from cli.evaluation.metric_tracks import _agentic, _multimodal, _preference, _safety
from cli.evaluation.reporting import EvaluationMetric

__all__ = ["compute_metrics", "coverage", "percentile"]


def compute_metrics(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    by_track = {
        track: [
            row
            for row in records
            if row.track_id == track and row.status != "unavailable"
        ]
        for track in (
            "routing",
            "model_pool",
            "joint",
            "agentic",
            "multimodal",
            "preference",
            "safety",
            "capacity",
        )
    }
    metrics: list[EvaluationMetric] = []
    metrics.extend(_routing(by_track["routing"]))
    metrics.extend(_model_pool(by_track["model_pool"], by_track["joint"]))
    metrics.extend(_joint(by_track["joint"], by_track["model_pool"]))
    metrics.extend(_agentic(by_track["agentic"]))
    metrics.extend(_multimodal(by_track["multimodal"]))
    metrics.extend(_preference(by_track["preference"]))
    metrics.extend(_safety(by_track["safety"]))
    metrics.extend(_capacity(by_track["capacity"]))
    return metrics
