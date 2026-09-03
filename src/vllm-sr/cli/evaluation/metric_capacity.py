"""Dispatch capacity records to their evidence-appropriate metric projection."""

from __future__ import annotations

from cli.evaluation.capacity_profile import CapacityProfile
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_capacity_profile import capacity_profile_metrics
from cli.evaluation.metric_capacity_recorded import recorded_capacity_metrics
from cli.evaluation.metric_core import MetricDraft


def capacity_metrics(
    records: list[ExecutionRecord], profile: CapacityProfile | None
) -> list[MetricDraft]:
    if profile is not None:
        return capacity_profile_metrics(records, profile)
    return recorded_capacity_metrics(records)
