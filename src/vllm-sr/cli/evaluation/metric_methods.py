"""Dispatch source-qualified method evidence to narrow reducers."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_agent_task import agent_task_metrics
from cli.evaluation.metric_core import MetricDraft
from cli.evaluation.metric_hard_policy import hard_policy_metrics
from cli.evaluation.metric_production_experiment import production_experiment_metrics
from cli.evaluation.metric_recovery import recovery_metrics
from cli.evaluation.metric_robustness import robustness_metrics


def method_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    tracks = {row.track_id for row in records}
    return [
        *(robustness_metrics(records) if "routing" in tracks else ()),
        *(agent_task_metrics(records) if "agentic" in tracks else ()),
        *(recovery_metrics(records) if "agentic" in tracks else ()),
        *(production_experiment_metrics(records) if "preference" in tracks else ()),
        *(hard_policy_metrics(records) if "safety" in tracks else ()),
    ]
