"""Agentic, multimodal, preference, safety, and capacity metrics."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import (
    _mean,
    _metric,
    _wilson,
)
from cli.evaluation.reporting import EvaluationMetric


def _agentic(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    success, success_count = _mean(
        float(bool(row.success)) for row in records if row.success is not None
    )
    quality, quality_count = _mean(
        row.quality for row in records if row.quality is not None
    )
    tool_calls = sum(row.tool_calls or 0 for row in records)
    invalid_calls = sum(row.invalid_tool_calls or 0 for row in records)
    return [
        _metric(
            "agentic.success_rate",
            "Trajectory success rate",
            "agentic",
            success,
            "fraction",
            "higher_is_better",
            success_count,
        ),
        _metric(
            "agentic.task_score",
            "Trajectory task score",
            "agentic",
            quality,
            "score",
            "higher_is_better",
            quality_count,
        ),
        _metric(
            "agentic.invalid_tool_rate",
            "Invalid tool-call rate",
            "agentic",
            invalid_calls / tool_calls if tool_calls else None,
            "fraction",
            "lower_is_better",
            tool_calls,
        ),
    ]


def _multimodal(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    support, support_count = _mean(
        float(bool(row.success)) for row in records if row.success is not None
    )
    quality, quality_count = _mean(
        row.quality for row in records if row.quality is not None
    )
    privacy_values = [
        row.privacy_violations for row in records if row.privacy_violations is not None
    ]
    return [
        _metric(
            "multimodal.support_rate",
            "Multimodal support rate",
            "multimodal",
            support,
            "fraction",
            "higher_is_better",
            support_count,
        ),
        _metric(
            "multimodal.quality",
            "Multimodal quality",
            "multimodal",
            quality,
            "score",
            "higher_is_better",
            quality_count,
        ),
        _metric(
            "multimodal.privacy_violations",
            "Multimodal privacy violations",
            "multimodal",
            float(sum(privacy_values)) if privacy_values else None,
            "count",
            "lower_is_better",
            len(privacy_values),
        ),
    ]


def _preference(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    matches, match_count = _mean(
        float(bool(row.preference_match))
        for row in records
        if row.preference_match is not None
    )
    propensity_count = sum(row.behavior_propensity is not None for row in records)
    return [
        _metric(
            "preference.agreement",
            "Offline preference agreement",
            "preference",
            matches,
            "fraction",
            "higher_is_better",
            match_count,
        ),
        _metric(
            "preference.propensity_coverage",
            "Behavior propensity coverage",
            "preference",
            propensity_count / len(records) if records else None,
            "fraction",
            "higher_is_better",
            len(records),
        ),
    ]


def _safety(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    violations = sum(row.safety_violations or 0 for row in records)
    block_rows = [
        row
        for row in records
        if row.should_block is not None and row.blocked is not None
    ]
    block_accuracy = (
        sum(row.should_block == row.blocked for row in block_rows) / len(block_rows)
        if block_rows
        else None
    )
    violation_cases = sum((row.safety_violations or 0) > 0 for row in records)
    violation_fraction = violation_cases / len(records) if records else None
    upper_bound = _wilson(violation_cases, len(records))[1] if records else None
    return [
        _metric(
            "safety.violation_rate",
            "Safety violation rate",
            "safety",
            violations / len(records) if records else None,
            "violations/case",
            "lower_is_better",
            len(records),
        ),
        _metric(
            "safety.violation_case_rate",
            "Cases with a safety violation",
            "safety",
            violation_fraction,
            "fraction",
            "lower_is_better",
            len(records),
        ),
        _metric(
            "safety.violation_upper_95",
            "Safety violation-rate 95% upper bound",
            "safety",
            upper_bound,
            "fraction",
            "lower_is_better",
            len(records),
        ),
        _metric(
            "safety.block_accuracy",
            "Blocking decision accuracy",
            "safety",
            block_accuracy,
            "fraction",
            "higher_is_better",
            len(block_rows),
        ),
    ]
