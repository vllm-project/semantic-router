"""Trial-clustered policy metrics for Router Learning replay evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from math import sqrt

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, canonical_ordered_float_sum
from cli.evaluation.router_learning_evidence import ROUTER_LEARNING_POLICY_IDS

_Z_95 = 1.959963984540054
_MIN_INTERVAL_TRIALS = 2


def _mean(values: list[float]) -> float | None:
    return canonical_ordered_float_sum(values) / len(values) if values else None


def _cluster_interval(
    values: list[float], *, bounds: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    if len(values) < _MIN_INTERVAL_TRIALS:
        return None
    center = _mean(values)
    if center is None:
        return None
    variance = canonical_ordered_float_sum((value - center) ** 2 for value in values)
    variance /= len(values) - 1
    margin = _Z_95 * sqrt(variance / len(values))
    lower, upper = center - margin, center + margin
    if bounds is not None:
        lower = max(bounds[0], lower)
        upper = min(bounds[1], upper)
    return lower, upper


def _trial_values(
    trials: dict[str, list[ExecutionRecord]],
    value_for: Callable[[list[ExecutionRecord]], float | None],
) -> list[float]:
    values: list[float] = []
    for trial_id in sorted(trials):
        value = value_for(trials[trial_id])
        if value is not None:
            values.append(value)
    return values


def _observed_mean(rows: list[ExecutionRecord], field: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if row.router_learning is not None
        and (value := getattr(row.router_learning, field)) is not None
    ]
    return _mean(values)


def _rate(rows: list[ExecutionRecord], field: str) -> float | None:
    values = [
        float(bool(getattr(row.router_learning, field)))
        for row in rows
        if row.router_learning is not None
    ]
    return _mean(values)


def _protection_rate(rows: list[ExecutionRecord]) -> float | None:
    protected = [
        row
        for row in rows
        if row.router_learning is not None and row.router_learning.protection_required
    ]
    return _rate(protected, "protection_violation")


def _metric(
    policy_id: str,
    statistic: str,
    name: str,
    value: float | None,
    unit: str,
    direction: str,
    sample_count: int,
    trial_values: list[float],
    *,
    bounded: bool = False,
) -> MetricDraft:
    return MetricDraft(
        id=f"joint.router_learning.{policy_id}.{statistic}",
        name=f"{name} ({policy_id})",
        track_id="joint",
        value=value,
        unit=unit,
        direction=direction,
        sample_count=sample_count,
        confidence_interval=_cluster_interval(
            trial_values, bounds=(0.0, 1.0) if bounded else None
        ),
        planned_analysis_units=sample_count,
    )


def _policy_performance_metrics(
    policy_id: str,
    rows: list[ExecutionRecord],
    trials: dict[str, list[ExecutionRecord]],
) -> tuple[MetricDraft, ...]:
    solve_trials = _trial_values(
        trials, lambda values: _rate(values, "outcome_success")
    )
    cost_trials = _trial_values(
        trials, lambda values: _observed_mean(values, "lifecycle_cost_usd")
    )
    latency_trials = _trial_values(
        trials,
        lambda values: _mean(
            [row.latency_ms for row in values if row.latency_ms is not None]
        ),
    )
    call_trials = _trial_values(
        trials, lambda values: _observed_mean(values, "call_count")
    )
    return (
        _metric(
            policy_id,
            "solve_rate",
            "Solve rate",
            _mean(solve_trials),
            "fraction",
            "higher_is_better",
            len(rows),
            solve_trials,
            bounded=True,
        ),
        _metric(
            policy_id,
            "lifecycle_cost_mean_usd",
            "Mean lifecycle cost",
            _mean(cost_trials),
            "USD/round",
            "lower_is_better",
            len(rows),
            cost_trials,
        ),
        _metric(
            policy_id,
            "latency_mean_ms",
            "Mean latency",
            _mean(latency_trials),
            "ms",
            "lower_is_better",
            len(rows),
            latency_trials,
        ),
        _metric(
            policy_id,
            "model_call_mean",
            "Mean model calls",
            _mean(call_trials),
            "calls/round",
            "lower_is_better",
            len(rows),
            call_trials,
        ),
    )


def _policy_guardrail_metrics(
    policy_id: str,
    rows: list[ExecutionRecord],
    trials: dict[str, list[ExecutionRecord]],
) -> tuple[MetricDraft, ...]:
    protected_count = sum(
        row.router_learning is not None and row.router_learning.protection_required
        for row in rows
    )
    protection_trials = _trial_values(trials, _protection_rate)
    hard_trials = _trial_values(
        trials, lambda values: _rate(values, "hard_constraint_violation")
    )
    return (
        _metric(
            policy_id,
            "protection_violation_rate",
            "Protection violation rate",
            _mean(protection_trials),
            "fraction",
            "lower_is_better",
            protected_count,
            protection_trials,
            bounded=True,
        ),
        _metric(
            policy_id,
            "hard_constraint_violation_rate",
            "Hard eligibility violation rate",
            _mean(hard_trials),
            "fraction",
            "lower_is_better",
            len(rows),
            hard_trials,
            bounded=True,
        ),
        _metric(
            policy_id,
            "propensity_coverage",
            "Recorded action-propensity coverage",
            0.0,
            "fraction",
            "higher_is_better",
            len(rows),
            [],
            bounded=True,
        ),
        _metric(
            policy_id,
            "trial_count",
            "Paired trial count",
            float(len(trials)) if trials else None,
            "trials",
            "higher_is_better",
            len(trials),
            [],
        ),
    )


def _policy_metrics(
    policy_id: str, rows: list[ExecutionRecord]
) -> tuple[MetricDraft, ...]:
    trials: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for row in rows:
        assert row.router_learning is not None
        trials[row.router_learning.trial_id].append(row)
    return _policy_performance_metrics(
        policy_id, rows, trials
    ) + _policy_guardrail_metrics(policy_id, rows, trials)


def router_learning_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    by_policy: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for row in records:
        if row.router_learning is not None:
            by_policy[row.router_learning.policy_id].append(row)
    if not by_policy:
        return []
    metrics: list[MetricDraft] = []
    for policy_id in ROUTER_LEARNING_POLICY_IDS:
        metrics.extend(_policy_metrics(policy_id, by_policy.get(policy_id, [])))
    return metrics
