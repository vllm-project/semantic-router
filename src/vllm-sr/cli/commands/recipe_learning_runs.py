"""Run-level outcomes for offline Router Learning analysis.

An agent run is many routing events (one per model call) that share a session
or conversation identity. Its outcome, such as whether the task's tests passed,
is reported once against one of those events, and its cost is the sum of every
event's recorded cost. This module groups replay records into runs and reports,
per decision and model, how often runs succeed and what a run costs, so a
deployment can choose its default model from its own outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cli.commands.recipe_learning_metrics import (
    numeric,
    record_actual_cost,
    record_decision,
    record_selected_model,
)

_SUCCESS_VERDICTS = frozenset({"good_fit", "overprovisioned"})
_FAILURE_VERDICTS = frozenset({"underpowered", "failed"})
_WILSON_Z = 1.96
MIN_RUNS_FOR_RECOMMENDATION = 20
# A sample spread needs at least two values.
_MIN_VALUES_FOR_SPREAD = 2


@dataclass
class _Run:
    decision: str
    model: str
    first_turn: tuple[int, float]
    cost: float = 0.0
    models: set[str] = field(default_factory=set)
    outcome: tuple[float, bool] | None = None  # (time, success)


def parse_time(value: Any) -> float:
    """RFC 3339 time as epoch seconds; replay timestamps drop trailing zeros, so
    they do not sort as strings. Unparsable values sort first."""
    text = str(value or "").strip()
    if not text:
        return float("-inf")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")


def run_identity(record: dict[str, Any]) -> str:
    """The run a record belongs to: its conversation, else its session."""
    for key in ("conversation_id", "session_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def record_run_outcome(record: dict[str, Any]) -> tuple[float, bool] | None:
    """The latest model-target outcome on a record, as (time, success)."""
    latest: tuple[float, bool] | None = None
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict) or outcome.get("target") != "model":
            continue
        verdict = str(outcome.get("verdict") or "")
        if verdict not in _SUCCESS_VERDICTS | _FAILURE_VERDICTS:
            continue
        stamp = parse_time(outcome.get("timestamp"))
        if latest is None or stamp >= latest[0]:
            latest = (stamp, verdict in _SUCCESS_VERDICTS)
    return latest


def group_runs(records: list[dict[str, Any]]) -> tuple[dict[str, _Run], int]:
    """Group records into runs; also return how many records had no identity."""
    runs: dict[str, _Run] = {}
    unidentified = 0
    for record in records:
        key = run_identity(record)
        if not key:
            unidentified += 1
            continue
        model = record_selected_model(record)
        turn = (
            int(numeric(record.get("turn_index"))),
            parse_time(record.get("timestamp")),
        )
        run = runs.get(key)
        if run is None:
            run = runs[key] = _Run(
                decision=record_decision(record), model=model, first_turn=turn
            )
        elif turn < run.first_turn:
            run.decision, run.model, run.first_turn = (
                record_decision(record),
                model,
                turn,
            )
        run.cost += record_actual_cost(record)
        if model:
            run.models.add(model)
        outcome = record_run_outcome(record)
        if outcome is not None and (
            run.outcome is None or outcome[0] >= run.outcome[0]
        ):
            run.outcome = outcome
    return runs, unidentified


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + _WILSON_Z**2 / trials
    centre = (p + _WILSON_Z**2 / (2 * trials)) / denom
    half = (
        _WILSON_Z
        * math.sqrt(p * (1 - p) / trials + _WILSON_Z**2 / (4 * trials**2))
        / denom
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean_interval(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < _MIN_VALUES_FOR_SPREAD:
        return mean, mean, mean
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = _WILSON_Z * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def model_run_stats(model_runs: list[_Run]) -> dict[str, Any]:
    """Success and cost statistics over a model's runs that stayed on it.

    A run whose calls went to more than one model mixes their work: the router
    moved it (a rescue), the harness fell back to another model, or a subagent
    of the run used another model. It is counted but kept out of the statistics."""
    kept = [r for r in model_runs if len(r.models) <= 1]
    labeled = [r for r in kept if r.outcome is not None]
    stats: dict[str, Any] = {
        "runs": len(model_runs),
        "switched_runs": len(model_runs) - len(kept),
        "labeled_runs": len(labeled),
        "success_rate": None,
        "success_interval": None,
        "cost_per_run": None,
        "cost_interval": None,
    }
    if labeled:
        successes = sum(1 for r in labeled if r.outcome and r.outcome[1])
        cost, cost_low, cost_high = _mean_interval([r.cost for r in labeled])
        stats.update(
            success_rate=successes / len(labeled),
            success_interval=list(wilson_interval(successes, len(labeled))),
            cost_per_run=cost,
            cost_interval=[cost_low, cost_high],
        )
    return stats


def build_run_model_choice(
    records: list[dict[str, Any]],
    values_per_success: list[float] | None = None,
) -> dict[str, Any]:
    """Per decision and model: run success rate and cost per run with 95%
    intervals, and, for each stated value of one successful run, the model with
    the highest expected utility."""
    runs, unidentified = group_runs(records)
    by_decision: dict[str, dict[str, list[_Run]]] = {}
    for run in runs.values():
        by_decision.setdefault(run.decision, {}).setdefault(run.model, []).append(run)

    decisions = {}
    for decision, by_model in sorted(by_decision.items()):
        models = {
            model: model_run_stats(model_runs)
            for model, model_runs in sorted(by_model.items())
        }
        decisions[decision] = {
            "models": models,
            "recommendations": [
                recommend_default(models, value) for value in values_per_success or []
            ],
        }
    return {
        "runs": len(runs),
        "unidentified_records": unidentified,
        "labeled_runs": sum(1 for r in runs.values() if r.outcome is not None),
        "per_decision": decisions,
    }


def recommend_default(models: dict[str, Any], value: float) -> dict[str, Any]:
    """The model with the highest expected utility (value x success rate - cost
    per run) among models with enough labeled runs.

    It is confident only when its utility at the pessimistic end of both
    intervals (low success, high cost) beats the optimistic end of every other
    model that has any labeled runs, including models with too few runs to be
    recommended themselves: an adaptive router serves the models that look weak
    least often, so their wide intervals must still be able to block a call."""
    bounds = {}
    for model, stats in models.items():
        if not stats["labeled_runs"]:
            continue
        low, high = stats["success_interval"]
        cost_low, cost_high = stats["cost_interval"]
        bounds[model] = (
            value * stats["success_rate"] - stats["cost_per_run"],
            value * low - cost_high,
            value * high - cost_low,
            stats["labeled_runs"] >= MIN_RUNS_FOR_RECOMMENDATION,
        )
    eligible = sorted(
        ((b[0], model) for model, b in bounds.items() if b[3]), reverse=True
    )
    if not eligible:
        return {
            "value_per_success": value,
            "model": None,
            "confident": False,
            "reason": f"no model has {MIN_RUNS_FOR_RECOMMENDATION} labeled runs",
        }
    best = eligible[0][1]
    others = [m for m in bounds if m != best]
    blocking = [m for m in others if bounds[best][1] <= bounds[m][2]]
    result = {
        "value_per_success": value,
        "model": best,
        "expected_utility": bounds[best][0],
        "confident": bool(others) and not blocking,
        "compared": sorted(bounds),
    }
    if not others:
        result["reason"] = "no other model has labeled runs"
    elif blocking:
        result["reason"] = "intervals overlap with " + ", ".join(sorted(blocking))
    return result
