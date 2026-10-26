"""Scores keep planned denominators and unknown billing visible."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Collection
from datetime import datetime
from fractions import Fraction
from http import HTTPStatus
from itertools import pairwise
from statistics import fmean, mean
from typing import Any

from . import VERSION
from .accounting import cache_neutral_cost, correction_metadata, effective_calls
from .contracts import BENCHMARK_WEIGHTS, STATELESS, planned_cells
from .failures import first_saved_failure
from .grading import (
    BINARY_RESULT,
    CACHE_OBSERVATION_RESULT,
    TOKEN_COVERAGE_RESULT,
    WEIGHTED_RUBRIC_RESULT,
    ResultKind,
    frozen_result_kind,
    require_binary_results,
    validate_result,
)
from .native_output import model_limits
from .target_contracts import effective_auxiliary_targets, target_inventory
from .transport import CACHE_REPORTING_STATES, TOOL_LOOP_PHASE

BUCKETS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")
COMPARABLE_STATUSES = frozenset({"completed", "failed"})
MIN_CONTINUITY_REQUESTS = 2


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0, center - half), min(1, center + half)]


def paired_conservative_interval(delta, groups, weights, alpha=0.05):
    """Hoeffding bound for independent paired differences in [-1, 1]."""
    squared_weights = sum(weights[b] ** 2 / len(values) for b, values in groups.items())
    radius = math.sqrt(2 * math.log(2 / alpha) * squared_weights)
    return [max(-1, delta - radius), min(1, delta + radius)]


def sum_cost(calls):
    if not calls or any(not _known_cost(c.get("cost_usd")) for c in calls):
        return None
    return sum(c["cost_usd"] for c in calls)


def _known_cost(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def output_diagnostics(results, subject_calls, total):
    """Count saved checks and call endings, without inferring absent output text."""
    details = [row.get("details") or {} for row in results]
    format_checks = [
        row["strict_format"]
        for row in details
        if isinstance(row.get("strict_format"), bool)
    ]
    finish_reasons = Counter(
        call["finish_reason"]
        for call in subject_calls
        if isinstance(call.get("finish_reason"), str) and call["finish_reason"]
    )
    return {
        "planned_cases": total,
        "result_cases": len(results),
        # This is a count of recorded flags, not proof that all other outputs
        # were complete. Missing result/check fields remain unassessed below.
        "output_limit_cases": sum(
            row.get("quality_failure") == "output_limit" for row in details
        ),
        "strict_format": {
            "checked_cases": len(format_checks),
            "failed_cases": format_checks.count(False),
            "unassessed_cases": total - len(format_checks),
        },
        "subject_calls": {
            "total": len(subject_calls),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "unknown_finish_reason": len(subject_calls) - sum(finish_reasons.values()),
        },
    }


def _observed_model(call: dict[str, Any]) -> str | None:
    if call.get("selected_model"):
        return call["selected_model"]
    if call.get("provider_model_observed") is False:
        return None
    return call.get("model") if call.get("status") == "completed" else None


def _changes(values: list[str]) -> int:
    return sum(before != after for before, after in pairwise(values))


def _task_accuracy(case_ids: list[str], correct: set[str]) -> float | None:
    if not case_ids:
        return None
    return sum(case_id in correct for case_id in case_ids) / len(case_ids)


def continuity(
    results: list[dict[str, Any]],
    subject_calls: list[dict[str, Any]],
    session_mode: str = STATELESS,
) -> dict[str, Any]:
    """Report model and decision changes within each task as facts, not penalties."""
    tasks: dict[str, list[dict]] = {}
    for call in subject_calls:
        if call.get("case_id"):
            tasks.setdefault(call["case_id"], []).append(call)
    correct = {
        case_id
        for r in results
        if r["status"] == "completed"
        and r.get("correct") is True
        and isinstance((case_id := r.get("case_id")), str)
    }
    switches: dict[str, int] = {}
    switches_by_phase: Counter[str] = Counter()
    decision_changed = unknown = multi_inference = 0
    for case_id, requests in tasks.items():
        if len(requests) < MIN_CONTINUITY_REQUESTS:
            continue
        # One request with several inference calls hides its own model sequence.
        if any((c.get("inference_call_count") or 0) > 1 for c in requests):
            multi_inference += 1
            continue
        known_requests = []
        for request in requests:
            model = _observed_model(request)
            if model:
                known_requests.append((model, request.get("phase")))
        models = [model for model, _ in known_requests]
        unknown += len(requests) - len(models)
        switches[case_id] = _changes(models)
        for before, after in pairwise(known_requests):
            if before[0] != after[0]:
                switches_by_phase[after[1] or "unknown"] += 1
        decision_changed += (
            _changes([c["decision"] for c in requests if c.get("decision")]) > 0
        )
    switched = [case_id for case_id, count in switches.items() if count]
    unswitched = [case_id for case_id, count in switches.items() if not count]
    result = {
        "session_mode": session_mode,
        "multi_request_tasks": len(switches),
        "switched_tasks": len(switched),
        "switched_accuracy": _task_accuracy(switched, correct),
        "unswitched_tasks": len(unswitched),
        "unswitched_accuracy": _task_accuracy(unswitched, correct),
        "model_switches": sum(switches.values()),
        "mean_switches_per_task": (
            sum(switches.values()) / len(switches) if switches else None
        ),
        "max_switches_per_task": max(switches.values(), default=None),
        "decision_changed_tasks": decision_changed,
        "unknown_model_requests": unknown,
        "multi_inference_tasks": multi_inference,
    }
    result["model_switches_by_phase"] = dict(sorted(switches_by_phase.items()))
    return result


def cache_read_metrics(subject_calls: list[dict]) -> dict:
    """Measure cache reads only for calls whose provider reported the field."""
    observations: list[dict] = []
    for call in subject_calls:
        model_usage = call.get("model_usage")
        if isinstance(model_usage, list) and model_usage:
            observations.extend(
                {
                    **item,
                    "usage": item.get("cache_read_usage", item.get("usage")),
                }
                for item in model_usage
                if isinstance(item, dict) and item.get("cache_read_reported") is True
            )
        elif call.get("cache_read_reported") is True:
            observations.append(
                {
                    **call,
                    "usage": call.get("cache_read_usage", call.get("usage")),
                }
            )

    if not observations:
        return {
            "cache_read_ratio": None,
            "cache_read_call_count": 0,
            "cache_read_tokens": 0,
            "cache_read_prompt_tokens": 0,
        }
    if any(not isinstance(call.get("usage"), dict) for call in observations):
        return {
            "cache_read_ratio": None,
            "cache_read_call_count": len(observations),
            "cache_read_tokens": None,
            "cache_read_prompt_tokens": None,
        }

    def token(observation: dict, name: str) -> int:
        value = observation["usage"].get(name, 0)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else 0
        )

    cached = sum(token(call, "cached_input_tokens") for call in observations)
    prompt = sum(
        sum(
            token(call, name)
            for name in ("input_tokens", "cached_input_tokens", "cache_write_tokens")
        )
        for call in observations
    )
    return {
        "cache_read_ratio": cached / prompt if prompt else None,
        "cache_read_call_count": len(observations),
        "cache_read_tokens": cached,
        "cache_read_prompt_tokens": prompt,
    }


def metric(
    target_id: str,
    results: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    total: int,
    *,
    planned_case_ids: Collection[str] | None = None,
    session_mode: str = STATELESS,
    result_kinds: dict[str, ResultKind] | None = None,
) -> dict[str, Any]:
    result_kinds = result_kinds or {}
    non_binary = any(kind != BINARY_RESULT for kind in result_kinds.values())
    completed = [r for r in results if r["status"] == "completed"]
    scored: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for row in completed:
        kind = result_kinds.get(row.get("benchmark", ""), BINARY_RESULT)
        if kind == CACHE_OBSERVATION_RESULT:
            validate_result(kind=kind, result=row)
            observed.append(row)
        elif kind != BINARY_RESULT:
            validate_result(kind=kind, result=row)
            scored.append(row)
        elif isinstance(row.get("correct"), bool):
            scored.append(row)
    correct = None if non_binary else sum(r["correct"] for r in scored)
    subject = [c for c in calls if c["role"] == "subject"]
    overhead = [c for c in calls if c["role"] != "subject"]
    receipted_cases = {c["case_id"] for c in subject if c.get("case_id")}
    subject_coverage_complete = len(receipted_cases) == total and all(
        r.get("case_id") in receipted_cases for r in results
    )
    if planned_case_ids is not None:
        subject_coverage_complete = (
            subject_coverage_complete and receipted_cases == set(planned_case_ids)
        )
    usage_complete = (
        bool(subject)
        and subject_coverage_complete
        and all(c.get("usage") is not None for c in subject)
    )
    cost_complete = (
        bool(subject)
        and subject_coverage_complete
        and all(_known_cost(c.get("cost_usd")) for c in subject)
    )
    tokens = (
        {k: sum((c.get("usage") or {}).get(k, 0) for c in subject) for k in BUCKETS}
        if usage_complete
        else None
    )
    latencies = [
        r.get("subject_latency_s", r["latency_s"])
        for r in completed
        if r.get("latency_s") is not None
    ]
    known = sum(c["cost_usd"] for c in subject if _known_cost(c.get("cost_usd")))
    result = {
        "id": target_id,
        "total": total,
        "completed": len(completed),
        "failed": sum(r["status"] not in {"completed", "running"} for r in results),
        "pending": total - len(results),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / total if not non_binary and total else None,
        "accuracy_denominator": (
            "binary accuracy is unavailable for non-binary or mixed cohorts"
            if non_binary
            else "all_planned_cases; failures and unanswered count as incorrect"
        ),
        "complete": len(completed) == total and len(scored) + len(observed) == total,
        "output_diagnostics": output_diagnostics(results, subject, total),
        "accuracy_ci95": None if non_binary else wilson(correct, total),
        "cost_usd": sum_cost(subject) if cost_complete else None,
        "cache_neutral_cost_usd": (
            sum(c["cache_neutral_cost_usd"] for c in subject)
            if subject
            and subject_coverage_complete
            and all(c.get("cache_neutral_cost_usd") is not None for c in subject)
            else None
        ),
        "cache_neutral_cost_basis": "Counterfactual token-equivalent subject cost: all prompt tokens at the frozen fresh-input rate plus output; not billed spend.",
        "known_cost_usd": known,
        "cost_complete": cost_complete,
        "evaluation_cost_usd": sum_cost(overhead) if overhead else 0,
        "total_spend_usd": sum_cost(calls) if subject_coverage_complete else None,
        "tokens": tokens,
        **cache_read_metrics(subject),
        "call_count": (
            sum(c["inference_call_count"] for c in subject)
            if subject
            and subject_coverage_complete
            and all(c.get("inference_call_count") is not None for c in subject)
            else None
        ),
        "request_count": len(subject),
        "selected_models": dict(
            Counter(model for model in map(_observed_model, subject) if model)
        ),
        "pending_selection_count": sum(
            c.get("status") in {"sent", "running"} and not c.get("selected_model")
            for c in subject
        ),
        "decisions": dict(Counter(c["decision"] for c in subject if c.get("decision"))),
        "continuity": continuity(results, subject, session_mode),
        "queue_wait_p50_s": percentile(
            [r["queue_wait_s"] for r in results if r.get("queue_wait_s") is not None],
            0.5,
        ),
        "queue_wait_p95_s": percentile(
            [r["queue_wait_s"] for r in results if r.get("queue_wait_s") is not None],
            0.95,
        ),
        "evaluation_call_count": len(overhead),
        "case_wall_time_p50_s": percentile(
            [r["latency_s"] for r in completed if r.get("latency_s") is not None], 0.5
        ),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "ttft_p50_s": percentile(
            [c["ttft_s"] for c in subject if c.get("ttft_s") is not None], 0.5
        ),
        "request_time_sum_s": sum(c.get("latency_s", 0) for c in subject),
    }
    if non_binary:
        result["result_kind"] = (
            "mixed"
            if len(set(result_kinds.values())) > 1
            else next(iter(result_kinds.values()))
        )
        result["continuity"]["switched_accuracy"] = None
        result["continuity"]["unswitched_accuracy"] = None
    if CACHE_OBSERVATION_RESULT in result_kinds.values():
        result["observed"] = len(observed)
    return result


def _rubric_cohort(
    cases: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    planned = {case["id"] for case in cases}
    rows = [row for row in results if row["case_id"] in planned]
    graded = [row for row in rows if row["status"] == "completed"]
    for row in graded:
        validate_result(kind=WEIGHTED_RUBRIC_RESULT, result=row)
    complete = bool(planned) and len(graded) == len(planned)
    return {
        "planned": len(planned),
        "graded": len(graded),
        "failed": sum(row["status"] == "failed" for row in rows),
        "cancelled": sum(row["status"] in {"cancelled", "interrupted"} for row in rows),
        "pending": len(planned)
        - len(rows)
        + sum(row["status"] in {"queued", "running"} for row in rows),
        "complete": complete,
        "mean_normalized": mean(row["score"] for row in graded) if complete else None,
        **{
            "mean_"
            + key: (
                mean(row["details"]["rubric"][key] for row in graded)
                if complete
                else None
            )
            for key in ("total", "positive_earned", "negative_penalty")
        },
        "frac_with_any_penalty": (
            mean(row["details"]["rubric"]["n_negative_triggered"] > 0 for row in graded)
            if complete
            else None
        ),
    }


def weighted_rubric_report(
    cases: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = {}
    case_domains: dict[str, str] = {}
    domain_results: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        domain = str(case.get("metadata", {}).get("stratum", "unknown"))
        domains.setdefault(domain, []).append(case)
        case_domains[case["id"]] = domain
    for result in results:
        if result["case_id"] in case_domains:
            domain_results.setdefault(case_domains[result["case_id"]], []).append(
                result
            )
    return {
        "version": WEIGHTED_RUBRIC_RESULT,
        "denominator": "all_planned_cases; incomplete cohorts have no aggregate quality",
        **_rubric_cohort(cases=cases, results=results),
        "per_domain": {
            domain: _rubric_cohort(cases=group, results=domain_results.get(domain, []))
            for domain, group in sorted(domains.items())
        },
    }


def token_coverage_report(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    planned = {case["id"] for case in cases}
    rows = [row for row in results if row["case_id"] in planned]
    graded = [row for row in rows if row["status"] == "completed"]
    for row in graded:
        validate_result(kind=TOKEN_COVERAGE_RESULT, result=row)
    complete = bool(planned) and len(graded) == len(planned)
    full_credit = sum(row["details"]["token_coverage"]["full_credit"] for row in graded)
    return {
        "version": TOKEN_COVERAGE_RESULT,
        "denominator": "all_planned_tasks; incomplete cohorts have no mean credit",
        "planned": len(planned),
        "graded": len(graded),
        "failed": sum(row["status"] == "failed" for row in rows),
        "cancelled": sum(row["status"] in {"cancelled", "interrupted"} for row in rows),
        "pending": len(planned)
        - len(rows)
        + sum(row["status"] in {"queued", "running"} for row in rows),
        "complete": complete,
        "mean_credit": (
            round(fmean(row["score"] for row in graded), 4) if complete else None
        ),
        "full_credit_tasks": full_credit,
        "full_completion_rate": (
            round(full_credit / len(planned), 4) if planned else None
        ),
        "full_completion_rate_denominator": (
            "all_planned_tasks; incomplete cohorts report an observed lower bound"
        ),
        **_scripted_task_evidence(cases=cases, calls=calls),
    }


def _scripted_task_evidence(
    cases: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        if call["role"] == "subject":
            by_case.setdefault(call["case_id"], []).append(call)
    turns: list[dict[str, Any]] = []
    tool_loop_switches = 0
    for case in cases:
        metadata = case["metadata"]
        task = metadata["scripted_task"]
        requests = by_case.get(case["id"], [])
        if len(requests) > len(task["turns"]):
            raise ValueError("Agent task receipts exceed the frozen scripted turns")
        previous_model: str | None = None
        for index, turn in enumerate(task["turns"]):
            call = requests[index] if index < len(requests) else {}
            model = _observed_model(call)
            if (
                previous_model
                and model
                and previous_model != model
                and turn["phase"] == TOOL_LOOP_PHASE
            ):
                tool_loop_switches += 1
            if model:
                previous_model = model
            turns.append(
                {
                    "case_id": case["id"],
                    "task": task["name"],
                    "task_suite": task["suite"],
                    "task_repetition": metadata["task_repetition"],
                    "turn": index,
                    "scripted_phase": turn["phase"],
                    "call_id": call.get("id"),
                    "status": call.get("status", "unattempted"),
                    "session_id": call.get("session_id"),
                    "selected_model": model,
                    "observed_phase": (
                        call.get("phase")
                        if call.get("phase_source") == "router"
                        else None
                    ),
                    "request_phase": (
                        call.get("phase")
                        if call.get("phase_source") == "request"
                        else None
                    ),
                    "phase_source": call.get("phase_source"),
                    "finish_reason": call.get("finish_reason"),
                    "error": call.get("error"),
                }
            )
    return {
        "task_count": len({turn["task"] for turn in turns}),
        "scripted_phase_counts": dict(
            Counter(turn["scripted_phase"] for turn in turns)
        ),
        "observed_router_phase_counts": dict(
            Counter(turn["observed_phase"] for turn in turns if turn["observed_phase"])
        ),
        "derived_request_phase_counts": dict(
            Counter(turn["request_phase"] for turn in turns if turn["request_phase"])
        ),
        "scripted_tool_loop_switches": tool_loop_switches,
        "turns": turns,
    }


def _cache_probe_validation(
    *,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    http_complete: bool,
    cache_problems: bool,
) -> dict[str, Any]:
    predicates: dict[str, Any] = {}
    for name, field in (
        ("min_success_rate", "success_rate"),
        ("min_cached_token_reporting", "cached_token_reporting"),
        ("min_cached_token_field_rate", "cached_token_field_rate"),
        ("min_cached_prompt_ratio", "cached_prompt_ratio"),
    ):
        required = thresholds[name]
        actual = metrics[field]
        reporting = name == "min_cached_token_reporting"
        enabled = required != "missing" if reporting else bool(required)
        reason: str | None = None
        if not enabled:
            status, reason = "pass", "threshold_disabled"
        elif name == "min_success_rate" and not http_complete:
            status, reason = "incomplete", "http_outcomes_incomplete"
        elif name != "min_success_rate" and (
            not metrics["execution_complete"] or cache_problems
        ):
            status, reason = "incomplete", "cache_evidence_incomplete"
        elif actual is None:
            status, reason = "incomplete", "required_observation_missing"
        else:
            sufficient = (
                CACHE_REPORTING_STATES.index(actual)
                >= CACHE_REPORTING_STATES.index(required)
                if reporting
                else actual >= required
            )
            status = "pass" if sufficient else "fail"
            if not sufficient:
                reason = "below_threshold"
        predicates[name] = {
            "enabled": enabled,
            "required": required,
            "actual": actual,
            "status": status,
            "reason": reason,
        }
    statuses = {predicate["status"] for predicate in predicates.values()}
    return {
        "status": (
            "fail"
            if "fail" in statuses
            else (
                "incomplete"
                if "incomplete" in statuses or not metrics["execution_complete"]
                else "pass"
            )
        ),
        "predicates": predicates,
    }


def cache_observation_report(
    *,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    thresholds: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        if call["role"] == "subject":
            by_case.setdefault(call["case_id"], []).append(call)
    result_by_case = {row["case_id"]: row for row in results}
    repeats: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    complete_results = True
    for case in cases:
        spec = case["metadata"]["cache_probe"]
        specs.append({"case_id": case["id"], **spec})
        requests = by_case.get(case["id"], [])
        if len(requests) > spec["repeats"]:
            raise ValueError("Cache probe receipts exceed the frozen repeat count")
        result = result_by_case.get(case["id"], {})
        if result.get("status") == "completed":
            validate_result(kind=CACHE_OBSERVATION_RESULT, result=result)
            if (
                result["details"]["cache_observation"]["planned_repeats"]
                != spec["repeats"]
            ):
                raise ValueError(
                    "Cache probe result differs from its frozen repeat count"
                )
        else:
            complete_results = False
        for index in range(spec["repeats"]):
            call = requests[index] if index < len(requests) else {}
            repeats.append(
                {
                    "case_id": case["id"],
                    "repeat_index": index,
                    "call_id": call.get("id"),
                    "status": call.get("status", "undispatched"),
                    "response_status": call.get("response_status"),
                    "output_complete": call.get("output_complete"),
                    "finish_reason": call.get("finish_reason"),
                    "latency_s": call.get("latency_s"),
                    "error": call.get("error"),
                    "session_id": call.get("session_id"),
                    "selected_model": _observed_model(call),
                    "config_hash": call.get("config_hash"),
                    "router_phase": (
                        call.get("phase")
                        if call.get("phase_source") == "router"
                        else None
                    ),
                    "phase_source": call.get("phase_source"),
                    "provider_cache_observation": call.get(
                        "provider_cache_observation"
                    ),
                }
            )
    completed = [
        row
        for row in repeats
        if row["status"] == "completed" and row["output_complete"] is True
    ]
    execution_complete = (
        bool(repeats) and complete_results and len(completed) == len(repeats)
    )
    successes = [
        row
        for row in repeats
        if type(row["response_status"]) is int
        and HTTPStatus.OK <= row["response_status"] < HTTPStatus.MULTIPLE_CHOICES
    ]
    observations = [
        row["provider_cache_observation"]
        for row in completed
        if isinstance(row["provider_cache_observation"], dict)
    ]
    receipts = [
        row["provider_cache_observation"]
        for row in repeats
        if isinstance(row["provider_cache_observation"], dict)
    ]
    cache_problems = len(observations) != len(completed) or any(
        row["qualification"] not in {"qualified", "missing"} or row.get("event_issues")
        for row in observations
    )
    states = [
        row["state"] for row in observations if row["state"] in CACHE_REPORTING_STATES
    ]
    usage_complete = (
        execution_complete
        and not cache_problems
        and len(observations) == len(repeats)
        and all(
            row["qualification"] == "qualified" and isinstance(row.get("usage"), dict)
            for row in observations
        )
    )
    tokens = (
        {key: sum(row["usage"][key] for row in observations) for key in BUCKETS}
        if usage_complete
        else None
    )
    prompt = (
        sum(tokens[key] for key in BUCKETS if key != "output_tokens")
        if tokens is not None
        else None
    )
    field_count = sum(row["field_present"] is True for row in observations)
    latencies = [
        row["latency_s"] * 1000 for row in completed if row["latency_s"] is not None
    ]
    metrics: dict[str, Any] = {
        "version": CACHE_OBSERVATION_RESULT,
        "denominator": "all_planned_repeats; incomplete or unqualified cohorts have no cache ratio",
        "target": {
            "id": target["id"],
            "kind": target["kind"],
            "base_url": target["base_url"],
            "requested_model": target["model"],
            "session_mode": target.get("session_mode", STATELESS),
            "expected_config_hash": target.get("config_hash"),
            "expected_response_model": target.get("expected_response_model"),
        },
        "native_probes": specs,
        "planned_repeats": len(repeats),
        "attempted_repeats": sum(row["call_id"] is not None for row in repeats),
        "completed_repeats": len(completed),
        "failed_repeats": sum(row["status"] == "failed" for row in repeats),
        "cancelled_repeats": sum(
            row["status"] in {"cancelled", "interrupted"} for row in repeats
        ),
        "undispatched_repeats": sum(row["status"] == "undispatched" for row in repeats),
        "execution_complete": execution_complete,
        "http_status_counts": dict(
            Counter(
                str(row["response_status"])
                for row in repeats
                if row["response_status"] is not None
            )
        ),
        "pending_repeats": sum(row["status"] in {"sent", "running"} for row in repeats),
        "unknown_repeats": sum(row["status"] == "sent_unknown" for row in repeats),
        "transport_status_counts": dict(Counter(row["status"] for row in repeats)),
        "successes": len(successes),
        "success_rate": round(len(successes) / len(repeats), 4) if repeats else None,
        "success_rate_basis": "all_planned_repeats; an observed lower bound while HTTP outcomes are missing",
        "cached_token_reporting": (
            max(states, key=CACHE_REPORTING_STATES.index) if states else None
        ),
        "cache_reporting_counts": dict(
            Counter(row["state"] or row["qualification"] for row in observations)
        ),
        "cached_token_field_present": field_count,
        "cached_token_field_rate": (
            round(field_count / len(successes), 4)
            if execution_complete and successes
            else None
        ),
        "cached_token_source_counts": dict(
            Counter(row["source"] for row in observations if row["source"])
        ),
        "evidence_reason_counts": dict(
            Counter(reason for row in receipts for reason in row["reasons"])
        ),
        "evidence_qualification_counts": dict(
            Counter(row["qualification"] for row in receipts)
        ),
        "prompt_tokens": prompt,
        "completion_tokens": tokens["output_tokens"] if tokens is not None else None,
        "cached_tokens": tokens["cached_input_tokens"] if tokens is not None else None,
        "cached_prompt_ratio": (
            round(tokens["cached_input_tokens"] / prompt, 6) if prompt else None
        ),
        "latency_ms": {
            "mean": round(fmean(latencies), 3) if latencies else None,
            **{
                name: round(value, 3) if value is not None else None
                for name, value in (
                    ("p50", percentile(latencies, 0.5)),
                    ("p95", percentile(latencies, 0.95)),
                    ("max", max(latencies, default=None)),
                )
            },
        },
        "repeats": repeats,
    }
    metrics["validation"] = _cache_probe_validation(
        metrics=metrics,
        thresholds=thresholds,
        http_complete=bool(repeats)
        and all(row["response_status"] is not None for row in repeats),
        cache_problems=cache_problems,
    )
    return metrics


def make_report(store, run_id):
    run = store.get(run_id)
    manifest = run["manifest"]
    results = store.results(run_id)
    calls = effective_calls(store, run_id, summary=True)
    calls = [
        {**call, "cache_neutral_cost_usd": cache_neutral_cost(manifest, call)}
        for call in calls
    ]
    metrics = []
    benchmarks = []
    cells = planned_cells(manifest)
    for target in manifest["targets"]:
        selected_ids = {
            cell["case_id"] for cell in cells if cell["target_id"] == target["id"]
        }
        rows = [r for r in results if r["target_id"] == target["id"]]
        tcalls = [c for c in calls if c["target_id"] == target["id"]]
        kinds = {
            c["benchmark"]: frozen_result_kind(
                manifest=manifest, benchmark=c["benchmark"]
            )
            for c in manifest["cases"]
            if c["id"] in selected_ids
        }
        session_mode: str = (
            target.get("session_mode", STATELESS)
            if manifest["mode"] == "live"
            else STATELESS
        )
        metrics.append(
            metric(
                target["id"],
                rows,
                tcalls,
                len(selected_ids),
                planned_case_ids=selected_ids,
                session_mode=session_mode,
                result_kinds={} if manifest["mode"] == "preview" else kinds,
            )
        )
        for benchmark in sorted({c["benchmark"] for c in manifest["cases"]}):
            ids = {
                c["id"]
                for c in manifest["cases"]
                if c["benchmark"] == benchmark and c["id"] in selected_ids
            }
            if not ids:
                continue
            benchmarks.append(
                {
                    "benchmark": benchmark,
                    "target_id": target["id"],
                    **metric(
                        target["id"],
                        [r for r in rows if r["case_id"] in ids],
                        [c for c in tcalls if c["case_id"] in ids],
                        len(ids),
                        planned_case_ids=ids,
                        session_mode=session_mode,
                        result_kinds=(
                            {}
                            if manifest["mode"] == "preview"
                            else {benchmark: kinds[benchmark]}
                        ),
                    ),
                }
            )
            if (
                kinds[benchmark] == WEIGHTED_RUBRIC_RESULT
                and manifest["mode"] != "preview"
            ):
                benchmarks[-1]["weighted_rubric"] = weighted_rubric_report(
                    cases=[c for c in manifest["cases"] if c["id"] in ids],
                    results=rows,
                )
                if len(kinds) == 1:
                    metrics[-1]["weighted_rubric"] = benchmarks[-1]["weighted_rubric"]
            if (
                kinds[benchmark] == TOKEN_COVERAGE_RESULT
                and manifest["mode"] != "preview"
            ):
                benchmarks[-1]["token_coverage"] = token_coverage_report(
                    cases=[c for c in manifest["cases"] if c["id"] in ids],
                    results=rows,
                    calls=tcalls,
                )
                if len(kinds) == 1:
                    metrics[-1]["token_coverage"] = benchmarks[-1]["token_coverage"]
            if (
                kinds[benchmark] == CACHE_OBSERVATION_RESULT
                and manifest["mode"] != "preview"
            ):
                benchmarks[-1]["cache_observation"] = cache_observation_report(
                    cases=[c for c in manifest["cases"] if c["id"] in ids],
                    results=rows,
                    calls=tcalls,
                    thresholds=manifest["benchmark_options"][benchmark]["thresholds"],
                    target=target,
                )
                if len(kinds) == 1:
                    metrics[-1]["cache_observation"] = benchmarks[-1][
                        "cache_observation"
                    ]
    selected_ids = {cell["case_id"] for cell in cells}
    present = {c["benchmark"] for c in manifest["cases"] if c["id"] in selected_ids}
    weights = manifest["benchmark_weights"]
    custom_subset = bool(
        "execution_cells" in manifest
        or (manifest.get("dataset") or {}).get("custom_subset", True)
    )
    extension_subset = not present.issubset(BENCHMARK_WEIGHTS)
    for item in metrics:
        rows = [row for row in benchmarks if row["target_id"] == item["id"]]
        target_full_suite = {row["benchmark"] for row in rows} == set(BENCHMARK_WEIGHTS)
        weight_sum = sum(weights[row["benchmark"]] for row in rows)
        macro = (
            sum(weights[row["benchmark"]] * row["accuracy"] for row in rows)
            / weight_sum
            if all(row["accuracy"] is not None for row in rows)
            else None
        )
        item["macro_accuracy"] = macro
        item["sr_bench_score"] = (
            macro
            if target_full_suite and item["complete"] and not manifest.get("recovery")
            else None
        )
        item["score_scope"] = (
            (
                "cache-observation diagnostic; no answer-quality score"
                if any("cache_observation" in row for row in rows)
                else (
                    "token-coverage diagnostic; no combined accuracy"
                    if any("token_coverage" in row for row in rows)
                    else (
                        "weighted-rubric diagnostic; no combined accuracy"
                        if any("weighted_rubric" in row for row in rows)
                        else (
                            "full sr-bench 1.0"
                            if target_full_suite
                            else (
                                "extension diagnostic; normalized adapter weights"
                                if extension_subset
                                else "selected benchmark subset; renormalized fixed weights"
                            )
                        )
                    )
                )
            )
            + f"; profile={manifest['profile']}"
            + ("; custom task subset" if custom_subset else "")
        )
    if manifest["mode"] in {"preview", "replay"}:
        routing_rows = (
            results
            if manifest["mode"] == "preview"
            else store.results(manifest["replay_sources"]["preview_run_id"])
        )
        for item in metrics + benchmarks:
            case_ids = {
                case["id"]
                for case in manifest["cases"]
                if not item.get("benchmark") or case["benchmark"] == item["benchmark"]
            }
            routing = [
                row.get("details", {}).get("routing", {})
                for row in routing_rows
                if row["target_id"] == item["id"] and row["case_id"] in case_ids
            ]
            for source, output in (
                ("selection_status", "selection_statuses"),
                ("selection_reason", "selection_reasons"),
            ):
                item[output] = dict(
                    Counter(row[source] for row in routing if row.get(source))
                )
            item["selected_models"] = dict(
                Counter(
                    row["selected_model"]
                    for row in routing
                    if row.get("selected_model")
                )
            )
            item["decisions"] = dict(
                Counter(
                    row["decision_result"]["decision_name"]
                    for row in routing
                    if (row.get("decision_result") or {}).get("decision_name")
                )
            )
    if manifest["mode"] == "preview":
        for item in metrics + benchmarks:
            for key in (
                "accuracy",
                "accuracy_ci95",
                "correct",
                "macro_accuracy",
                "sr_bench_score",
            ):
                if key in item:
                    item[key] = None
    if manifest["mode"] == "replay":
        for item in metrics + benchmarks:
            for key in (
                "accuracy",
                "macro_accuracy",
                "cost_usd",
                "cache_neutral_cost_usd",
                "latency_p50_s",
                "latency_p95_s",
                "ttft_p50_s",
            ):
                if key in item:
                    item["estimated_" + key] = item[key]
                    item[key] = None
            item["sr_bench_score"] = None
            item["total_spend_usd"] = 0
            item["cost_complete"] = False
    limitations = [
        "Scores apply only to this frozen sr-bench protocol and selected cases, not an upstream full benchmark score.",
        "Repeated tuning on development cases requires a separate untouched holdout.",
        "Self-hosted token-equivalent cost is not a measured hardware invoice.",
        "Wilson intervals describe case uncertainty; they do not account for source contamination or tuning selection.",
    ]
    if any("weighted_rubric" in benchmark for benchmark in benchmarks):
        limitations.append(
            "Weighted rubric diagnostics retain signed totals and penalties. Missing or failed grades are not zero; aggregate quality requires every planned grade. Binary accuracy, Wilson intervals and a combined sr-bench score do not describe these cohorts."
        )
    if any("cache_observation" in benchmark for benchmark in benchmarks):
        limitations.append(
            "Cache probes observe response fields, not answer quality or cache efficiency. "
            "Their complete-cohort ratios are separate from generic reported-subset cache metrics "
            "and existing billing. Unverified legacy aliases cannot qualify cache-dependent thresholds. "
            "Streaming and strict stopping differ from the retained buffered probe; "
            "target URLs, session IDs and latency alone do not prove model equivalence or cache reuse."
        )
    if manifest["mode"] == "replay":
        limitations.append(
            "Offline single-selection replay reuses saved answers, token counts and latencies. It makes zero model requests and is not a formal live score or measured routing performance."
        )
    if manifest["mode"] == "preview":
        limitations.append(
            "Preview evaluates routing only; it does not measure answer quality."
        )
    if any(m["total_spend_usd"] is None for m in metrics):
        limitations.append(
            "Some subject or auxiliary call costs are unknown; total cost savings cannot be claimed for those targets."
        )
    if run["status"] != "completed":
        limitations.append(
            "Run is incomplete; planned denominators include missing cases."
        )
    if manifest.get("recovery"):
        limitations.append(
            "This is a separate recovery subset attempt. Parent measurements and spend are inherited context only; child completion does not complete the original benchmark."
        )
    if manifest["cost_policy"] == "capability_only":
        limitations.append(
            "Capability-only protocol: USD budget cannot bound missing or unpriced usage; wall time, output and call caps still apply."
        )
    limitations.append(
        "Cost reservations use conservative serialized-input bounds and frozen inference-call limits; provider billing outside reported token buckets is excluded."
    )
    metadata = manifest.get("limitations", [])
    limitations.extend(metadata)
    accounting = correction_metadata(store, run_id)
    if accounting:
        limitations.append(
            "Accounting was reconciled offline from saved stream evidence and frozen prices. Original call/result detail receipts are unchanged; report totals use the versioned correction."
        )
    runner = store.provenance(run_id)
    if runner is None:
        limitations.append(
            "Runner source and dependency provenance was not captured for this saved run; it is unknown."
        )
    if any(c["benchmark"] == "gpqa-diamond" for c in manifest["cases"]):
        limitations.append(
            "GPQA labels were previously seen in this project; this is a retest, not an unseen holdout claim."
        )
    preparation = (manifest.get("dataset") or {}).get("preparation", {})
    if any(entry["history_snapshot"] is not None for entry in preparation.values()):
        limitations.append(
            "History exclusions cover only the named frozen dataset/run memberships. They do not certify complete prior generation, browsing, human exposure or upstream contamination history."
        )
    retests = sorted(
        family
        for family, entry in preparation.items()
        if entry["evaluation_role"] == "retest"
    )
    if retests:
        limitations.append(
            "Explicit retest families included in this report: "
            + ", ".join(retests)
            + ". Aggregate scores including them are not an unseen holdout aggregate."
        )
    if manifest["cost_policy"] == "require_priced":
        limitations.append(
            "Dispatch reservations use frozen request estimates. Provider or router prompt/output transformations can exceed those estimates; max_cost_usd stops future dispatch from measured spend but is not a universal hard billing cap."
        )
    wall = (
        datetime.fromisoformat(run["updated_at"])
        - datetime.fromisoformat(run["created_at"])
    ).total_seconds()
    return {
        "version": VERSION,
        "run_id": run_id,
        "status": run["status"],
        "mode": manifest["mode"],
        "cost_policy": manifest["cost_policy"],
        "failure": store.first_failure(run_id) or first_saved_failure(results),
        "recovery": manifest.get("recovery"),
        "child_attempts": [
            {
                "id": child["id"],
                "status": child["status"],
                "progress": child["progress"],
            }
            for child in store.children(run_id)
        ],
        "summary": {
            "targets": metrics,
            "wall_time_s": wall,
            "total_spend_usd": (
                0
                if manifest["mode"] == "replay"
                else (
                    sum_cost(calls)
                    if all(item["total_spend_usd"] is not None for item in metrics)
                    else None
                )
            ),
        },
        "benchmarks": benchmarks,
        "limitations": limitations,
        "provenance": {
            "accounting_correction": accounting,
            "runner": runner,
            "plan_sha256": manifest["plan_sha256"],
            "case_sha256": manifest["case_sha256"],
            "dataset": manifest.get("dataset"),
            "sampling": manifest["sampling"],
            "adapter_versions": manifest.get("adapter_versions", {}),
            **(
                {"result_kinds": manifest["result_kinds"]}
                if manifest.get("result_kinds")
                else {}
            ),
            "custom_subset": custom_subset,
            "target_profiles": [
                {
                    "id": t["id"],
                    "model": t["model"],
                    "kind": t["kind"],
                    "request_params": t.get("request_params", {}),
                    "config_hash": t.get("config_hash"),
                }
                for t in manifest["targets"]
            ],
            "profile": manifest["profile"],
            "seed": manifest["seed"],
        },
    }


def _comparison_protocol(baseline, candidate):
    if baseline.get("output_policy", "bounded") != candidate.get(
        "output_policy", "bounded"
    ):
        raise ValueError("Cannot compare different output_policy")
    if baseline.get("output_policy", "bounded") == "native" and model_limits(
        baseline
    ) != model_limits(candidate):
        raise ValueError("Cannot compare changed frozen native model limits")
    for key in (
        "case_sha256",
        "sampling",
        "benchmark_options",
        "benchmark_weights",
        "adapter_versions",
        "result_kinds",
        "mode",
        "limits",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Cannot compare different {key}")

    if effective_auxiliary_targets(baseline) != effective_auxiliary_targets(candidate):
        raise ValueError("Cannot compare changed effective judge/simulator targets")

    prices, profiles = {}, {}
    for manifest in (baseline, candidate):
        for target in target_inventory(manifest).values():
            for model, price in target.get("prices", {}).items():
                if model in prices and prices[model] != price:
                    raise ValueError("Cannot compare changed frozen model prices")
                prices[model] = price
            if target["kind"] == "single":
                model = target["model"]
                profile = {**manifest["sampling"], **target.get("request_params", {})}
                if model in profiles and profiles[model] != profile:
                    raise ValueError(
                        "Cannot compare changed single-model request parameters"
                    )
                profiles[model] = profile


def _exact_quality(report, target, weights):
    rows = [row for row in report["benchmarks"] if row["target_id"] == target]
    weighted = [
        (
            Fraction(str(weights[row["benchmark"]])),
            Fraction(row["correct"], row["total"]),
        )
        for row in rows
    ]
    return sum(w * score for w, score in weighted) / sum(w for w, _ in weighted)


def _strongest_single(singles, report, weights):
    """Break exact quality ties by complete subject plus auxiliary spend."""
    metrics = {row["id"]: row for row in report["summary"]["targets"]}
    quality = {
        target["id"]: _exact_quality(report, target["id"], weights)
        for target in singles
    }
    maximum = max(quality.values())
    tied = sorted(target for target, score in quality.items() if score == maximum)

    def priced(target):
        item = metrics[target]
        return item["total_spend_usd"] is not None

    best = min(
        tied,
        key=lambda target: (
            not priced(target),
            metrics[target]["total_spend_usd"] if priced(target) else math.inf,
            target,
        ),
    )
    return best, tied, all(priced(target) for target in tied)


def _cost_saving_percent(baseline, candidate, eligible):
    if eligible and baseline is not None and baseline > 0 and candidate is not None:
        return (1 - candidate / baseline) * 100
    return None


class ComparisonValidator:
    """Shared protocol and explicit terminal outcomes for discovery and submit."""

    def __init__(self, store, baseline):
        self.store = store
        self.baseline = baseline
        bm = baseline["manifest"]
        require_binary_results(manifest=bm, operation="Comparison")
        if baseline["status"] not in COMPARABLE_STATUSES or bm["mode"] != "live":
            raise ValueError("Baseline must be a completed or failed live run")
        self.singles = [t for t in bm["targets"] if t["kind"] == "single"]
        if not self.singles:
            raise ValueError("Baseline must contain a single-model target")
        planned = planned_cells(bm)
        selected = {
            target["id"]: {
                c["case_id"] for c in planned if c["target_id"] == target["id"]
            }
            for target in bm["targets"]
        }
        self.expected = selected[self.singles[0]["id"]]
        if not self.expected or any(ids != self.expected for ids in selected.values()):
            raise ValueError(
                "Baseline targets must cover identical nonempty selected cases"
            )
        self.case_ids = [c["id"] for c in bm["cases"] if c["id"] in self.expected]
        results = store.results(baseline["id"])
        self.by_target = self._quality_rows(results, bm["targets"], "Baseline")

    def _quality_rows(self, results, targets, label):
        by_target = {target["id"]: {} for target in targets}
        for result in results:
            rows = by_target.get(result["target_id"])
            if rows is None:
                continue
            case_id = result["case_id"]
            status = result["status"]
            if (
                case_id in rows
                or case_id not in self.expected
                or status not in COMPARABLE_STATUSES
                or (
                    status == "completed"
                    and not isinstance(result.get("correct"), bool)
                )
            ):
                raise ValueError(
                    f"{label} quality results lack unique terminal outcomes"
                )
            # A failed outcome is incorrect under the existing report denominator.
            # This derived score never changes its saved status, grade or receipt.
            rows[case_id] = {
                **result,
                "correct": result["correct"] if status == "completed" else False,
            }
        if any(set(rows) != self.expected for rows in by_target.values()):
            raise ValueError(f"{label} quality results are incomplete")
        return by_target

    def validate(self, candidate):
        require_binary_results(manifest=candidate["manifest"], operation="Comparison")
        if self.baseline["id"] == candidate["id"]:
            raise ValueError("Choose two distinct runs for comparison")
        _comparison_protocol(self.baseline["manifest"], candidate["manifest"])
        if candidate["status"] not in COMPARABLE_STATUSES:
            raise ValueError("Both runs must be completed or failed before comparison")
        planned = planned_cells(candidate["manifest"])
        if any(
            {cell["case_id"] for cell in planned if cell["target_id"] == target["id"]}
            != self.expected
            for target in candidate["manifest"]["targets"]
        ):
            raise ValueError("Candidate targets must cover identical planned cases")
        return self._quality_rows(
            self.store.results(candidate["id"]),
            candidate["manifest"]["targets"],
            "Candidate",
        )


def compare(store, baseline_id, candidate_id):
    baseline = store.get(baseline_id)
    candidate = store.get(candidate_id)
    validator = ComparisonValidator(store, baseline)
    candidate_rows = validator.validate(candidate)
    bm, cm = baseline["manifest"], candidate["manifest"]
    singles, expected = validator.singles, validator.expected
    case_ids, by_target = validator.case_ids, validator.by_target
    report_b = make_report(store, baseline_id)
    report_c = make_report(store, candidate_id)
    metric_by_id = {t["id"]: t for t in report_b["summary"]["targets"]}
    best, tied, tied_costs_complete = _strongest_single(
        singles, report_b, bm["benchmark_weights"]
    )
    base = by_target[best]
    base_metric = metric_by_id[best]
    priced_policy = bm["cost_policy"] == cm["cost_policy"] == "require_priced"
    cost_eligible = tied_costs_complete and priced_policy
    baseline_cost_reason = (
        None
        if tied_costs_complete
        else "At least one strongest single-model reference has incomplete total spend (subject plus auxiliary); the cheapest strongest baseline cannot be established."
    )
    comparisons = []
    for target in cm["targets"]:
        rows = candidate_rows[target["id"]]
        diffs = [int(rows[i]["correct"]) - int(base[i]["correct"]) for i in case_ids]
        groups = {
            b: [
                int(rows[c["id"]]["correct"]) - int(base[c["id"]]["correct"])
                for c in bm["cases"]
                if c["benchmark"] == b and c["id"] in expected
            ]
            for b in sorted(
                {c["benchmark"] for c in bm["cases"] if c["id"] in expected}
            )
        }
        weights = {
            b: bm["benchmark_weights"][b]
            / sum(bm["benchmark_weights"][k] for k in groups)
            for b in groups
        }
        macro_delta = float(
            _exact_quality(report_c, target["id"], cm["benchmark_weights"])
            - _exact_quality(report_b, best, bm["benchmark_weights"])
        )
        rng = random.Random(20260918)
        samples = sorted(
            sum(
                weights[b] * mean(rng.choices(values, k=len(values)))
                for b, values in groups.items()
            )
            for _ in range(2000)
        )
        cand_metric = next(
            t for t in report_c["summary"]["targets"] if t["id"] == target["id"]
        )
        bc = base_metric["cost_usd"]
        cc = cand_metric["cost_usd"]
        total_base = base_metric["total_spend_usd"]
        total_candidate = cand_metric["total_spend_usd"]
        neutral_base = base_metric["cache_neutral_cost_usd"]
        neutral_candidate = cand_metric["cache_neutral_cost_usd"]
        total_cost_reason = baseline_cost_reason
        if total_cost_reason is None and not priced_policy:
            total_cost_reason = (
                "Total cost savings require priced accounting for both runs."
            )
        if total_cost_reason is None and total_candidate is None:
            total_cost_reason = "Candidate total spend is incomplete: every planned subject case and every actual auxiliary call require known costs."
        if total_cost_reason is None and total_base == 0:
            total_cost_reason = (
                "A zero-cost baseline has no defined percentage savings."
            )
        comparisons.append(
            {
                "baseline_target_id": best,
                "candidate_target_id": target["id"],
                "paired_cases": len(diffs),
                "quality_delta": macro_delta,
                "micro_quality_delta": mean(diffs),
                "quality_metric": "fixed-weight benchmark macro accuracy",
                "quality_delta_ci95": paired_conservative_interval(
                    macro_delta, groups, weights
                ),
                "quality_delta_ci95_method": "weighted-paired-hoeffding",
                "quality_delta_ci95_qualification": "Case-independent bounded-difference interval with frozen benchmark weights; excludes strongest-baseline-selection, source contamination and tuning-selection uncertainty.",
                "quality_delta_bootstrap_ci95": [samples[49], samples[1949]],
                "subject_cost_saving_percent": _cost_saving_percent(
                    bc, cc, cost_eligible
                ),
                "baseline_subject_cost_usd": bc,
                "candidate_subject_cost_usd": cc,
                "total_cost_saving_percent": _cost_saving_percent(
                    total_base, total_candidate, cost_eligible
                ),
                "baseline_total_cost_usd": total_base,
                "candidate_total_cost_usd": total_candidate,
                "baseline_evaluation_cost_usd": base_metric["evaluation_cost_usd"],
                "candidate_evaluation_cost_usd": cand_metric["evaluation_cost_usd"],
                "total_cost_comparison_reason": total_cost_reason,
                "cache_neutral_baseline_cost_usd": neutral_base,
                "cache_neutral_candidate_cost_usd": neutral_candidate,
                "cache_neutral_cost_saving_percent": _cost_saving_percent(
                    neutral_base, neutral_candidate, cost_eligible
                ),
                "cache_neutral_cost_basis": "Subject-only counterfactual fresh-input token-equivalent cost against the same selected baseline; excludes auxiliary costs and cache-read/write discounts and premiums, not measured billing.",
                "wins": sum(d > 0 for d in diffs),
                "losses": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs),
            }
        )
    return {
        "version": VERSION,
        "baseline_run_id": baseline_id,
        "candidate_run_id": candidate_id,
        "baseline_status": baseline["status"],
        "candidate_status": candidate["status"],
        "baseline_quality_complete": all(
            t["complete"] for t in report_b["summary"]["targets"]
        ),
        "candidate_quality_complete": all(
            t["complete"] for t in report_c["summary"]["targets"]
        ),
        "baseline_targets": report_b["summary"]["targets"],
        "candidate_targets": report_c["summary"]["targets"],
        "quality_denominator": "all_planned_cases; explicit failed outcomes count as incorrect",
        "baseline_selection": "best observed single model on identical planned cases; explicit failures count as incorrect; selection uncertainty is not included",
        "baseline_selection_qualification": "This ranks delivered outcomes under the frozen limits, not model capability without execution failures. All single-model targets remain in the ranking, including failed outcomes and unknown costs.",
        "baseline_selected_target_id": best,
        "baseline_tied_best_target_ids": tied,
        "baseline_tie_policy": "Exact frozen weighted quality; ties prefer the lowest complete known total spend (subject plus auxiliary), then stable target ID. Unknown-total ties rank after known totals and suppress all savings claims.",
        "baseline_cost_comparison_eligible": tied_costs_complete,
        "baseline_cost_comparison_reason": baseline_cost_reason,
        "comparisons": comparisons,
    }
