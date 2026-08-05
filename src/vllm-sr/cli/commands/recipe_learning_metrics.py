"""Metrics helpers for offline Router Learning recipe analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_SUPPORT_REPLAY_LIMIT = 10


@dataclass
class DecisionMetrics:
    records: int = 0
    switches: int = 0
    holds: int = 0
    allows: int = 0
    rescues: int = 0
    sampled: int = 0
    learning_records: int = 0
    missing_protection_records: int = 0
    broad_candidate_records: int = 0
    outcome_records: int = 0
    provider_failures: int = 0
    actual_cost: float = 0.0
    baseline_cost: float = 0.0
    cost_savings: float = 0.0
    latency_ms_total: float = 0.0
    latency_records: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_tokens: int = 0
    selected_models: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    route_correct: int = 0
    route_evaluated: int = 0
    model_correct: int = 0
    model_evaluated: int = 0
    cost_violations: int = 0
    latency_violations: int = 0
    supporting_replay_ids: dict[str, list[str]] = field(default_factory=dict)
    decision_tiers: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        switch_rate = self.switches / self.records if self.records else 0.0
        hold_rate = self.holds / self.records if self.records else 0.0
        rescue_rate = self.rescues / self.records if self.records else 0.0
        learning_coverage = (
            self.learning_records / self.records if self.records else 0.0
        )
        outcome_coverage = self.outcome_records / self.records if self.records else 0.0
        missing_protection_rate = (
            self.missing_protection_records / self.records if self.records else 0.0
        )
        broad_candidate_rate = (
            self.broad_candidate_records / self.records if self.records else 0.0
        )
        cache_rate = (
            self.cached_prompt_tokens / self.prompt_tokens
            if self.prompt_tokens
            else 0.0
        )
        avg_latency_ms = (
            self.latency_ms_total / self.latency_records
            if self.latency_records
            else None
        )
        route_correctness = (
            self.route_correct / self.route_evaluated if self.route_evaluated else None
        )
        model_fit = (
            self.model_correct / self.model_evaluated if self.model_evaluated else None
        )
        return {
            "records": self.records,
            "switch_rate": round(switch_rate, 4),
            "hold_rate": round(hold_rate, 4),
            "rescue_rate": round(rescue_rate, 4),
            "sampling_rate": (
                round(self.sampled / self.records, 4) if self.records else 0.0
            ),
            "learning_coverage": round(learning_coverage, 4),
            "outcome_coverage": round(outcome_coverage, 4),
            "missing_protection_rate": round(missing_protection_rate, 4),
            "broad_candidate_rate": round(broad_candidate_rate, 4),
            "cache_preservation": round(cache_rate, 4),
            "actual_cost": round(self.actual_cost, 6),
            "baseline_cost": round(self.baseline_cost, 6),
            "cost_savings": round(self.cost_savings, 6),
            "avg_latency_ms": (
                round(avg_latency_ms, 2) if avg_latency_ms is not None else None
            ),
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.total_tokens,
            "route_correctness": (
                round(route_correctness, 4) if route_correctness is not None else None
            ),
            "model_fit": round(model_fit, 4) if model_fit is not None else None,
            "route_evaluated": self.route_evaluated,
            "model_evaluated": self.model_evaluated,
            "cost_violations": self.cost_violations,
            "latency_violations": self.latency_violations,
            "provider_failures": self.provider_failures,
            "decision_tiers": dict(sorted(self.decision_tiers.items())),
            "selected_models": dict(sorted(self.selected_models.items())),
            "outcomes": {
                target: dict(sorted(verdicts.items()))
                for target, verdicts in sorted(self.outcomes.items())
            },
            "supporting_replay_ids": {
                key: values[:_SUPPORT_REPLAY_LIMIT]
                for key, values in sorted(self.supporting_replay_ids.items())
                if values
            },
        }


@dataclass(frozen=True)
class EvalCase:
    replay_id: str = ""
    request_id: str = ""
    expected_decision: str = ""
    expected_model: str = ""
    max_cost: float | None = None
    max_latency_ms: float | None = None


def update_metrics(
    metrics: DecisionMetrics, record: dict[str, Any], case_index: dict[str, EvalCase]
) -> None:
    metrics.records += 1
    replay_id = record_replay_id(record)
    decision_tier = record_decision_tier(record)
    selected_model = record_selected_model(record)
    actual_cost = record_actual_cost(record)
    latency_ms = record_latency_ms(record)

    update_decision_tier_metrics(metrics, decision_tier)
    update_model_selection_metrics(metrics, record, replay_id, selected_model)
    update_learning_metrics(metrics, record, replay_id)
    update_outcome_metrics(metrics, record, replay_id)
    update_cost_latency_token_metrics(metrics, record, actual_cost, latency_ms)
    update_eval_case_metrics(
        metrics,
        record,
        case_index,
        replay_id,
        selected_model,
        actual_cost,
        latency_ms,
    )


def update_model_selection_metrics(
    metrics: DecisionMetrics,
    record: dict[str, Any],
    replay_id: str,
    selected_model: str,
) -> None:
    if selected_model:
        metrics.selected_models[selected_model] = (
            metrics.selected_models.get(selected_model, 0) + 1
        )
    if record_switched(record):
        metrics.switches += 1
        append_support(metrics, "switches", replay_id)


def update_decision_tier_metrics(metrics: DecisionMetrics, decision_tier: int) -> None:
    key = tier_metrics_key(decision_tier)
    metrics.decision_tiers[key] = metrics.decision_tiers.get(key, 0) + 1


def update_learning_metrics(
    metrics: DecisionMetrics, record: dict[str, Any], replay_id: str
) -> None:
    learning = record.get("learning")
    if not isinstance(learning, dict):
        return
    metrics.learning_records += 1
    adaptation = learning.get("adaptation")
    if isinstance(adaptation, dict):
        update_adaptation_metrics(metrics, adaptation, replay_id)
    protection = learning.get("protection")
    if isinstance(protection, dict):
        update_protection_metrics(metrics, protection, replay_id)
    elif isinstance(adaptation, dict):
        metrics.missing_protection_records += 1
        append_support(metrics, "missing_protection", replay_id)


def update_adaptation_metrics(
    metrics: DecisionMetrics, adaptation: dict[str, Any], replay_id: str
) -> None:
    sampling = adaptation.get("sampling")
    if isinstance(sampling, dict) and sampling.get("used") is True:
        metrics.sampled += 1
    if adaptation.get("candidate_set") == "global":
        metrics.broad_candidate_records += 1
        append_support(metrics, "broad_candidate_set", replay_id)


def update_protection_metrics(
    metrics: DecisionMetrics, protection: dict[str, Any], replay_id: str
) -> None:
    action = protection.get("action")
    if action == "hold_current":
        metrics.holds += 1
        append_support(metrics, "holds", replay_id)
    elif action in {"allow_switch", "rescue_switch"}:
        metrics.allows += 1
        if action == "rescue_switch":
            metrics.rescues += 1
            append_support(metrics, "rescues", replay_id)


def update_outcome_metrics(
    metrics: DecisionMetrics, record: dict[str, Any], replay_id: str
) -> None:
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        return
    metrics.outcome_records += 1
    for outcome in outcomes:
        if isinstance(outcome, dict):
            update_single_outcome_metrics(metrics, outcome, replay_id)


def update_single_outcome_metrics(
    metrics: DecisionMetrics, outcome: dict[str, Any], replay_id: str
) -> None:
    target = str(outcome.get("target") or "unknown")
    verdict = str(outcome.get("verdict") or "unknown")
    bucket = metrics.outcomes.setdefault(target, {})
    bucket[verdict] = bucket.get(verdict, 0) + 1
    if target == "model" and verdict in {
        "underpowered",
        "overprovisioned",
        "failed",
    }:
        append_support(metrics, f"model_{verdict}", replay_id)
    if target == "provider" and verdict == "failed":
        metrics.provider_failures += 1
        append_support(metrics, "provider_failed", replay_id)


def update_cost_latency_token_metrics(
    metrics: DecisionMetrics,
    record: dict[str, Any],
    actual_cost: float,
    latency_ms: float | None,
) -> None:
    metrics.actual_cost += actual_cost
    metrics.baseline_cost += numeric(record.get("baseline_cost"))
    metrics.cost_savings += numeric(record.get("cost_savings"))
    if latency_ms is not None:
        metrics.latency_ms_total += latency_ms
        metrics.latency_records += 1
    metrics.prompt_tokens += int(numeric(record.get("prompt_tokens")))
    metrics.cached_prompt_tokens += int(numeric(record.get("cached_prompt_tokens")))
    metrics.total_tokens += int(numeric(record.get("total_tokens")))


def update_eval_case_metrics(
    metrics: DecisionMetrics,
    record: dict[str, Any],
    case_index: dict[str, EvalCase],
    replay_id: str,
    selected_model: str,
    actual_cost: float,
    latency_ms: float | None,
) -> None:
    case = case_for_record(case_index, record)
    if case is None:
        return
    update_route_eval_metrics(metrics, record, case, replay_id)
    update_model_eval_metrics(metrics, case, replay_id, selected_model)
    update_budget_eval_metrics(metrics, case, replay_id, actual_cost, latency_ms)


def update_route_eval_metrics(
    metrics: DecisionMetrics,
    record: dict[str, Any],
    case: EvalCase,
    replay_id: str,
) -> None:
    if not case.expected_decision:
        return
    metrics.route_evaluated += 1
    if case.expected_decision == record_decision(record):
        metrics.route_correct += 1
    else:
        append_support(metrics, "wrong_decision", replay_id)


def update_model_eval_metrics(
    metrics: DecisionMetrics,
    case: EvalCase,
    replay_id: str,
    selected_model: str,
) -> None:
    if not case.expected_model:
        return
    metrics.model_evaluated += 1
    if case.expected_model == selected_model:
        metrics.model_correct += 1
    else:
        append_support(metrics, "wrong_model", replay_id)


def update_budget_eval_metrics(
    metrics: DecisionMetrics,
    case: EvalCase,
    replay_id: str,
    actual_cost: float,
    latency_ms: float | None,
) -> None:
    if case.max_cost is not None and actual_cost > case.max_cost:
        metrics.cost_violations += 1
        append_support(metrics, "cost_violations", replay_id)
    if (
        case.max_latency_ms is not None
        and latency_ms is not None
        and latency_ms > case.max_latency_ms
    ):
        metrics.latency_violations += 1
        append_support(metrics, "latency_violations", replay_id)


def update_experience_counts(
    experience_counts: dict[tuple[str, int, str], dict[str, int]],
    record: dict[str, Any],
) -> None:
    decision = record_decision(record)
    tier = record_decision_tier(record)
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict) or outcome.get("target") != "model":
            continue
        model = str(outcome.get("target_ref") or record_selected_model(record)).strip()
        verdict = str(outcome.get("verdict") or "").strip()
        if not model or verdict not in {
            "good_fit",
            "underpowered",
            "overprovisioned",
            "failed",
        }:
            continue
        key = (decision, tier, model)
        counts = experience_counts.setdefault(
            key,
            {
                "good_fit": 0,
                "underpowered": 0,
                "overprovisioned": 0,
                "failed": 0,
            },
        )
        counts[verdict] += 1


def record_decision(record: dict[str, Any]) -> str:
    route = record.get("route_diagnostics")
    if isinstance(route, dict) and route.get("decision"):
        return str(route["decision"])
    return str(record.get("decision") or "unknown")


def record_decision_tier(record: dict[str, Any]) -> int:
    route = record.get("route_diagnostics")
    if isinstance(route, dict):
        for key in ("decision_tier", "tier"):
            if key in route:
                return int(numeric(route.get(key)))
    return int(numeric(record.get("decision_tier")))


def tier_metrics_key(decision_tier: int) -> str:
    return f"tier_{decision_tier}" if decision_tier > 0 else "unknown"


def record_replay_id(record: dict[str, Any]) -> str:
    for key in ("id", "replay_id", "request_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def record_selected_model(record: dict[str, Any]) -> str:
    route = record.get("route_diagnostics")
    if isinstance(route, dict) and route.get("selected_model"):
        return str(route["selected_model"])
    return str(record.get("selected_model") or "")


def record_switched(record: dict[str, Any]) -> bool:
    learning = record.get("learning")
    if isinstance(learning, dict):
        protection = learning.get("protection")
        if isinstance(protection, dict):
            action = str(protection.get("action") or "")
            if action in {"allow_switch", "rescue_switch"}:
                return True
            if action in {"establish_baseline", "hold_current", "bypass"}:
                return False
        adaptation = learning.get("adaptation")
        if isinstance(adaptation, dict):
            action = str(adaptation.get("action") or "")
            if action == "propose_switch":
                base = str(adaptation.get("base_model") or "")
                proposal = str(adaptation.get("proposal_model") or "")
                return bool(base and proposal and base != proposal)
            if action in {"keep_base", "observe", "bypass"}:
                return False

    route = record.get("route_diagnostics")
    if isinstance(route, dict):
        original = str(
            route.get("original_model") or record.get("original_model") or ""
        )
        selected = str(
            route.get("selected_model") or record.get("selected_model") or ""
        )
    else:
        original = str(record.get("original_model") or "")
        selected = str(record.get("selected_model") or "")
    if original.lower() in {"auto", "vllm-sr/auto"}:
        return False
    return bool(original and selected and original != selected)


def record_actual_cost(record: dict[str, Any]) -> float:
    return numeric(record.get("actual_cost"))


def case_for_record(
    case_index: dict[str, EvalCase], record: dict[str, Any]
) -> EvalCase | None:
    for key in (
        str(record.get("id") or ""),
        str(record.get("replay_id") or ""),
        str(record.get("request_id") or ""),
    ):
        if key and key in case_index:
            return case_index[key]
    return None


def record_latency_ms(record: dict[str, Any]) -> float | None:
    for key in ("latency_ms", "duration_ms", "response_latency_ms"):
        if key in record:
            return numeric(record.get(key))
    duration = record.get("duration")
    if isinstance(duration, str) and duration.endswith("ms"):
        return numeric(duration[:-2])
    if isinstance(duration, str) and duration.endswith("s"):
        return numeric(duration[:-1]) * 1000
    route = record.get("route_diagnostics")
    if isinstance(route, dict):
        for key in ("latency_ms", "duration_ms"):
            if key in route:
                return numeric(route.get(key))
    return None


def append_support(metrics: DecisionMetrics, key: str, replay_id: str) -> None:
    replay_id = replay_id.strip()
    if not replay_id:
        return
    bucket = metrics.supporting_replay_ids.setdefault(key, [])
    if replay_id not in bucket:
        bucket.append(replay_id)


def numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def optional_numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return numeric(value)


def optional_numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return numeric(value)
