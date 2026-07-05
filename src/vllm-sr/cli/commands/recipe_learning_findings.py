"""Finding generation for offline Router Learning recipe analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cli.commands.recipe_learning_metrics import numeric, optional_numeric_or_none

_SUPPORT_REPLAY_LIMIT = 10
_ROUTE_CORRECTNESS_HIGH_THRESHOLD = 0.8
_MODEL_FIT_HIGH_THRESHOLD = 0.8
_UNDERPOWERED_HIGH_RATE = 0.25
_MODEL_FAILURE_HIGH_RATE = 0.10
_EXCESSIVE_SWITCH_RATE = 0.35
_EXCESSIVE_HOLD_RATE = 0.60
_BROAD_CANDIDATE_RATE = 0.25
_BROAD_CANDIDATE_SWITCH_RATE = 0.20
_LOW_OUTCOME_COVERAGE_RATE = 0.20


@dataclass(frozen=True)
class DecisionFindingContext:
    decision: str
    metrics: dict[str, Any]
    records: int
    model_outcomes: dict[str, Any]
    support: dict[str, Any]
    route_correctness: float | None
    model_fit: float | None
    underpowered: int
    overprovisioned: int
    failed: int


def build_recipe_learning_findings(
    metrics_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for decision, metrics in metrics_payload.get("per_decision", {}).items():
        ctx = decision_finding_context(decision, metrics)
        if ctx is None:
            continue
        findings.extend(build_findings_for_decision(ctx))
    return findings


def decision_finding_context(
    decision: str, metrics: Any
) -> DecisionFindingContext | None:
    if not isinstance(metrics, dict):
        return None
    records = int(metrics.get("records") or 0)
    if records == 0:
        return None
    model_outcomes = metrics.get("outcomes", {}).get("model", {})
    if not isinstance(model_outcomes, dict):
        model_outcomes = {}
    support = metrics.get("supporting_replay_ids", {})
    if not isinstance(support, dict):
        support = {}
    return DecisionFindingContext(
        decision=decision,
        metrics=metrics,
        records=records,
        model_outcomes=model_outcomes,
        support=support,
        route_correctness=optional_numeric_or_none(metrics.get("route_correctness")),
        model_fit=optional_numeric_or_none(metrics.get("model_fit")),
        underpowered=int(model_outcomes.get("underpowered") or 0),
        overprovisioned=int(model_outcomes.get("overprovisioned") or 0),
        failed=int(model_outcomes.get("failed") or 0),
    )


def build_findings_for_decision(ctx: DecisionFindingContext) -> list[dict[str, Any]]:
    builders: tuple[
        Callable[[DecisionFindingContext], dict[str, Any] | None],
        ...,
    ] = (
        wrong_decision_finding,
        wrong_model_selection_finding,
        underpowered_model_finding,
        model_overuse_finding,
        model_reliability_finding,
        provider_reliability_finding,
        excessive_switching_finding,
        excessive_holds_finding,
        missing_protection_finding,
        overly_broad_candidate_set_finding,
        latency_violation_finding,
        cost_violation_finding,
        low_outcome_coverage_finding,
    )
    return [item for builder in builders if (item := builder(ctx)) is not None]


def wrong_decision_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.route_correctness is None or ctx.route_correctness >= 1:
        return None
    return finding(
        "wrong_decision",
        (
            "high"
            if ctx.route_correctness < _ROUTE_CORRECTNESS_HIGH_THRESHOLD
            else "medium"
        ),
        ctx.decision,
        "Eval cases indicate this route selected the wrong decision for some replay records.",
        {
            "route_correctness": ctx.route_correctness,
            "route_evaluated": ctx.metrics.get("route_evaluated", 0),
            "replay_ids": ctx.support.get("wrong_decision", []),
        },
        "Review this decision's rules, examples, thresholds, priority, or the expected decision's priority.",
    )


def wrong_model_selection_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.model_fit is None or ctx.model_fit >= 1:
        return None
    return finding(
        "wrong_model_selection",
        "high" if ctx.model_fit < _MODEL_FIT_HIGH_THRESHOLD else "medium",
        ctx.decision,
        "Eval cases indicate the selected model did not match expected model behavior.",
        {
            "model_fit": ctx.model_fit,
            "model_evaluated": ctx.metrics.get("model_evaluated", 0),
            "replay_ids": ctx.support.get("wrong_model", []),
        },
        "Review this decision's modelRefs or use adaptation over a broader candidate set for this route.",
    )


def underpowered_model_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.underpowered:
        return None
    return finding(
        "underpowered_model",
        (
            "high"
            if ctx.underpowered / ctx.records >= _UNDERPOWERED_HIGH_RATE
            else "medium"
        ),
        ctx.decision,
        f"{ctx.underpowered} underpowered model outcome(s) indicate the selected model may be too weak or mismatched.",
        {
            "underpowered": ctx.underpowered,
            "records": ctx.records,
            "replay_ids": ctx.support.get("model_underpowered", []),
        },
        "Review this decision's modelRefs or allow a broader adaptation candidate_set for this route.",
    )


def model_overuse_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.overprovisioned:
        return None
    return finding(
        "model_overuse",
        "medium",
        ctx.decision,
        f"{ctx.overprovisioned} overprovisioned model outcome(s) indicate the selected model may be stronger or costlier than needed.",
        {
            "overprovisioned": ctx.overprovisioned,
            "records": ctx.records,
            "replay_ids": ctx.support.get("model_overprovisioned", []),
        },
        "Add a cheaper eligible model to the decision or use adaptation over the decision/tier candidate set.",
    )


def model_reliability_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.failed:
        return None
    return finding(
        "model_reliability",
        "high" if ctx.failed / ctx.records >= _MODEL_FAILURE_HIGH_RATE else "medium",
        ctx.decision,
        f"{ctx.failed} model failure outcome(s) indicate reliability risk.",
        {
            "failed": ctx.failed,
            "records": ctx.records,
            "replay_ids": ctx.support.get("model_failed", []),
        },
        "Check provider health, retry policy, and whether adaptation should avoid this model until reliability recovers.",
    )


def provider_reliability_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.metrics.get("provider_failures", 0):
        return None
    return finding(
        "provider_reliability",
        "high",
        ctx.decision,
        "Provider outcomes indicate backend reliability failures for this decision.",
        {
            "provider_failures": ctx.metrics["provider_failures"],
            "replay_ids": ctx.support.get("provider_failed", []),
        },
        "Check provider health and retry policy before changing route policy.",
    )


def excessive_switching_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.metrics.get("switch_rate", 0) <= _EXCESSIVE_SWITCH_RATE:
        return None
    return finding(
        "excessive_switching",
        "medium",
        ctx.decision,
        "Model switches are frequent for this decision.",
        {
            "switch_rate": ctx.metrics["switch_rate"],
            "records": ctx.records,
            "replay_ids": ctx.support.get("switches", []),
        },
        "Increase protection stability_weight for this decision or narrow the adaptation candidate set.",
    )


def excessive_holds_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.metrics.get("hold_rate", 0) <= _EXCESSIVE_HOLD_RATE or not ctx.underpowered:
        return None
    return finding(
        "excessive_holds",
        "medium",
        ctx.decision,
        "Protection often holds the model while model outcomes indicate underpowered behavior.",
        {
            "hold_rate": ctx.metrics["hold_rate"],
            "underpowered": ctx.underpowered,
            "replay_ids": ctx.support.get("holds", []),
        },
        "Lower protection stability_weight or add rescue-oriented evidence for this decision.",
    )


def missing_protection_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.metrics.get("missing_protection_rate", 0) <= 0:
        return None
    return finding(
        "missing_protection",
        "medium",
        ctx.decision,
        "Adaptation diagnostics exist without protection diagnostics for some records.",
        {
            "missing_protection_rate": ctx.metrics["missing_protection_rate"],
            "replay_ids": ctx.support.get("missing_protection", []),
        },
        "Enable protection globally or set this decision's protection mode to apply when continuity matters.",
    )


def overly_broad_candidate_set_finding(
    ctx: DecisionFindingContext,
) -> dict[str, Any] | None:
    broad_rate = numeric(ctx.metrics.get("broad_candidate_rate"))
    if broad_rate <= _BROAD_CANDIDATE_RATE:
        return None
    if not has_broad_candidate_risk_evidence(ctx):
        return None
    return finding(
        "overly_broad_candidate_set",
        "medium",
        ctx.decision,
        "A broad adaptation candidate set appears on records with quality, reliability, cost, latency, or switching risk evidence.",
        {
            "broad_candidate_rate": ctx.metrics["broad_candidate_rate"],
            "switch_rate": ctx.metrics.get("switch_rate", 0),
            "route_correctness": ctx.metrics.get("route_correctness"),
            "model_fit": ctx.metrics.get("model_fit"),
            "replay_ids": ctx.support.get("broad_candidate_set", []),
        },
        "Narrow this decision to candidate_set: decision unless cross-decision adaptation is intentional.",
    )


def has_broad_candidate_risk_evidence(ctx: DecisionFindingContext) -> bool:
    switch_rate = numeric(ctx.metrics.get("switch_rate"))
    if switch_rate > _BROAD_CANDIDATE_SWITCH_RATE:
        return True
    if ctx.overprovisioned or ctx.underpowered or ctx.failed:
        return True
    if ctx.route_correctness is not None and ctx.route_correctness < 1:
        return True
    if ctx.model_fit is not None and ctx.model_fit < 1:
        return True
    for key in ("cost_violations", "latency_violations", "provider_failures"):
        if numeric(ctx.metrics.get(key)) > 0:
            return True
    return False


def latency_violation_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.metrics.get("latency_violations", 0):
        return None
    return finding(
        "latency_violation",
        "medium",
        ctx.decision,
        "Eval cases observed responses above the expected latency budget.",
        {
            "latency_violations": ctx.metrics["latency_violations"],
            "avg_latency_ms": ctx.metrics.get("avg_latency_ms"),
            "replay_ids": ctx.support.get("latency_violations", []),
        },
        "Prefer a lower-latency candidate set or adjust this decision's modelRefs.",
    )


def cost_violation_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if not ctx.metrics.get("cost_violations", 0):
        return None
    return finding(
        "cost_violation",
        "medium",
        ctx.decision,
        "Eval cases observed routes above the expected cost budget.",
        {
            "cost_violations": ctx.metrics["cost_violations"],
            "actual_cost": ctx.metrics.get("actual_cost"),
            "replay_ids": ctx.support.get("cost_violations", []),
        },
        "Prefer a cheaper candidate set or adjust this decision's modelRefs.",
    )


def low_outcome_coverage_finding(ctx: DecisionFindingContext) -> dict[str, Any] | None:
    if ctx.metrics.get("outcome_coverage", 0) >= _LOW_OUTCOME_COVERAGE_RATE:
        return None
    return finding(
        "low_outcome_coverage",
        "low",
        ctx.decision,
        "Few replay records have typed outcomes, so online experience and offline recipe learning have weak evidence.",
        {
            "outcome_coverage": ctx.metrics.get("outcome_coverage", 0),
            "records": ctx.records,
        },
        "Submit model, route, policy, stability, provider, or router outcomes from agents and evals for this decision.",
    )


def finding(
    finding_type: str,
    severity: str,
    decision: str,
    message: str,
    evidence: dict[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    replay_ids = evidence.get("replay_ids")
    if not isinstance(replay_ids, list):
        replay_ids = []
    finding_id = stable_finding_id(finding_type, decision, replay_ids, evidence)
    return {
        "id": finding_id,
        "type": finding_type,
        "severity": severity,
        "decision": decision,
        "affected_decisions": [decision] if decision else [],
        "message": message,
        "evidence": evidence,
        "recommendation": recommendation,
        "next_action": recommendation,
    }


def stable_finding_id(
    finding_type: str,
    decision: str,
    replay_ids: list[Any],
    evidence: dict[str, Any],
) -> str:
    replay_key = ",".join(str(value) for value in replay_ids[:_SUPPORT_REPLAY_LIMIT])
    if not replay_key:
        evidence_key = json.dumps(evidence, sort_keys=True, default=str)
    else:
        evidence_key = replay_key
    digest = hashlib.sha256(
        f"{finding_type}|{decision}|{evidence_key}".encode()
    ).hexdigest()[:12]
    return f"rlf_{digest}"
