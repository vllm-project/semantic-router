"""Production experiment controls and online preference gates G8-G9."""

from __future__ import annotations

from cli.evaluation.gate_context import GateEvidenceContext, GateRunMetadata, build_gate
from cli.evaluation.gate_contract import GateDefinition, GateDisposition
from cli.evaluation.reporting import EvaluationGate, GateThreshold


def evaluate_g8(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    if context.production_candidate_safe is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="No complete sealed production assignment/exposure ledger with SRM, risk-budget, stop, and rollback receipts was attached.",
        )
    risk_rate = context.production_risk_event_rate
    risk_upper_bound = context.production_risk_event_upper_confidence_bound
    risk_budget = context.production_risk_budget_max_rate
    if risk_rate is None or risk_upper_bound is None or risk_budget is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="The production ledger lacks its frozen risk budget or full-window risk confidence bound.",
        )
    support = context.production_assignment_support
    balance_p = context.production_balance_p_value
    passed = context.production_candidate_safe
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if passed else "fail",
        observed=risk_upper_bound,
        threshold=GateThreshold(operator="<=", value=risk_budget, unit="fraction"),
        rationale=(
            "The complete sealed production window passed policy-arm support, SRM, frozen risk-budget, minimum cohort, stop, and rollback-readiness controls; "
            f"point risk={risk_rate!r}, assignment support={support!r}, SRM p-value={balance_p!r}."
            if passed
            else "The complete sealed production window failed policy-arm support, SRM, frozen risk-budget, minimum cohort, stop, or rollback-readiness controls; "
            f"point risk={risk_rate!r}, assignment support={support!r}, SRM p-value={balance_p!r}."
        ),
    )


def evaluate_g9(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    if context.online_preference_qualified is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="No complete production outcome ledger with exposure, propensity, support, ESS, confidence, and segment coverage was attached.",
        )
    minimum_lift = context.online_minimum_reward_lift
    if minimum_lift is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="The production ledger does not freeze a target-vs-reference minimum reward lift.",
        )
    if context.online_causal_eligible is not True:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            threshold=GateThreshold(
                operator=">=", value=minimum_lift, unit="reward lift"
            ),
            rationale="The complete production outcome window failed support, propensity, ESS, or segment eligibility; no target-vs-reference causal claim is allowed.",
        )
    lower_bound = context.online_reward_lift_lower_bound
    if lower_bound is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            threshold=GateThreshold(
                operator=">=", value=minimum_lift, unit="reward lift"
            ),
            rationale="The causally eligible window is missing its common-window target-vs-reference lift confidence bound.",
        )
    passed = context.online_preference_qualified
    decision = (
        "The complete production outcome window is causally eligible and its 95% target-vs-reference reward-lift lower bound meets the frozen minimum; "
        if passed
        else "The complete production outcome window is causally eligible, but its 95% target-vs-reference reward-lift lower bound misses the frozen minimum; "
    )
    details = (
        f"target SNIPS={context.online_snips_reward!r}, reference SNIPS={context.online_reference_snips_reward!r}, "
        f"point lift={context.online_reward_lift!r}, ESS={context.online_effective_sample_size!r}, "
        f"outcome coverage={context.online_outcome_coverage!r}, segment coverage={context.online_segment_coverage!r}."
    )
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if passed else "fail",
        observed=lower_bound,
        threshold=GateThreshold(operator=">=", value=minimum_lift, unit="reward lift"),
        rationale=decision + details,
    )
