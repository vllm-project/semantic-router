"""Qualified robustness, fidelity, recovery, and capacity gates G4-G7."""

from __future__ import annotations

from cli.evaluation.gate_context import (
    GateEvidenceContext,
    GateRunMetadata,
    build_gate,
    metric_value,
    qualified_boolean_gate,
)
from cli.evaluation.gate_contract import GateDefinition, GateDisposition
from cli.evaluation.reporting import EvaluationGate, EvaluationMetric, GateThreshold


def evaluate_g4(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    missing = (
        "The source-qualified perturbation method produced no complete typed pair and declared-slice reduction."
        if "G4" in context.method_qualified_gate_ids
        else "No source-qualified declared-shift robustness evidence was attached."
    )
    return qualified_boolean_gate(
        definition,
        disposition,
        metadata,
        context.robustness_qualified,
        success="Every source-qualified perturbation relation and declared slice passed.",
        missing=missing,
        failure="A source-qualified perturbation relation or declared slice failed.",
    )


def evaluate_g5(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    return qualified_boolean_gate(
        definition,
        disposition,
        metadata,
        context.live_fidelity_qualified,
        success="Qualified reference-to-fresh-live fidelity and complete live failure accounting passed.",
        missing="No qualified reference-to-fresh-live pair for the unchanged candidate was attached.",
        failure="Reference-to-fresh-live fidelity or live failure accounting failed.",
    )


def evaluate_g6(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    missing = (
        "No server-brokered live exact-step fault ledger with paired baseline/treatment "
        "receipts, state, retry, latency, and side-effect evidence was attached; "
        "Continuity labeled-failover exports are diagnostic only."
    )
    observed = context.recovery_cluster_pass_rate_lower_bound
    threshold = context.recovery_cluster_minimum_pass_rate_lower_bound
    if (
        context.recovery_cluster_qualified is None
        or observed is None
        or threshold is None
    ):
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale=missing,
        )
    passed = context.recovery_cluster_qualified
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if passed else "fail",
        observed=observed,
        threshold=GateThreshold(operator=">=", value=threshold, unit="fraction"),
        rationale=(
            "The complete live fault window met exact pair requirements and the one-sided 95% reliability bound across its independent recovery clusters."
            if passed
            else "The complete live fault window failed an exact pair requirement or its independent-cluster reliability lower bound."
        ),
    )


def evaluate_g7(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    metrics: list[EvaluationMetric],
) -> EvaluationGate:
    slo_headroom = metric_value(metrics, "capacity.slo_headroom")
    if slo_headroom is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="A frozen live capacity SLO and measured concurrency headroom are missing.",
        )
    passed = slo_headroom >= 0
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if passed else "fail",
        observed=slo_headroom,
        threshold=GateThreshold(operator=">=", value=0, unit="concurrency"),
        rationale=(
            "The measured operating envelope met every frozen latency, error, throughput, and scaling requirement at the required concurrency."
            if passed
            else "The measured operating envelope fell short of the frozen capacity SLO."
        ),
    )
