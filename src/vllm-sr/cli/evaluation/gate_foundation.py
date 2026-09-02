"""Reproducibility, contract, hard-policy, and campaign-owned gates G0-G3."""

from __future__ import annotations

from cli.evaluation.gate_context import (
    GateEvidenceContext,
    GateRunMetadata,
    build_gate,
    metric_value,
)
from cli.evaluation.gate_contract import GateDefinition, GateDisposition
from cli.evaluation.reporting import EvaluationGate, EvaluationMetric, GateThreshold


def evaluate_g0(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    *,
    has_records: bool,
    context: GateEvidenceContext,
) -> EvaluationGate:
    checks = (
        has_records,
        context.manifest_validated,
        context.snapshots_complete,
        context.artifact_lineage_complete,
    )
    if any(check is False for check in checks):
        verdict = "fail"
    elif any(check is None for check in checks):
        verdict = "unavailable"
    else:
        verdict = "pass"
    observed_checks = [check for check in checks if check is not None]
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict=verdict,
        observed=(sum(observed_checks) / len(checks) if observed_checks else None),
        threshold=GateThreshold(operator=">=", value=1.0, unit="fraction"),
        rationale=(
            "Manifest, immutable snapshots, normalized records, and lineage are complete."
            if verdict == "pass"
            else "One or more reproducibility requirements are absent or invalid."
        ),
    )


def evaluate_g1(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    *,
    context: GateEvidenceContext,
) -> EvaluationGate:
    checks = (context.manifest_validated, context.snapshots_complete)
    if any(check is False for check in checks):
        verdict = "fail"
    elif any(check is None for check in checks):
        verdict = "unavailable"
    else:
        verdict = "pass"
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict=verdict,
        observed=(1.0 if verdict == "pass" else 0.0 if verdict == "fail" else None),
        threshold=GateThreshold(operator=">=", value=1.0, unit="boolean"),
        rationale=(
            "Strict contracts and snapshot references validated."
            if verdict == "pass"
            else "Schema, reference, or snapshot validation failed."
        ),
    )


def evaluate_g2(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    *,
    metrics: list[EvaluationMetric],
    context: GateEvidenceContext,
) -> EvaluationGate:
    violation_rate = metric_value(metrics, "safety.violation_rate")
    block_accuracy = metric_value(metrics, "safety.block_accuracy")
    if context.hard_policy_static_passed is not True:
        failed = context.hard_policy_static_passed is False
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="fail" if failed else "unavailable",
            rationale=(
                "Static hard-policy enforcement failed."
                if failed
                else "No qualified live static enforcement proof was attached; dynamic safety metrics remain diagnostic."
            ),
        )
    if violation_rate is not None and violation_rate > 0:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            observed=violation_rate,
            threshold=GateThreshold(operator="<=", value=0.0, unit="violations/case"),
            rationale="Qualified live evidence contains a hard-policy violation.",
        )
    if block_accuracy is not None and block_accuracy < 1.0:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            observed=block_accuracy,
            threshold=GateThreshold(operator=">=", value=1.0, unit="fraction"),
            rationale="A deterministic hard-policy blocking decision was incorrect.",
        )
    if violation_rate is None or block_accuracy is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="The live static proof exists, but dynamic violation and blocker coverage is incomplete.",
        )
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass",
        observed=violation_rate,
        threshold=GateThreshold(operator="<=", value=0.0, unit="violations/case"),
        rationale="Live static enforcement and required dynamic hard-policy checks passed.",
    )


def evaluate_g3(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
) -> EvaluationGate:
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="unavailable",
        rationale=(
            "G3 is decided by the server-owned comparative reducer over an immutable "
            "baseline/candidate pair; Campaign consumes the same reduction and a single "
            "worker run cannot self-attest it."
        ),
    )
