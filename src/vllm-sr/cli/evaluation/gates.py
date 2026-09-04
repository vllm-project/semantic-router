"""Evidence-aware G0-G9 release gates; missing evidence never passes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.gate_contract import (
    DEFAULT_CHANGE_PROFILE,
    GATE_CONTRACT_VERSION,
    ChangeProfile,
    GateDefinition,
    GateDisposition,
    gate_applicability,
)
from cli.evaluation.reporting import (
    EvaluationCoverage,
    EvaluationGate,
    EvaluationMetric,
    GateThreshold,
)

_DEFAULT_NORMALIZED_REGRET_MAX = 0.25
_DEFAULT_CAPACITY_SUCCESS_MIN = 0.95


@dataclass(frozen=True)
class GateEvidenceContext:
    """Evidence that cannot be inferred safely from an aggregate metric.

    ``None`` means that the evidence was not produced. A false value is a
    qualified failure. This distinction is essential for release decisions.
    """

    manifest_validated: bool | None = None
    snapshots_complete: bool | None = None
    artifact_lineage_complete: bool | None = None
    hard_policy_static_passed: bool | None = None
    baseline_qualified: bool | None = None
    robustness_qualified: bool | None = None
    live_fidelity_qualified: bool | None = None
    trajectory_qualified: bool | None = None
    capacity_slo_qualified: bool | None = None
    shadow_canary_qualified: bool | None = None
    online_preference_qualified: bool | None = None


@dataclass(frozen=True)
class _GateRunMetadata:
    change_profile: ChangeProfile
    evidence_refs: tuple[str, ...]
    sample_count: int | None
    coverage: EvaluationCoverage | None
    owner: str
    evaluated_at: datetime | None


_TRACK_BY_GATE = {
    "G2": "safety",
    "G3": "joint",
    "G4": "routing",
    "G5": "joint",
    "G6": "agentic",
    "G7": "capacity",
    "G9": "preference",
}

_EVIDENCE_LEVEL_BY_GATE = {
    "G0": "E0",
    "G1": "E0",
    "G2": "E3",
    "G3": "E4",
    "G4": "E4",
    "G5": "E5",
    "G6": "E5",
    "G7": "E5",
    "G8": "E5",
    "G9": "E5",
}

_EVIDENCE_REFS_BY_GATE = {
    "G0": (
        "run-manifest.json",
        "lineage.json",
        "provenance.json",
        "checksums.sha256",
    ),
    "G1": ("run-manifest.json", "records.jsonl"),
    "G2": ("records.jsonl", "metric:safety.violation_rate"),
    "G3": ("metrics.json", "metric:joint.normalized_regret"),
    "G4": ("records.jsonl", "metric:routing.accuracy"),
    "G5": ("records.jsonl", "provenance.json"),
    "G6": ("records.jsonl", "metric:agentic.success_rate"),
    "G7": ("records.jsonl", "metrics.json"),
    "G8": ("run-manifest.json", "records.jsonl"),
    "G9": ("records.jsonl", "metric:preference.propensity_coverage"),
}

_OWNER_BY_GATE = {
    "G0": "evaluation-platform",
    "G1": "evaluation-platform",
    "G2": "router-policy",
    "G3": "recipe-and-model-pool",
    "G4": "evaluation-workload",
    "G5": "router-and-serving-runtime",
    "G6": "agent-runtime",
    "G7": "serving-capacity",
    "G8": "release-operations",
    "G9": "online-learning",
}


def _lookup(metrics: list[EvaluationMetric], metric_id: str) -> EvaluationMetric | None:
    return next((metric for metric in metrics if metric.id == metric_id), None)


def _metric_value(metrics: list[EvaluationMetric], metric_id: str) -> float | None:
    metric = _lookup(metrics, metric_id)
    return metric.value if metric is not None else None


def _gate(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    *,
    verdict: str,
    rationale: str,
    observed: float | None = None,
    threshold: GateThreshold | None = None,
) -> EvaluationGate:
    if disposition == "not_applicable":
        verdict = "not_applicable"
    return EvaluationGate(
        id=definition.id,
        name=definition.name,
        description=definition.description,
        track_id=_TRACK_BY_GATE.get(definition.id),
        disposition=disposition,
        verdict=verdict,
        change_profile=metadata.change_profile,
        contract_version=GATE_CONTRACT_VERSION,
        evidence_refs=metadata.evidence_refs,
        evidence_level=_EVIDENCE_LEVEL_BY_GATE[definition.id],
        observed=observed,
        threshold=threshold,
        sample_count=metadata.sample_count,
        coverage=metadata.coverage,
        owner=metadata.owner,
        evaluated_at=metadata.evaluated_at,
        rationale=(
            f"{rationale} Gate contract: {GATE_CONTRACT_VERSION}."
            if disposition != "not_applicable"
            else f"Not applicable for this change profile. {rationale}"
        ),
    )


def _qualified_boolean_gate(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    qualified: bool | None,
    *,
    success: str,
    missing: str,
    failure: str,
) -> EvaluationGate:
    if disposition == "not_applicable":
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="not_applicable",
            rationale=missing,
        )
    if qualified is None:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale=missing,
        )
    return _gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if qualified else "fail",
        rationale=success if qualified else failure,
        observed=1.0 if qualified else 0.0,
        threshold=GateThreshold(operator=">=", value=1.0, unit="boolean"),
    )


def _g0(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
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
    return _gate(
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


def _g1(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
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
    return _gate(
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


def _g2(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    *,
    metrics: list[EvaluationMetric],
    context: GateEvidenceContext,
) -> EvaluationGate:
    violation_rate = _metric_value(metrics, "safety.violation_rate")
    block_accuracy = _metric_value(metrics, "safety.block_accuracy")
    if violation_rate is not None and violation_rate > 0:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            observed=violation_rate,
            threshold=GateThreshold(operator="<=", value=0.0, unit="violations/case"),
            rationale="Qualified evidence contains an observed hard-policy violation.",
        )
    if block_accuracy is not None and block_accuracy < 1.0:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            observed=block_accuracy,
            threshold=GateThreshold(operator=">=", value=1.0, unit="fraction"),
            rationale="A deterministic hard-policy blocking decision was incorrect.",
        )
    if context.hard_policy_static_passed is not True:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict=(
                "fail" if context.hard_policy_static_passed is False else "unavailable"
            ),
            rationale=(
                "Static hard-policy enforcement failed."
                if context.hard_policy_static_passed is False
                else "No qualified static enforcement proof was attached; finite 0/N observations cannot prove a zero violation rate."
            ),
        )
    if violation_rate is None or block_accuracy is None:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="Static proof exists, but required dynamic violation and blocker coverage is incomplete.",
        )
    return _gate(
        definition,
        disposition,
        metadata,
        verdict="pass",
        observed=violation_rate,
        threshold=GateThreshold(operator="<=", value=0.0, unit="violations/case"),
        rationale="Static enforcement and required dynamic hard-policy checks passed.",
    )


def _g3(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    *,
    metrics: list[EvaluationMetric],
    context: GateEvidenceContext,
) -> EvaluationGate:
    regret = _metric_value(metrics, "joint.normalized_regret")
    if context.baseline_qualified is False:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            rationale="The candidate failed its declared baseline/frontier comparison.",
            observed=regret,
        )
    if context.baseline_qualified is not True or regret is None:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="A paired incumbent/no-information frontier comparison is missing or pool-normalized regret is unavailable.",
            observed=regret,
        )
    passed = regret <= _DEFAULT_NORMALIZED_REGRET_MAX
    return _gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if passed else "fail",
        observed=regret,
        threshold=GateThreshold(
            operator="<=", value=_DEFAULT_NORMALIZED_REGRET_MAX, unit="fraction"
        ),
        rationale=(
            "The candidate qualified against its declared baselines and met the default normalized-regret bound."
            if passed
            else "The candidate qualified against its baseline but exceeded the normalized-regret bound."
        ),
    )


def _g7(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    *,
    metrics: list[EvaluationMetric],
    context: GateEvidenceContext,
    cost_accounted: bool,
) -> EvaluationGate:
    success_rate = _metric_value(metrics, "capacity.success_rate")
    if success_rate is not None and success_rate < _DEFAULT_CAPACITY_SUCCESS_MIN:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="fail",
            observed=success_rate,
            threshold=GateThreshold(
                operator=">=", value=_DEFAULT_CAPACITY_SUCCESS_MIN, unit="fraction"
            ),
            rationale="The bounded load sweep exceeded the default error budget.",
        )
    if not cost_accounted:
        return _gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale="Runtime, evaluation-overhead, and capacity-TCO ledgers are incomplete; missing cost was not inferred as zero.",
        )
    return _qualified_boolean_gate(
        definition,
        disposition,
        metadata,
        context.capacity_slo_qualified,
        success="The declared load profile, SLO crossing, saturation, and headroom contract passed with all three cost ledgers.",
        missing="Cost ledgers exist, but no qualified latency/capacity SLO and headroom contract was attached.",
        failure="The declared latency/capacity SLO or headroom contract failed.",
    )


def _gate_metadata(
    definition: GateDefinition,
    change_profile: ChangeProfile,
    records: list[ExecutionRecord] | None,
    evaluated_at: datetime | None,
) -> _GateRunMetadata:
    rows: list[ExecutionRecord] | None = None
    if records is not None:
        track_id = _TRACK_BY_GATE.get(definition.id)
        rows = (
            records
            if track_id is None and definition.id in {"G0", "G1", "G8"}
            else [record for record in records if record.track_id == track_id]
        )
    sample_count = None
    gate_coverage = None
    if rows is not None:
        sample_count = sum(record.status != "unavailable" for record in rows)
        total = len(rows)
        unavailable = total - sample_count
        gate_coverage = EvaluationCoverage(
            evaluated=sample_count,
            total=total,
            fraction=sample_count / total if total else 0,
            unavailable=unavailable,
        )
    return _GateRunMetadata(
        change_profile=change_profile,
        evidence_refs=_EVIDENCE_REFS_BY_GATE[definition.id],
        sample_count=sample_count,
        coverage=gate_coverage,
        owner=_OWNER_BY_GATE[definition.id],
        evaluated_at=evaluated_at,
    )


def _evidence_qualified_gate(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: _GateRunMetadata,
    context: GateEvidenceContext,
) -> EvaluationGate:
    if definition.id == "G4":
        return _qualified_boolean_gate(
            definition,
            disposition,
            metadata,
            context.robustness_qualified,
            success="Qualified invariant, expected-change, OOD, and contamination checks passed.",
            missing="No qualified robustness/OOD and contamination evidence was attached.",
            failure="A qualified robustness/OOD or contamination check failed.",
        )
    if definition.id == "G5":
        return _qualified_boolean_gate(
            definition,
            disposition,
            metadata,
            context.live_fidelity_qualified,
            success="Paired replay/live fidelity and complete live failure accounting passed.",
            missing="No paired replay-to-live gap and fresh-output evidence was attached.",
            failure="Replay-to-live fidelity or live failure accounting failed.",
        )
    if definition.id == "G6":
        return _qualified_boolean_gate(
            definition,
            disposition,
            metadata,
            context.trajectory_qualified,
            success="Qualified terminal, continuity, recovery, state-isolation, and tool-side-effect evidence passed.",
            missing="Single-turn success is insufficient; qualified trajectory, continuity, recovery, and side-effect evidence is missing.",
            failure="A qualified trajectory, continuity, recovery, state-isolation, or tool-side-effect check failed.",
        )
    if definition.id == "G8":
        return _qualified_boolean_gate(
            definition,
            disposition,
            metadata,
            context.shadow_canary_qualified,
            success="Qualified shadow/canary assignment, guardrail, stop, and rollback evidence passed.",
            missing="No production assignment/exposure, risk-budget, stop, and rollback evidence was attached.",
            failure="A qualified shadow/canary guardrail, risk-budget, or rollback condition failed.",
        )
    return _qualified_boolean_gate(
        definition,
        disposition,
        metadata,
        context.online_preference_qualified,
        success="Qualified online preference evidence includes exposure, propensity, effective sample size, confidence, and segments.",
        missing="Offline preference agreement is not online causal evidence; assignment propensity and exposure evidence is missing.",
        failure="The qualified online preference contract failed.",
    )


def compute_gates(
    metrics: list[EvaluationMetric],
    *,
    has_records: bool,
    cost_accounted: bool,
    change_profile: ChangeProfile = DEFAULT_CHANGE_PROFILE,
    evidence: GateEvidenceContext | None = None,
    records: list[ExecutionRecord] | None = None,
    evaluated_at: datetime | None = None,
) -> list[EvaluationGate]:
    """Evaluate every gate under one explicit change-profile matrix."""

    context = evidence or GateEvidenceContext()
    gates: list[EvaluationGate] = []
    for definition, disposition in gate_applicability(change_profile):
        metadata = _gate_metadata(definition, change_profile, records, evaluated_at)
        if definition.id == "G0":
            gate = _g0(
                definition,
                disposition,
                metadata,
                has_records=has_records,
                context=context,
            )
        elif definition.id == "G1":
            gate = _g1(definition, disposition, metadata, context=context)
        elif definition.id == "G2":
            gate = _g2(
                definition,
                disposition,
                metadata,
                metrics=metrics,
                context=context,
            )
        elif definition.id == "G3":
            gate = _g3(
                definition,
                disposition,
                metadata,
                metrics=metrics,
                context=context,
            )
        elif definition.id == "G7":
            gate = _g7(
                definition,
                disposition,
                metadata,
                metrics=metrics,
                context=context,
                cost_accounted=cost_accounted,
            )
        else:
            gate = _evidence_qualified_gate(definition, disposition, metadata, context)
        gates.append(gate)
    return gates
