"""Shared immutable context and presentation metadata for release gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.gate_contract import (
    GATE_CONTRACT_VERSION,
    ChangeProfile,
    GateDefinition,
    GateDisposition,
)
from cli.evaluation.reporting import (
    EvaluationCoverage,
    EvaluationGate,
    EvaluationMetric,
    GateThreshold,
)


@dataclass(frozen=True)
class GateEvidenceContext:
    """Server-reduced evidence that aggregate metrics cannot safely imply.

    ``None`` means that the required evidence was not produced. ``False`` means
    a complete, qualified method failed. Keeping those states distinct prevents
    missing production evidence from becoming a promotional failure or pass.
    """

    manifest_validated: bool | None = None
    snapshots_complete: bool | None = None
    artifact_lineage_complete: bool | None = None
    hard_policy_static_passed: bool | None = None
    robustness_qualified: bool | None = None
    live_fidelity_qualified: bool | None = None
    recovery_cluster_qualified: bool | None = None
    recovery_cluster_pass_rate_lower_bound: float | None = None
    recovery_cluster_minimum_pass_rate_lower_bound: float | None = None
    production_candidate_safe: bool | None = None
    online_preference_qualified: bool | None = None
    production_assignment_support: float | None = None
    production_balance_p_value: float | None = None
    production_risk_event_rate: float | None = None
    production_risk_event_upper_confidence_bound: float | None = None
    production_risk_budget_max_rate: float | None = None
    online_outcome_coverage: float | None = None
    online_effective_sample_size: float | None = None
    online_minimum_effective_sample_size: float | None = None
    online_effective_sample_ratio: float | None = None
    online_minimum_effective_sample_ratio: float | None = None
    online_segment_coverage: float | None = None
    online_snips_reward: float | None = None
    online_reference_snips_reward: float | None = None
    online_causal_eligible: bool | None = None
    online_reward_lift: float | None = None
    online_reward_lift_lower_bound: float | None = None
    online_minimum_reward_lift: float | None = None
    method_qualified_gate_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GateRunMetadata:
    change_profile: ChangeProfile
    evidence_refs: tuple[str, ...]
    sample_count: int | None
    coverage: EvaluationCoverage | None
    owner: str
    evaluated_at: datetime | None


TRACK_BY_GATE = MappingProxyType(
    {
        "G2": "safety",
        "G3": "joint",
        "G4": "routing",
        "G5": "joint",
        "G6": "agentic",
        "G7": "capacity",
        "G8": "preference",
        "G9": "preference",
    }
)

EVIDENCE_LEVEL_BY_GATE = MappingProxyType(
    {
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
)

EVIDENCE_REFS_BY_GATE = MappingProxyType(
    {
        "G0": (
            "run-manifest.json",
            "lineage.json",
            "provenance.json",
            "checksums.sha256",
        ),
        "G1": ("run-manifest.json", "records.jsonl"),
        "G2": (
            "records.jsonl",
            "metric:safety.violation_rate",
            "metric:safety.block_accuracy",
            "method:evaluation-hard-policy-proof.v1",
        ),
        "G3": ("metrics.json", "metric:joint.normalized_regret"),
        "G4": (
            "records.jsonl",
            "metric:routing.robustness_pass_rate",
            "metric:routing.robustness_worst_slice_pass_rate",
        ),
        "G5": ("records.jsonl", "provenance.json"),
        "G6": (
            "records.jsonl",
            "metric:agentic.recovery_cluster_pass_rate_lower_95",
        ),
        "G7": (
            "run-manifest.json",
            "capacity-profile.json",
            "metric:capacity.error_rate_upper_bound",
            "metric:capacity.error_rate_cluster_range_max",
            "metric:capacity.measurement_cluster_count_min",
            "metric:capacity.slo_headroom",
        ),
        "G8": (
            "run-manifest.json",
            "records.jsonl",
            "metric:experiment.assignment_balance_p_value",
            "metric:experiment.risk_event_upper_confidence_bound",
            "metric:experiment.risk_budget_max_rate",
            "metric:experiment.candidate_safe",
        ),
        "G9": (
            "records.jsonl",
            "metric:preference.online_reward_lift",
            "metric:preference.online_effective_sample_size",
            "metric:preference.online_segment_coverage",
        ),
    }
)

OWNER_BY_GATE = MappingProxyType(
    {
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
)


def metric_value(metrics: list[EvaluationMetric], metric_id: str) -> float | None:
    metric = next((item for item in metrics if item.id == metric_id), None)
    return metric.value if metric is not None else None


def build_gate(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    *,
    verdict: str,
    rationale: str,
    observed: float | None = None,
    threshold: GateThreshold | None = None,
) -> EvaluationGate:
    if disposition == "not_applicable":
        verdict = "not_applicable"
        observed = None
        threshold = None
    return EvaluationGate(
        id=definition.id,
        name=definition.name,
        description=definition.description,
        track_id=TRACK_BY_GATE.get(definition.id),
        disposition=disposition,
        verdict=verdict,
        change_profile=metadata.change_profile,
        contract_version=GATE_CONTRACT_VERSION,
        evidence_refs=metadata.evidence_refs,
        evidence_level=EVIDENCE_LEVEL_BY_GATE[definition.id],
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


def qualified_boolean_gate(
    definition: GateDefinition,
    disposition: GateDisposition,
    metadata: GateRunMetadata,
    qualified: bool | None,
    *,
    success: str,
    missing: str,
    failure: str,
) -> EvaluationGate:
    if qualified is None:
        return build_gate(
            definition,
            disposition,
            metadata,
            verdict="unavailable",
            rationale=missing,
        )
    return build_gate(
        definition,
        disposition,
        metadata,
        verdict="pass" if qualified else "fail",
        rationale=success if qualified else failure,
        observed=1.0 if qualified else 0.0,
        threshold=GateThreshold(operator=">=", value=1.0, unit="boolean"),
    )


def gate_metadata(
    definition: GateDefinition,
    change_profile: ChangeProfile,
    records: list[ExecutionRecord] | None,
    evaluated_at: datetime | None,
) -> GateRunMetadata:
    sample_count = None
    coverage = None
    if records is not None:
        track_id = TRACK_BY_GATE.get(definition.id)
        if track_id is None:
            planned = {(record.track_id, record.case_id) for record in records}
            evaluated = {
                (record.track_id, record.case_id)
                for record in records
                if record.status != "unavailable"
            }
        else:
            planned = {
                record.case_id for record in records if record.track_id == track_id
            }
            evaluated = {
                record.case_id
                for record in records
                if record.track_id == track_id and record.status != "unavailable"
            }
        sample_count = len(evaluated)
        total = len(planned)
        coverage = EvaluationCoverage(
            evaluated=sample_count,
            total=total,
            fraction=sample_count / total if total else 0,
            unavailable=total - sample_count,
        )
    return GateRunMetadata(
        change_profile=change_profile,
        evidence_refs=EVIDENCE_REFS_BY_GATE[definition.id],
        sample_count=sample_count,
        coverage=coverage,
        owner=OWNER_BY_GATE[definition.id],
        evaluated_at=evaluated_at,
    )
