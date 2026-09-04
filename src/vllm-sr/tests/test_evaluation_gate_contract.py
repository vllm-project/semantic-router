from __future__ import annotations

import pytest
from cli.evaluation.gate_context import GateEvidenceContext
from cli.evaluation.gate_contract import (
    GATE_CONTRACT_VERSION,
    change_profiles,
    gate_applicability,
)
from cli.evaluation.gates import compute_gates
from cli.evaluation.metric_core import metric_analysis_provenance
from cli.evaluation.reporting import EvaluationGate, EvaluationMetric
from pydantic import ValidationError


def _metric(
    metric_id: str,
    value: float | None,
    *,
    track_id: str,
    direction: str = "higher_is_better",
    unit: str = "fraction",
) -> EvaluationMetric:
    return EvaluationMetric(
        id=metric_id,
        name=metric_id,
        track_id=track_id,
        value=value,
        unit=unit,
        direction=direction,
        sample_count=20,
        analysis_provenance=metric_analysis_provenance(
            metric_id, observed_exclusions=0
        ),
    )


def _qualified_metrics() -> list[EvaluationMetric]:
    return [
        _metric(
            "safety.violation_rate",
            0.0,
            track_id="safety",
            direction="lower_is_better",
        ),
        _metric("safety.block_accuracy", 1.0, track_id="safety"),
        _metric(
            "joint.normalized_regret",
            0.1,
            track_id="joint",
            direction="lower_is_better",
        ),
        _metric(
            "capacity.slo_headroom",
            1.0,
            track_id="capacity",
            unit="concurrency",
        ),
    ]


def test_gate_matrix_always_returns_exactly_g0_through_g9() -> None:
    assert GATE_CONTRACT_VERSION == "evaluation-release-gates.v2"
    assert change_profiles() == (
        "schema_adapter",
        "recipe",
        "selector",
        "model_pool",
        "runtime_capacity",
        "agent_multimodal",
        "online_adaptation",
    )
    for profile in change_profiles():
        matrix = gate_applicability(profile)
        assert [definition.id for definition, _ in matrix] == [
            f"G{index}" for index in range(10)
        ]


def test_every_gate_is_self_describing_and_auditable() -> None:
    gates = compute_gates(
        _qualified_metrics(),
        has_records=True,
        change_profile="recipe",
    )
    assert all(gate.change_profile == "recipe" for gate in gates)
    assert all(gate.contract_version == GATE_CONTRACT_VERSION for gate in gates)
    assert all(gate.evidence_refs for gate in gates)
    assert all(gate.owner for gate in gates)


def test_missing_qualification_evidence_never_becomes_a_pass() -> None:
    metrics = _qualified_metrics()
    metrics[-1] = metrics[-1].model_copy(update={"value": None})
    gates = compute_gates(
        metrics,
        has_records=True,
        change_profile="online_adaptation",
    )
    verdicts = {gate.id: gate.verdict for gate in gates}
    assert verdicts == {
        "G0": "unavailable",
        "G1": "unavailable",
        "G2": "unavailable",
        "G3": "unavailable",
        "G4": "unavailable",
        "G5": "unavailable",
        "G6": "unavailable",
        "G7": "unavailable",
        "G8": "unavailable",
        "G9": "unavailable",
    }

    for gate_id in ("G2", "G3", "G7"):
        gate = next(gate for gate in gates if gate.id == gate_id)
        assert gate.observed is None
        assert gate.threshold is None


def test_observed_hard_policy_violation_remains_diagnostic_without_static_proof() -> (
    None
):
    metrics = _qualified_metrics()
    metrics[0] = metrics[0].model_copy(update={"value": 0.05})
    gates = compute_gates(
        metrics,
        has_records=True,
        change_profile="recipe",
    )
    hard_policy = next(gate for gate in gates if gate.id == "G2")
    assert hard_policy.verdict == "unavailable"
    assert hard_policy.observed is None
    assert hard_policy.threshold is None
    assert metrics[0].value == 0.05


def test_negative_capacity_headroom_is_an_exact_failure_without_a_boolean_proxy() -> (
    None
):
    metrics = _qualified_metrics()
    metrics[-1] = metrics[-1].model_copy(update={"value": -1.0})
    gates = compute_gates(
        metrics,
        has_records=True,
        change_profile="runtime_capacity",
    )
    capacity = next(gate for gate in gates if gate.id == "G7")
    assert capacity.verdict == "fail"
    assert capacity.observed == -1.0
    assert capacity.threshold is not None
    assert capacity.threshold.value == 0
    assert capacity.threshold.unit == "concurrency"
    assert metrics[-1].value == -1.0


def test_full_qualified_online_context_leaves_comparative_g3_to_campaign() -> None:
    context = GateEvidenceContext(
        manifest_validated=True,
        snapshots_complete=True,
        artifact_lineage_complete=True,
        hard_policy_static_passed=True,
        robustness_qualified=True,
        live_fidelity_qualified=True,
        recovery_cluster_qualified=True,
        recovery_cluster_pass_rate_lower_bound=0.9,
        recovery_cluster_minimum_pass_rate_lower_bound=0.8,
        production_candidate_safe=True,
        online_preference_qualified=True,
        production_assignment_support=1.0,
        production_balance_p_value=1.0,
        production_risk_event_rate=0.0,
        production_risk_event_upper_confidence_bound=0.01,
        production_risk_budget_max_rate=0.01,
        online_outcome_coverage=1.0,
        online_effective_sample_size=100.0,
        online_minimum_effective_sample_size=50.0,
        online_effective_sample_ratio=1.0,
        online_minimum_effective_sample_ratio=0.5,
        online_segment_coverage=1.0,
        online_snips_reward=0.75,
        online_reference_snips_reward=0.5,
        online_causal_eligible=True,
        online_reward_lift=0.25,
        online_reward_lift_lower_bound=0.15,
        online_minimum_reward_lift=0.1,
    )
    gates = compute_gates(
        _qualified_metrics(),
        has_records=True,
        change_profile="online_adaptation",
        evidence=context,
    )
    assert all(gate.disposition == "required" for gate in gates)
    assert all(gate.verdict == "pass" for gate in gates if gate.id != "G3")
    comparative = next(gate for gate in gates if gate.id == "G3")
    assert comparative.verdict == "unavailable"
    assert comparative.observed is None
    assert comparative.threshold is None
    capacity = next(gate for gate in gates if gate.id == "G7")
    assert capacity.observed == 1.0
    assert capacity.threshold is not None
    assert capacity.threshold.unit == "concurrency"
    assert capacity.threshold.value == 0


def test_qualified_capacity_gate_is_decided_by_exact_slo_headroom() -> None:
    metrics = _qualified_metrics()
    metrics[-1] = metrics[-1].model_copy(update={"value": -1.0})
    gates = compute_gates(
        metrics,
        has_records=True,
        change_profile="runtime_capacity",
    )
    capacity = next(gate for gate in gates if gate.id == "G7")
    assert capacity.verdict == "fail"
    assert capacity.observed == -1.0
    assert capacity.threshold is not None
    assert capacity.threshold.unit == "concurrency"
    assert capacity.threshold.value == 0


def test_not_applicable_is_explicit_not_omitted() -> None:
    evidence = GateEvidenceContext(
        recovery_cluster_qualified=True,
        recovery_cluster_pass_rate_lower_bound=0.9,
        recovery_cluster_minimum_pass_rate_lower_bound=0.8,
    )
    gates = compute_gates(
        [],
        has_records=True,
        change_profile="schema_adapter",
        evidence=evidence,
    )
    by_id = {gate.id: gate for gate in gates}
    assert by_id["G6"].disposition == "not_applicable"
    assert by_id["G6"].verdict == "not_applicable"
    assert by_id["G6"].observed is None
    assert by_id["G6"].threshold is None
    assert by_id["G8"].verdict == "not_applicable"
    assert by_id["G9"].verdict == "not_applicable"

    payload = by_id["G6"].model_dump(mode="python")
    with pytest.raises(ValidationError, match="disposition and verdict must match"):
        EvaluationGate.model_validate({**payload, "verdict": "pass"})
    with pytest.raises(ValidationError, match="observation or threshold"):
        EvaluationGate.model_validate({**payload, "observed": 0.9})
