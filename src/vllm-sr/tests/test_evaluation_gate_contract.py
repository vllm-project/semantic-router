from __future__ import annotations

from cli.evaluation.gate_contract import (
    GATE_CONTRACT_VERSION,
    change_profiles,
    gate_applicability,
)
from cli.evaluation.gates import GateEvidenceContext, compute_gates
from cli.evaluation.reporting import EvaluationMetric


def _metric(
    metric_id: str,
    value: float | None,
    *,
    track_id: str,
    direction: str = "higher_is_better",
) -> EvaluationMetric:
    return EvaluationMetric(
        id=metric_id,
        name=metric_id,
        track_id=track_id,
        value=value,
        unit="fraction",
        direction=direction,
        sample_count=20,
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
        _metric("capacity.success_rate", 1.0, track_id="capacity"),
    ]


def test_gate_matrix_always_returns_exactly_g0_through_g9() -> None:
    assert GATE_CONTRACT_VERSION == "evaluation-release-gates.v1"
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
        cost_accounted=True,
        change_profile="recipe",
    )
    assert all(gate.change_profile == "recipe" for gate in gates)
    assert all(gate.contract_version == GATE_CONTRACT_VERSION for gate in gates)
    assert all(gate.evidence_refs for gate in gates)
    assert all(gate.owner for gate in gates)


def test_missing_qualification_evidence_never_becomes_a_pass() -> None:
    gates = compute_gates(
        _qualified_metrics(),
        has_records=True,
        cost_accounted=True,
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


def test_observed_hard_policy_violation_fails_even_without_static_proof() -> None:
    metrics = _qualified_metrics()
    metrics[0] = metrics[0].model_copy(update={"value": 0.05})
    gates = compute_gates(
        metrics,
        has_records=True,
        cost_accounted=True,
        change_profile="recipe",
    )
    hard_policy = next(gate for gate in gates if gate.id == "G2")
    assert hard_policy.verdict == "fail"
    assert hard_policy.observed == 0.05


def test_full_qualified_online_context_can_pass_all_required_gates() -> None:
    context = GateEvidenceContext(
        manifest_validated=True,
        snapshots_complete=True,
        artifact_lineage_complete=True,
        hard_policy_static_passed=True,
        baseline_qualified=True,
        robustness_qualified=True,
        live_fidelity_qualified=True,
        trajectory_qualified=True,
        capacity_slo_qualified=True,
        shadow_canary_qualified=True,
        online_preference_qualified=True,
    )
    gates = compute_gates(
        _qualified_metrics(),
        has_records=True,
        cost_accounted=True,
        change_profile="online_adaptation",
        evidence=context,
    )
    assert all(gate.disposition == "required" for gate in gates)
    assert all(gate.verdict == "pass" for gate in gates)


def test_not_applicable_is_explicit_not_omitted() -> None:
    gates = compute_gates(
        [],
        has_records=True,
        cost_accounted=False,
        change_profile="schema_adapter",
    )
    by_id = {gate.id: gate for gate in gates}
    assert by_id["G6"].disposition == "not_applicable"
    assert by_id["G6"].verdict == "not_applicable"
    assert by_id["G8"].verdict == "not_applicable"
    assert by_id["G9"].verdict == "not_applicable"
