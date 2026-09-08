from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from cli.evaluation.builtin_executors import LiveRuntimeExecutor
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import track_evidence_level
from cli.evaluation.evidence_source_ids import (
    LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
    LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
    LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.fault_recovery_ledger import (
    execute_fault_recovery_ledger,
)
from cli.evaluation.gate_context import GateEvidenceContext
from cli.evaluation.gates import compute_gates
from cli.evaluation.hard_policy_ledger import (
    execute_hard_policy_ledger,
)
from cli.evaluation.method_evidence import (
    ProductionExperimentMethodEvidence,
    RobustnessMethodEvidence,
)
from cli.evaluation.metric_hard_policy import reduce_hard_policy
from cli.evaluation.metric_production_experiment import (
    production_experiment_metrics,
    reduce_production_experiment,
)
from cli.evaluation.metric_recovery import reduce_recovery
from cli.evaluation.metric_robustness import reduce_robustness
from cli.evaluation.reporting import EvidenceLevel
from evaluation_method_ledger_test_support import (
    _CONFIG,
    _POLICY,
    _TOPOLOGY,
    _digest,
    _execute_method_kind,
    _execute_production,
    _fault_recovery_ledger,
    _hard_policy_ledger,
    _LedgerClient,
    _mixture,
    _production_assignment,
    _production_ledger,
    _recovery_pair,
)


def test_production_ledger_yields_g8_controls_and_g9_causal_estimates() -> None:
    execution = _execute_production(_production_ledger())
    reduced = reduce_production_experiment(execution.records)

    assert reduced.candidate_safe is True
    assert reduced.causal_eligible is True
    assert reduced.assignment_support == 1.0
    assert reduced.outcome_coverage == 1.0
    assert reduced.effective_sample_size == pytest.approx(10.0)
    assert reduced.snips_reward == pytest.approx(1.0)
    assert reduced.reference_snips_reward == pytest.approx(0.5)
    assert reduced.reward_lift == pytest.approx(0.5)
    assert reduced.preference_passed is True
    assert reduced.snips_confidence_interval is not None
    metric_values = {
        metric.id: metric.value
        for metric in production_experiment_metrics(execution.records)
    }
    assert metric_values["experiment.risk_budget_max_rate"] == 0.2
    assert metric_values["preference.online_snips_reward"] == pytest.approx(1.0)

    context = GateEvidenceContext(
        production_candidate_safe=reduced.candidate_safe,
        online_preference_qualified=reduced.preference_passed,
        production_assignment_support=reduced.assignment_support,
        production_balance_p_value=reduced.assignment_balance_p_value,
        production_risk_event_rate=reduced.risk_event_rate,
        production_risk_event_upper_confidence_bound=(
            reduced.risk_event_upper_confidence_bound
        ),
        production_risk_budget_max_rate=reduced.risk_budget_max_rate,
        online_outcome_coverage=reduced.outcome_coverage,
        online_effective_sample_size=reduced.effective_sample_size,
        online_minimum_effective_sample_size=reduced.minimum_effective_sample_size,
        online_effective_sample_ratio=reduced.effective_sample_ratio,
        online_minimum_effective_sample_ratio=reduced.minimum_effective_sample_ratio,
        online_segment_coverage=reduced.segment_coverage,
        online_snips_reward=reduced.snips_reward,
        online_reference_snips_reward=reduced.reference_snips_reward,
        online_causal_eligible=reduced.causal_eligible,
        online_reward_lift=reduced.reward_lift,
        online_reward_lift_lower_bound=reduced.reward_lift_confidence_interval[0],
        online_minimum_reward_lift=reduced.minimum_reward_lift,
    )
    gates = compute_gates(
        production_experiment_metrics(execution.records),
        has_records=True,
        change_profile="online_adaptation",
        evidence=context,
        records=execution.records,
    )
    g8 = next(gate for gate in gates if gate.id == "G8")
    g9 = next(gate for gate in gates if gate.id == "G9")
    assert g8.verdict == "pass"
    assert g8.observed == pytest.approx(reduced.risk_event_upper_confidence_bound)
    assert g8.threshold.value == 0.2
    assert (g9.verdict, g9.observed, g9.threshold.value) == ("pass", 0.5, 0.1)
    assert "target SNIPS=1.0" in g9.rationale


def test_production_gate_never_qualifies_a_partial_sealed_window() -> None:
    ledger = _production_ledger()
    with pytest.raises(ValueError, match="cover every assignment"):
        _execute_production(ledger, sample_limit=1)

    full = _execute_production(ledger)
    reduced = reduce_production_experiment(full.records[:10])
    assert reduced.candidate_safe is False
    assert reduced.causal_eligible is False
    assert reduced.snips_reward is None


def test_preference_without_full_outcomes_has_no_causal_claim() -> None:
    execution = _execute_production(_production_ledger(outcome_count=19))
    reduced = reduce_production_experiment(execution.records)
    assert reduced.candidate_safe is True
    assert reduced.outcome_coverage == 0.95
    assert reduced.causal_eligible is False
    assert reduced.ips_reward is None
    assert reduced.snips_reward is None
    assert reduced.preference_passed is None


def test_production_risk_budget_failure_is_reported_against_frozen_threshold() -> None:
    execution = _execute_production(_production_ledger(risk_event_count=5))
    reduced = reduce_production_experiment(execution.records)
    assert reduced.risk_event_rate == 0.25
    assert reduced.risk_event_upper_confidence_bound > 0.2
    assert reduced.risk_budget_max_rate == 0.2
    assert reduced.candidate_safe is False


def test_tiny_clean_production_window_cannot_pass_g8() -> None:
    execution = _execute_production(_production_ledger(assignment_count=2))
    reduced = reduce_production_experiment(execution.records)
    assert reduced.risk_event_rate == 0
    assert reduced.risk_event_upper_confidence_bound > 0.2
    assert reduced.candidate_safe is False


def test_causally_eligible_but_regressed_target_policy_fails_g9() -> None:
    execution = _execute_production(
        _production_ledger(target_reward=0.25, reference_reward=0.75)
    )
    reduced = reduce_production_experiment(execution.records)
    assert reduced.causal_eligible is True
    assert reduced.reward_lift == pytest.approx(-0.5)
    assert reduced.reward_lift_confidence_interval is not None
    assert reduced.preference_passed is False


def test_successful_rollback_does_not_turn_a_triggered_stop_into_a_g8_pass() -> None:
    reduced = reduce_production_experiment(
        _execute_production(_production_ledger(stop_triggered=True)).records
    )
    assert reduced.controls_operational is True
    assert reduced.candidate_safe is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_assignment_count", 2),
        ("minimum_effective_sample_size", 1),
        ("minimum_effective_sample_ratio", 0.1),
        ("minimum_segment_sample_size", 1),
        ("minimum_reward_lift", -0.1),
        ("risk_budget_max_rate", 0.5),
    ),
)
def test_production_ledger_cannot_relax_platform_minima(
    field: str, value: object
) -> None:
    payload = _production_assignment(
        1, assignment_count=20, total_outcomes=20
    ).model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError):
        ProductionExperimentMethodEvidence.model_validate(payload)


def test_hard_policy_requires_exact_binding_pairs_and_full_window() -> None:
    ledger = _hard_policy_ledger()
    client = _LedgerClient(ledger.model_dump(mode="json"))
    execution = execute_hard_policy_ledger(
        client,  # type: ignore[arg-type]
        "https://policy.example.test/window",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=_mixture(),
        sample_limit=2,
        seed=11,
    )
    assert client.calls[0]["broker_operation"] == "hard-policy.ledger"
    assert (
        reduce_hard_policy(
            execution.records,
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
        ).dynamic_passed
        is True
    )
    partial = reduce_hard_policy(
        execution.records[:1],
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
    )
    assert partial.static_proof_passed is False
    assert partial.dynamic_passed is False
    with pytest.raises(ValueError, match="cover every observation"):
        execute_hard_policy_ledger(
            client,  # type: ignore[arg-type]
            "https://policy.example.test/window",
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
            target_id=_mixture().id,
            backend_topology_digest=_TOPOLOGY,
            mixture=_mixture(),
            sample_limit=1,
            seed=11,
        )


def test_g4_and_g6_partial_native_exports_remain_unavailable() -> None:
    source = ExecutionRecord(
        id="routing-source",
        track_id="routing",
        case_id="source",
        attempt_id="source-attempt",
        status="succeeded",
        selected_arm_id="arm-a",
        success=True,
    )
    target = ExecutionRecord(
        id="routing-target",
        track_id="routing",
        case_id="target",
        attempt_id="target-attempt",
        status="succeeded",
        selected_arm_id="arm-a",
        success=True,
        robustness=RobustnessMethodEvidence(
            method_id="routerarena.robustness.v1",
            pair_id="pair-1",
            source_case_id="source",
            target_case_id="target",
            shift_type="paraphrase",
            relation="invariant",
            source_action_id="arm-a",
            slice_ids=("routerarena:paraphrase",),
            native_pair_count=2,
            source_record_digest=_digest("b"),
        ),
    )
    assert reduce_robustness([source, target]).passed is None

    recovery = _recovery_pair(1)
    row = ExecutionRecord(
        id="agentic-row",
        track_id="agentic",
        case_id="agentic-case",
        attempt_id="agentic-attempt",
        status="succeeded",
        success=True,
        recovery=recovery,
    )
    assert reduce_recovery([row]).passed is None


def test_live_fault_recovery_ledger_is_full_window_and_snapshot_bound() -> None:
    ledger = _fault_recovery_ledger()
    client = _LedgerClient(ledger.model_dump(mode="json"))
    execution = execute_fault_recovery_ledger(
        client,  # type: ignore[arg-type]
        "https://faults.example.test/window",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=_mixture(),
        sample_limit=20,
        seed=13,
    )
    assert client.calls[0]["broker_operation"] == "fault-recovery.ledger"
    assert (
        reduce_recovery(
            execution.records,
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
        ).passed
        is True
    )
    with pytest.raises(ValueError, match="cover every pair"):
        execute_fault_recovery_ledger(
            client,  # type: ignore[arg-type]
            "https://faults.example.test/window",
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
            target_id=_mixture().id,
            backend_topology_digest=_TOPOLOGY,
            mixture=_mixture(),
            sample_limit=1,
            seed=13,
        )


def test_repeated_pairs_do_not_inflate_fault_recovery_clusters() -> None:
    records = [
        ExecutionRecord(
            id=f"agentic-cluster-repeat-{index}",
            track_id="agentic",
            case_id=f"cluster-repeat-{index}",
            attempt_id=f"agentic-cluster-repeat-{index}",
            status="succeeded",
            success=True,
            recovery=_recovery_pair(index).model_copy(
                update={"cluster_id": "shared-cluster"}
            ),
        )
        for index in range(1, 21)
    ]
    repeated = reduce_recovery(records)
    singleton = reduce_recovery(records[:1])

    assert repeated.pair_count == 20
    assert repeated.cluster_count == 1
    assert repeated.cluster_pass_rate == 1
    assert repeated.cluster_pass_rate_confidence_interval == (
        singleton.cluster_pass_rate_confidence_interval
    )
    assert repeated.passed is None


def test_recovery_cluster_reducer_matches_shared_parity_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/recovery_cluster_metric_parity.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == "recovery-cluster-metric-parity.v1"
    rows = fixture["rows"]
    records: list[ExecutionRecord] = []
    for index, row in enumerate(rows, start=1):
        method = _recovery_pair(index).model_copy(
            update={
                "ledger_total_pair_count": len(rows),
                "minimum_pair_count": fixture["minimum_pair_count"],
                "minimum_cluster_count": fixture["minimum_cluster_count"],
                "maximum_recovery_latency_ms": fixture["maximum_recovery_latency_ms"],
                "maximum_retry_amplification": fixture["maximum_retry_amplification"],
                "cluster_id": row["cluster_id"],
                "baseline_recovery_latency_ms": row["baseline_latency_ms"],
                "treatment_recovery_latency_ms": row["treatment_latency_ms"],
                "baseline_retry_count": row["baseline_retry_count"],
                "treatment_retry_count": row["treatment_retry_count"],
            }
        )
        records.append(
            ExecutionRecord(
                id=f"agentic-parity-{index}",
                track_id="agentic",
                case_id=f"parity-{index}",
                attempt_id=f"agentic-parity-{index}",
                status="succeeded" if row["passed"] else "failed",
                success=row["passed"],
                recovery=method,
            )
        )

    reduced = reduce_recovery(records)
    expected = fixture["expected"]
    assert reduced.pair_count == expected["pair_count"]
    assert reduced.cluster_count == expected["cluster_count"]
    assert reduced.cluster_pass_rate == pytest.approx(expected["cluster_pass_rate"])
    assert reduced.cluster_pass_rate_confidence_interval == pytest.approx(
        (
            expected["cluster_pass_rate_lower_95"],
            expected["cluster_pass_rate_upper_95"],
        )
    )
    assert reduced.baseline_success_rate == pytest.approx(
        expected["baseline_success_rate"]
    )
    assert reduced.treatment_success_rate == pytest.approx(
        expected["treatment_success_rate"]
    )
    assert reduced.success_delta == pytest.approx(expected["success_delta"])
    assert reduced.mean_latency_delta_ms == pytest.approx(
        expected["mean_cluster_worst_latency_delta_ms"]
    )
    assert reduced.maximum_retry_amplification == pytest.approx(
        expected["maximum_retry_amplification"]
    )
    assert reduced.passed is expected["passed"]


@pytest.mark.parametrize(
    ("kind", "track_id", "source_id", "payload_field", "expected_level"),
    (
        (
            "fault-recovery",
            "agentic",
            LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
            "recovery",
            "E5",
        ),
        (
            "hard-policy",
            "safety",
            LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
            "hard_policy",
            "E4",
        ),
        (
            "production",
            "preference",
            LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
            "production_experiment",
            "E5",
        ),
    ),
)
def test_live_method_ledgers_require_registered_typed_batch_evidence(
    kind: str,
    track_id: str,
    source_id: str,
    payload_field: str,
    expected_level: EvidenceLevel,
) -> None:
    records = _execute_method_kind(kind)

    assert {record.evidence_kind for record in records} == {source_id}
    assert (
        track_evidence_level("live", LiveRuntimeExecutor.contract, track_id, records)
        == expected_level
    )
    unknown = [
        record.model_copy(update={"evidence_kind": "unknown-ledger.v1"})
        for record in records
    ]
    assert (
        track_evidence_level("live", LiveRuntimeExecutor.contract, track_id, unknown)
        == "E0"
    )
    malformed = [
        records[0].model_copy(update={payload_field: None}),
        *records[1:],
    ]
    assert (
        track_evidence_level("live", LiveRuntimeExecutor.contract, track_id, malformed)
        == "E0"
    )
    assert (
        track_evidence_level(
            "live", LiveRuntimeExecutor.contract, track_id, records[:-1]
        )
        == "E0"
    )
    split_receipt = [
        records[0].model_copy(update={"broker_receipt": "sha256:" + "f" * 64}),
        *records[1:],
    ]
    assert (
        track_evidence_level(
            "live", LiveRuntimeExecutor.contract, track_id, split_receipt
        )
        == "E0"
    )


@pytest.mark.parametrize("kind", ("fault-recovery", "hard-policy", "production"))
@pytest.mark.parametrize(
    "substitution",
    ("target", "topology", "mixture"),
)
def test_method_ledgers_reject_runtime_subject_substitution(
    kind: str, substitution: str
) -> None:
    kwargs: dict[str, object] = {}
    if substitution == "target":
        kwargs["target_id"] = "different-target"
    elif substitution == "topology":
        kwargs["topology_digest"] = _digest("different-topology")
    else:
        kwargs["mixture"] = _mixture().model_copy(
            update={"binding_digest": _digest("different-binding")}
        )
    with pytest.raises(ValueError, match="different runtime snapshot"):
        _execute_method_kind(kind, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ("fault-recovery", "hard-policy", "production"))
@pytest.mark.parametrize(
    ("fetch_delta", "message"),
    (
        (timedelta(seconds=-1), "future"),
        (timedelta(hours=24, seconds=1), "freshness"),
    ),
)
def test_method_ledgers_reject_future_and_stale_seals(
    kind: str, fetch_delta: timedelta, message: str
) -> None:
    sealed_at = {
        "fault-recovery": _fault_recovery_ledger().sealed_at,
        "hard-policy": _hard_policy_ledger().sealed_at,
        "production": _production_ledger().sealed_at,
    }[kind]
    with pytest.raises(ValueError, match=message):
        _execute_method_kind(kind, fetched_at=sealed_at + fetch_delta)
