"""Shared immutable fixtures for evaluation method-ledger contract tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.fault_recovery_ledger import (
    FAULT_RECOVERY_LEDGER_VERSION,
    FaultRecoveryLedger,
    execute_fault_recovery_ledger,
)
from cli.evaluation.hard_policy_ledger import (
    HARD_POLICY_LEDGER_VERSION,
    HardPolicyLedger,
    execute_hard_policy_ledger,
)
from cli.evaluation.http_client import HTTPResult
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.method_evidence import (
    ExperimentPolicyArm,
    HardPolicyEnforcementBinding,
    HardPolicyMethodEvidence,
    HardPolicyStaticProof,
    OnlinePreferenceOutcome,
    ProductionExperimentMethodEvidence,
    RecoveryMethodEvidence,
)
from cli.evaluation.method_ledger_identity import method_mixture_binding
from cli.evaluation.production_experiment_ledger import (
    PRODUCTION_EXPERIMENT_LEDGER_VERSION,
    ProductionExperimentLedger,
    execute_production_experiment_ledger,
)
from cli.evaluation.target_contracts import (
    EvaluationTargetArm,
    ManifestMixture,
    MixtureDecisionBinding,
)
from evaluation_contract_test_support import build_routing_recipe_plan


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


_START = datetime(2026, 8, 30, 1, tzinfo=UTC)
_POLICY = _digest("1")
_CONFIG = _digest("2")
_BROKER_RECEIPT = _digest("3")
_TOPOLOGY = _digest("method-topology")


class _LedgerClient:
    def __init__(
        self, payload: dict[str, object], *, fetched_at: datetime | None = None
    ):
        self.payload = payload
        self.fetched_at = fetched_at or _START + timedelta(hours=1)
        self.calls: list[dict[str, object]] = []

    def get(self, endpoint: str, **kwargs: object) -> HTTPResult:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return HTTPResult(
            success=True,
            status_code=200,
            payload=self.payload,
            latency_ms=1.0,
            headers={},
            broker_receipt=_BROKER_RECEIPT,
            fetched_at=self.fetched_at,
        )


def _policy_arms() -> tuple[ExperimentPolicyArm, ...]:
    return (
        ExperimentPolicyArm(
            id="policy-a",
            config_digest=_digest("4"),
            assignment_probability=0.5,
            target_policy_probability=0.0,
            reference_policy_probability=1.0,
        ),
        ExperimentPolicyArm(
            id="policy-b",
            config_digest=_digest("5"),
            assignment_probability=0.5,
            target_policy_probability=1.0,
            reference_policy_probability=0.0,
        ),
    )


def _production_assignment(
    index: int,
    *,
    assignment_count: int,
    total_outcomes: int,
    risk_event: bool = False,
    stop_triggered: bool = False,
) -> ProductionExperimentMethodEvidence:
    arm = "policy-a" if index % 2 else "policy-b"
    return ProductionExperimentMethodEvidence(
        contract_version="evaluation-production-experiment.v1",
        experiment_id="experiment-1",
        ledger_id="ledger-1",
        ledger_total_assignment_count=assignment_count,
        ledger_total_outcome_count=total_outcomes,
        source_id="production-router",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture_snapshot_digest=method_mixture_binding(_mixture()).snapshot_digest,
        environment="production",
        assignment_scheme="randomized",
        assignment_id=f"assignment-{index}",
        exposure_id=f"exposure-{index}",
        participant_digest=_digest(str(index + 5)),
        segment_id="segment-a",
        policy_arms=_policy_arms(),
        assigned_policy_arm_id=arm,
        selected_model_id=f"model-{arm[-1]}",
        assignment_probability=0.5,
        exposure_probability=1.0,
        behavior_propensity=0.5,
        target_policy_probability=0.0 if arm == "policy-a" else 1.0,
        minimum_effective_sample_size=10,
        minimum_effective_sample_ratio=0.5,
        minimum_segment_sample_size=20,
        minimum_assignment_count=20,
        minimum_reward_lift=0.1,
        confidence_level=0.95,
        risk_event=risk_event,
        risk_budget_max_rate=0.2,
        stop_rule_id="stop-rule-1",
        stop_rule_evaluated_at=_START + timedelta(minutes=10),
        stop_triggered=stop_triggered,
        rollback_receipt_id="rollback-receipt-1",
        rollback_validated_at=_START + timedelta(minutes=12),
        rollback_ready=True,
        rollback_executed_at=(
            _START + timedelta(minutes=11) if stop_triggered else None
        ),
        rollback_succeeded=True if stop_triggered else None,
        assigned_at=_START + timedelta(seconds=index),
        exposed_at=_START + timedelta(seconds=index + 30),
        ledger_sealed_at=_START + timedelta(minutes=13),
    )


def _production_ledger(
    *,
    assignment_count: int = 20,
    outcome_count: int | None = None,
    risk_event_count: int = 0,
    target_reward: float = 1.0,
    reference_reward: float = 0.5,
    stop_triggered: bool = False,
) -> ProductionExperimentLedger:
    if outcome_count is None:
        outcome_count = assignment_count
    assignments = tuple(
        _production_assignment(
            index,
            assignment_count=assignment_count,
            total_outcomes=outcome_count,
            risk_event=index <= risk_event_count,
            stop_triggered=stop_triggered,
        )
        for index in range(1, assignment_count + 1)
    )
    outcomes = tuple(
        OnlinePreferenceOutcome(
            contract_version="evaluation-online-preference-ledger.v1",
            outcome_id=f"outcome-{index}",
            assignment_id=assignments[index - 1].assignment_id,
            exposure_id=assignments[index - 1].exposure_id,
            participant_digest=assignments[index - 1].participant_digest,
            segment_id=assignments[index - 1].segment_id,
            reward=(
                reference_reward
                if assignments[index - 1].assigned_policy_arm_id == "policy-a"
                else target_reward
            ),
            observed_at=_START + timedelta(minutes=5, seconds=index),
        )
        for index in range(1, outcome_count + 1)
    )
    return ProductionExperimentLedger(
        contract_version=PRODUCTION_EXPERIMENT_LEDGER_VERSION,
        experiment_id="experiment-1",
        ledger_id="ledger-1",
        source_id="production-router",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=method_mixture_binding(_mixture()),
        environment="production",
        assignment_scheme="randomized",
        risk_budget_max_rate=0.2,
        stop_rule_id="stop-rule-1",
        stop_rule_evaluated_at=_START + timedelta(minutes=10),
        stop_triggered=stop_triggered,
        rollback_receipt_id="rollback-receipt-1",
        rollback_validated_at=_START + timedelta(minutes=12),
        rollback_ready=True,
        rollback_executed_at=(
            _START + timedelta(minutes=11) if stop_triggered else None
        ),
        rollback_succeeded=True if stop_triggered else None,
        minimum_effective_sample_size=10,
        minimum_effective_sample_ratio=0.5,
        minimum_segment_sample_size=20,
        minimum_assignment_count=20,
        minimum_reward_lift=0.1,
        confidence_level=0.95,
        window_started_at=_START,
        window_ended_at=_START + timedelta(minutes=10),
        sealed_at=_START + timedelta(minutes=13),
        assignments=assignments,
        preference_outcomes=outcomes,
    )


def _model_arms() -> tuple[EvaluationTargetArm, ...]:
    return tuple(
        EvaluationTargetArm(
            id=f"model-{suffix}",
            model=f"provider/model-{suffix}",
            provider_model_id_digest=_digest(digit),
            input_cost_per_million_tokens_usd=0,
            output_cost_per_million_tokens_usd=0,
        )
        for suffix, digit in (("a", "8"), ("b", "9"))
    )


def _mixture() -> ManifestMixture:
    arms = _model_arms()
    recipe_name = "method-ledger-recipe"
    selector_policy = _digest("method-selector-policy")
    recipe_digest = _digest("method-recipe")
    pool_digest = model_pool_snapshot_digest(arms)
    selector_digest = selector_snapshot_digest(selector_policy, ())
    adaptation_digest = _digest("method-adaptation")
    binding_digest = _digest("method-binding")
    return ManifestMixture(
        id=mixture_target_id(recipe_name),
        entrypoint_model="method-entrypoint",
        aliases=("method-entrypoint",),
        recipe_name=recipe_name,
        recipe_description="Frozen method-ledger evaluation subject",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=(),
        fallback_arm_id=arms[0].id,
        decisions=(
            MixtureDecisionBinding(
                name="default",
                algorithm="single",
                arm_ids=tuple(sorted(arm.id for arm in arms)),
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=tuple(arm.id for arm in arms),
            fallback_arm_id=arms[0].id,
            signals=(),
            projections=(),
        ),
    )


def _execute_production(
    ledger: ProductionExperimentLedger, *, sample_limit: int | None = None
):
    client = _LedgerClient(ledger.model_dump(mode="json"))
    execution = execute_production_experiment_ledger(
        client,  # type: ignore[arg-type]
        "https://ledger.example.test/window",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=_mixture(),
        model_arms=_model_arms(),
        sample_limit=sample_limit or len(ledger.assignments),
        seed=7,
    )
    assert client.calls[0]["broker_operation"] == "production.experiment-ledger"
    return execution


def _hard_policy_ledger() -> HardPolicyLedger:
    bindings = (
        HardPolicyEnforcementBinding(
            rule_id="jailbreak", enforcement_point="extproc-request"
        ),
        HardPolicyEnforcementBinding(
            rule_id="pii-output", enforcement_point="extproc-response"
        ),
    )
    proof = HardPolicyStaticProof(
        contract_version="evaluation-hard-policy-proof.v1",
        proof_id="proof-1",
        source_id="production-router",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture_snapshot_digest=method_mixture_binding(_mixture()).snapshot_digest,
        runtime_instance_digest=_digest("a"),
        ledger_total_observation_count=2,
        required_bindings=bindings,
        verified_at=_START,
    )
    observations = tuple(
        HardPolicyMethodEvidence(
            contract_version="evaluation-hard-policy-observation.v1",
            proof=proof,
            observation_id=f"observation-{index}",
            attack_id=f"attack-{index}",
            rule_id=binding.rule_id,
            enforcement_point=binding.enforcement_point,
            decision_receipt_id=f"decision-{index}",
            should_block=True,
            blocked=True,
            violations=0,
            observed_at=_START + timedelta(minutes=index),
        )
        for index, binding in enumerate(bindings, start=1)
    )
    return HardPolicyLedger(
        contract_version=HARD_POLICY_LEDGER_VERSION,
        ledger_id="hard-policy-ledger-1",
        source_id="production-router",
        environment="production",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=method_mixture_binding(_mixture()),
        proof=proof,
        window_started_at=_START,
        window_ended_at=_START + timedelta(minutes=3),
        sealed_at=_START + timedelta(minutes=4),
        observations=observations,
    )


def _recovery_pair(index: int) -> RecoveryMethodEvidence:
    return RecoveryMethodEvidence(
        method_id="live-fault-recovery.v1",
        ledger_id="fault-ledger-1",
        source_id="runtime-router",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture_snapshot_digest=method_mixture_binding(_mixture()).snapshot_digest,
        ledger_total_pair_count=20,
        minimum_pair_count=20,
        minimum_cluster_count=20,
        minimum_distinct_seed_count=5,
        fault_id=f"fault-{index}",
        cohort_pair_id="pair-1",
        repetition_id=f"repetition-{index}",
        conversation_id="conversation-1",
        cluster_id=f"cluster-{index}",
        seed=index % 5,
        concurrency=1,
        treatment_system="treatment",
        fault_kind="timeout",
        fault_sequence=0,
        failure_turn=0,
        fault_plan_digest=_digest(f"fault-plan-{index}"),
        fault_injection_receipt_digest=_digest(f"injection-{index}"),
        baseline_record_digest=_digest(f"baseline-{index}"),
        treatment_record_digest=_digest(f"treatment-{index}"),
        injection_observed=True,
        recovered=True,
        state_preserved=True,
        baseline_terminal_success=False,
        treatment_terminal_success=True,
        baseline_recovery_latency_ms=100,
        treatment_recovery_latency_ms=120,
        baseline_retry_count=1,
        treatment_retry_count=1,
        maximum_recovery_latency_ms=200,
        maximum_retry_amplification=1.5,
        side_effect_scope="none",
        side_effect_count=0,
        duplicate_side_effect_count=0,
        observed_at=_START + timedelta(minutes=index),
    )


def _fault_recovery_ledger() -> FaultRecoveryLedger:
    return FaultRecoveryLedger(
        contract_version=FAULT_RECOVERY_LEDGER_VERSION,
        ledger_id="fault-ledger-1",
        source_id="runtime-router",
        environment="production",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=_mixture().id,
        backend_topology_digest=_TOPOLOGY,
        mixture=method_mixture_binding(_mixture()),
        minimum_pair_count=20,
        minimum_cluster_count=20,
        minimum_distinct_seed_count=5,
        maximum_recovery_latency_ms=200,
        maximum_retry_amplification=1.5,
        window_started_at=_START,
        window_ended_at=_START + timedelta(minutes=21),
        sealed_at=_START + timedelta(minutes=22),
        pairs=tuple(_recovery_pair(index) for index in range(1, 21)),
    )


def _execute_method_kind(
    kind: str,
    *,
    target_id: str | None = None,
    topology_digest: str | None = None,
    mixture: ManifestMixture | None = None,
    fetched_at: datetime | None = None,
) -> list[ExecutionRecord]:
    selected_mixture = mixture or _mixture()
    selected_target = target_id or _mixture().id
    selected_topology = topology_digest or _TOPOLOGY
    if kind == "fault-recovery":
        ledger = _fault_recovery_ledger()
        client = _LedgerClient(ledger.model_dump(mode="json"), fetched_at=fetched_at)
        execution = execute_fault_recovery_ledger(
            client,  # type: ignore[arg-type]
            "https://faults.example.test/window",
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
            target_id=selected_target,
            backend_topology_digest=selected_topology,
            mixture=selected_mixture,
            sample_limit=len(ledger.pairs),
            seed=17,
        )
        return execution.records
    if kind == "hard-policy":
        ledger = _hard_policy_ledger()
        client = _LedgerClient(ledger.model_dump(mode="json"), fetched_at=fetched_at)
        execution = execute_hard_policy_ledger(
            client,  # type: ignore[arg-type]
            "https://policy.example.test/window",
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
            target_id=selected_target,
            backend_topology_digest=selected_topology,
            mixture=selected_mixture,
            sample_limit=len(ledger.observations),
            seed=17,
        )
        return execution.records
    ledger = _production_ledger()
    client = _LedgerClient(ledger.model_dump(mode="json"), fetched_at=fetched_at)
    execution = execute_production_experiment_ledger(
        client,  # type: ignore[arg-type]
        "https://production.example.test/window",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=selected_target,
        backend_topology_digest=selected_topology,
        mixture=selected_mixture,
        model_arms=_model_arms(),
        sample_limit=len(ledger.assignments),
        seed=17,
    )
    return execution.records
