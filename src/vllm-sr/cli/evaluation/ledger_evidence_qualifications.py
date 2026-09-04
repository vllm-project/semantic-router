"""Qualification contracts for complete server-brokered evidence ledgers."""

from __future__ import annotations

from cli.evaluation.agent_task_evidence import AgentTaskMethodEvidence
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_qualification import (
    EvidenceAttestationRequirement,
    EvidenceQualificationContract,
    EvidenceReceiptRequirement,
    TypedEvidencePayloadRequirement,
    status_matches_success,
)
from cli.evaluation.evidence_source_ids import (
    LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID,
    LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
    LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
    LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.method_evidence import (
    HardPolicyMethodEvidence,
    ProductionExperimentMethodEvidence,
    RecoveryMethodEvidence,
)


def _agent_task_payload(record: ExecutionRecord) -> bool:
    method = record.agent_task
    return (
        method is not None
        and status_matches_success(record)
        and method.task_success is record.success
        and record.selected_arm_id == method.selected_arm_id
        and record.quality == method.task_score
        and record.trajectory_steps == method.trajectory_steps
        and record.tool_calls == method.tool_call_count
        and record.invalid_tool_calls == method.invalid_tool_call_count
        and record.privacy_violations == method.privacy_exposure_count
        and record.input_tokens == method.input_tokens
        and record.output_tokens == method.output_tokens
        and record.runtime_cost == method.runtime_cost_usd
        and record.evaluation_cost == method.evaluation_cost_usd
        and record.grader == method.grader_id
    )


def _agent_task_attestation(record: ExecutionRecord) -> bool:
    method = record.agent_task
    return bool(
        method is not None
        and method.execution_receipt_digest
        and method.grading_receipt_digest
        and method.privacy_audit_receipt_digest
    )


def _agent_task_ledger_identity(
    method: AgentTaskMethodEvidence,
) -> tuple[object, ...]:
    return (
        method.ledger_id,
        method.source_id,
        method.suite_id,
        method.suite_revision,
        method.task_set_digest,
        method.benchmark_parity_claim,
        method.execution_semantics,
        method.policy_snapshot_digest,
        method.config_digest,
        method.target_id,
        method.backend_topology_digest,
        method.mixture_snapshot_digest,
        method.ledger_total_attempt_count,
        method.ledger_total_distinct_task_count,
        method.minimum_distinct_task_count,
        method.minimum_attempts_per_task,
    )


def _agent_task_batch(records: list[ExecutionRecord]) -> bool:
    methods = tuple(record.agent_task for record in records)
    if any(method is None for method in methods):
        return False
    typed = tuple(method for method in methods if method is not None)
    first = typed[0]
    common_identity = _agent_task_ledger_identity(first)
    if len(typed) != first.ledger_total_attempt_count or any(
        _agent_task_ledger_identity(method) != common_identity for method in typed
    ):
        return False
    task_counts: dict[str, int] = {}
    for method in typed:
        task_counts[method.task_id] = task_counts.get(method.task_id, 0) + 1
    return (
        len({method.attempt_id for method in typed}) == len(typed)
        and len({method.trajectory_id for method in typed}) == len(typed)
        and len({(method.task_id, method.repetition_id) for method in typed})
        == len(typed)
        and len({(method.task_id, method.seed) for method in typed}) == len(typed)
        and len(task_counts) == first.ledger_total_distinct_task_count
        and len(task_counts) >= first.minimum_distinct_task_count
        and all(
            count >= first.minimum_attempts_per_task for count in task_counts.values()
        )
        and len({receipt for method in typed for receipt in method.receipts})
        == sum(len(method.receipts) for method in typed)
    )


def _recovery_passed(method: RecoveryMethodEvidence) -> bool:
    retry_amplification = (method.treatment_retry_count + 1) / (
        method.baseline_retry_count + 1
    )
    return (
        method.injection_observed
        and method.recovered
        and method.state_preserved
        and method.treatment_terminal_success
        and method.duplicate_side_effect_count == 0
        and method.treatment_recovery_latency_ms <= method.maximum_recovery_latency_ms
        and retry_amplification <= method.maximum_retry_amplification
    )


def _recovery_payload(record: ExecutionRecord) -> bool:
    method = record.recovery
    if method is None:
        return False
    passed = _recovery_passed(method)
    return (
        record.status == ("succeeded" if passed else "failed")
        and record.success is passed
        and record.quality == float(passed)
    )


def _recovery_attestation(record: ExecutionRecord) -> bool:
    method = record.recovery
    return bool(
        method is not None
        and method.fault_plan_digest
        and method.fault_injection_receipt_digest
        and method.baseline_record_digest
        and method.treatment_record_digest
    )


def _recovery_ledger_identity(
    method: RecoveryMethodEvidence,
) -> tuple[object, ...]:
    return (
        method.ledger_id,
        method.source_id,
        method.policy_snapshot_digest,
        method.config_digest,
        method.target_id,
        method.backend_topology_digest,
        method.mixture_snapshot_digest,
        method.ledger_total_pair_count,
        method.minimum_pair_count,
        method.minimum_distinct_seed_count,
        method.minimum_cluster_count,
        method.maximum_recovery_latency_ms,
        method.maximum_retry_amplification,
    )


def _recovery_batch(records: list[ExecutionRecord]) -> bool:
    methods = tuple(record.recovery for record in records)
    if any(method is None for method in methods):
        return False
    typed = tuple(method for method in methods if method is not None)
    first = typed[0]
    common_identity = _recovery_ledger_identity(first)
    if len(typed) != first.ledger_total_pair_count or any(
        _recovery_ledger_identity(method) != common_identity for method in typed
    ):
        return False
    return (
        len(typed) >= first.minimum_pair_count
        and len({method.seed for method in typed}) >= first.minimum_distinct_seed_count
        and len({method.cluster_id for method in typed}) >= first.minimum_cluster_count
        and len({method.fault_id for method in typed}) == len(typed)
        and len({(method.cohort_pair_id, method.repetition_id) for method in typed})
        == len(typed)
        and len({method.fault_injection_receipt_digest for method in typed})
        == len(typed)
    )


def _hard_policy_payload(record: ExecutionRecord) -> bool:
    method = record.hard_policy
    return (
        method is not None
        and record.status == "succeeded"
        and record.success is True
        and record.should_block is method.should_block
        and record.blocked is method.blocked
        and record.safety_violations == method.violations
        and record.quality
        == float(method.blocked == method.should_block and method.violations == 0)
    )


def _hard_policy_attestation(record: ExecutionRecord) -> bool:
    method = record.hard_policy
    return bool(
        method is not None
        and method.decision_receipt_id
        and method.proof.runtime_instance_digest
        and (method.rule_id, method.enforcement_point)
        in {
            (binding.rule_id, binding.enforcement_point)
            for binding in method.proof.required_bindings
        }
    )


def _hard_policy_batch(records: list[ExecutionRecord]) -> bool:
    methods = tuple(record.hard_policy for record in records)
    if any(method is None for method in methods):
        return False
    typed = tuple(method for method in methods if method is not None)
    proof = typed[0].proof
    required = {
        (binding.rule_id, binding.enforcement_point)
        for binding in proof.required_bindings
    }
    return (
        len(typed) == proof.ledger_total_observation_count
        and all(method.proof == proof for method in typed)
        and len({method.observation_id for method in typed}) == len(typed)
        and len({method.attack_id for method in typed}) == len(typed)
        and len({method.decision_receipt_id for method in typed}) == len(typed)
        and {(method.rule_id, method.enforcement_point) for method in typed} == required
    )


def _production_payload(record: ExecutionRecord) -> bool:
    method = record.production_experiment
    if method is None:
        return False
    preference = record.online_preference
    return (
        record.status == "succeeded"
        and record.success is True
        and record.selected_arm_id == method.assigned_policy_arm_id
        and record.behavior_propensity == method.behavior_propensity
        and (
            (preference is None and record.quality is None)
            or (
                preference is not None
                and preference.experiment == method
                and record.quality == preference.outcome.reward
            )
        )
    )


def _production_attestation(record: ExecutionRecord) -> bool:
    method = record.production_experiment
    return bool(
        method is not None
        and method.participant_digest
        and method.rollback_receipt_id
        and method.ledger_sealed_at.tzinfo is not None
    )


def _production_ledger_identity(
    method: ProductionExperimentMethodEvidence,
) -> tuple[object, ...]:
    return (
        method.experiment_id,
        method.ledger_id,
        method.ledger_total_assignment_count,
        method.ledger_total_outcome_count,
        method.source_id,
        method.policy_snapshot_digest,
        method.config_digest,
        method.target_id,
        method.backend_topology_digest,
        method.mixture_snapshot_digest,
        method.environment,
        method.assignment_scheme,
        method.policy_arms,
        method.minimum_effective_sample_size,
        method.minimum_effective_sample_ratio,
        method.minimum_segment_sample_size,
        method.minimum_assignment_count,
        method.minimum_reward_lift,
        method.confidence_level,
        method.risk_budget_max_rate,
        method.stop_rule_id,
        method.stop_rule_evaluated_at,
        method.stop_triggered,
        method.rollback_receipt_id,
        method.rollback_validated_at,
        method.rollback_ready,
        method.rollback_executed_at,
        method.rollback_succeeded,
        method.ledger_sealed_at,
    )


def _production_batch(records: list[ExecutionRecord]) -> bool:
    methods = tuple(record.production_experiment for record in records)
    if any(method is None for method in methods):
        return False
    typed = tuple(method for method in methods if method is not None)
    first = typed[0]
    common_identity = _production_ledger_identity(first)
    if len(typed) != first.ledger_total_assignment_count or any(
        _production_ledger_identity(method) != common_identity for method in typed
    ):
        return False
    return (
        len(typed) >= first.minimum_assignment_count
        and len({method.assignment_id for method in typed}) == len(typed)
        and len({method.exposure_id for method in typed}) == len(typed)
        and len({method.participant_digest for method in typed}) == len(typed)
        and sum(record.online_preference is not None for record in records)
        == first.ledger_total_outcome_count
    )


LIVE_LEDGER_EVIDENCE_QUALIFICATION_CONTRACTS = (
    EvidenceQualificationContract(
        source_id=LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID,
        allowed_tracks=("agentic",),
        level="E5",
        ceiling="E5",
        payload=TypedEvidencePayloadRequirement(
            field_name="agent_task",
            payload_type=AgentTaskMethodEvidence,
            validator=_agent_task_payload,
        ),
        receipt=EvidenceReceiptRequirement(scope="batch"),
        attestations=(
            EvidenceAttestationRequirement(
                id="sealed-agent-task-attempt",
                validator=_agent_task_attestation,
            ),
        ),
        batch_validator=_agent_task_batch,
    ),
    EvidenceQualificationContract(
        source_id=LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
        allowed_tracks=("agentic",),
        level="E5",
        ceiling="E5",
        payload=TypedEvidencePayloadRequirement(
            field_name="recovery",
            payload_type=RecoveryMethodEvidence,
            validator=_recovery_payload,
        ),
        receipt=EvidenceReceiptRequirement(scope="batch"),
        attestations=(
            EvidenceAttestationRequirement(
                id="sealed-fault-injection-pair",
                validator=_recovery_attestation,
            ),
        ),
        batch_validator=_recovery_batch,
    ),
    EvidenceQualificationContract(
        source_id=LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
        allowed_tracks=("safety",),
        level="E4",
        ceiling="E4",
        payload=TypedEvidencePayloadRequirement(
            field_name="hard_policy",
            payload_type=HardPolicyMethodEvidence,
            validator=_hard_policy_payload,
        ),
        receipt=EvidenceReceiptRequirement(scope="batch"),
        attestations=(
            EvidenceAttestationRequirement(
                id="static-proof-bound-decision",
                validator=_hard_policy_attestation,
            ),
        ),
        batch_validator=_hard_policy_batch,
    ),
    EvidenceQualificationContract(
        source_id=LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
        allowed_tracks=("preference",),
        level="E5",
        ceiling="E5",
        payload=TypedEvidencePayloadRequirement(
            field_name="production_experiment",
            payload_type=ProductionExperimentMethodEvidence,
            validator=_production_payload,
        ),
        receipt=EvidenceReceiptRequirement(scope="batch"),
        attestations=(
            EvidenceAttestationRequirement(
                id="sealed-randomized-assignment",
                validator=_production_attestation,
            ),
        ),
        batch_validator=_production_batch,
    ),
)
