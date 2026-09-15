"""Typed evidence owned by source-qualified evaluation methods."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id

MINIMUM_RECOVERY_PAIR_COUNT = 20
MINIMUM_RECOVERY_CLUSTER_COUNT = 20
MINIMUM_RECOVERY_DISTINCT_SEED_COUNT = 5
MINIMUM_PRODUCTION_ASSIGNMENT_COUNT = 20
MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_SIZE = 10.0
MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_RATIO = 0.5
MINIMUM_PRODUCTION_SEGMENT_SAMPLE_SIZE = 5
MINIMUM_PRODUCTION_REWARD_LIFT = 0.0
MAXIMUM_PRODUCTION_RISK_BUDGET_RATE = 0.2
PRODUCTION_POLICY_ARM_COUNT = 2
PROBABILITY_BINDING_TOLERANCE = 1e-12


class RobustnessMethodEvidence(StrictModel):
    """One exact declared-shift pair.

    ``routerarena.robustness.v1`` is retained only for E0 normalized replay
    diagnostics. ``declared-shift.server-live.v1`` is the server-portable
    record emitted after the installed relation is executed against a live
    brokered Mixture. It does not claim parity with an upstream native runner.
    """

    method_id: Literal["routerarena.robustness.v1", "declared-shift.server-live.v1"]
    suite_id: str | None = None
    suite_revision: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    qualification_receipt_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    perturbation_artifact_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    pair_id: str
    source_case_id: str
    target_case_id: str
    shift_type: Literal["paraphrase"]
    relation: Literal["invariant", "expected_change"]
    source_action_id: str
    expected_action_id: str | None = None
    slice_ids: tuple[str, ...] = Field(min_length=1)
    native_pair_count: int = Field(ge=1, strict=True)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _pair_id = field_validator("pair_id", "source_case_id")(_validate_id)

    @model_validator(mode="after")
    def validate_relation(self) -> RobustnessMethodEvidence:
        if self.source_case_id == self.target_case_id:
            raise ValueError("robustness source and target cases must differ")
        if self.relation == "expected_change" and self.expected_action_id is None:
            raise ValueError("expected-change robustness requires an expected action")
        if self.relation == "invariant" and self.expected_action_id is not None:
            raise ValueError("invariant robustness cannot declare an expected action")
        if not self.source_action_id.strip():
            raise ValueError("robustness source action must be non-empty")
        if len(self.slice_ids) != len(set(self.slice_ids)) or any(
            not item or item.strip() != item for item in self.slice_ids
        ):
            raise ValueError("robustness slices must be unique non-empty values")
        live_bindings = (
            self.suite_id,
            self.suite_revision,
            self.qualification_receipt_digest,
            self.perturbation_artifact_digest,
        )
        if self.method_id == "declared-shift.server-live.v1":
            if any(value is None for value in live_bindings):
                raise ValueError(
                    "server-live declared-shift evidence requires exact suite bindings"
                )
            _validate_id(self.suite_id)
        elif any(value is not None for value in live_bindings):
            raise ValueError(
                "normalized replay robustness cannot claim server-live suite bindings"
            )
        return self


class RecoveryMethodEvidence(StrictModel):
    """One live baseline/treatment exact-step injected-fault receipt pair."""

    method_id: Literal["live-fault-recovery.v1"]
    ledger_id: str
    source_id: str
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_total_pair_count: int = Field(ge=1, strict=True)
    minimum_pair_count: int = Field(ge=MINIMUM_RECOVERY_PAIR_COUNT, strict=True)
    minimum_cluster_count: int = Field(ge=MINIMUM_RECOVERY_CLUSTER_COUNT, strict=True)
    minimum_distinct_seed_count: int = Field(
        ge=MINIMUM_RECOVERY_DISTINCT_SEED_COUNT, strict=True
    )
    fault_id: str
    cohort_pair_id: str
    repetition_id: str
    conversation_id: str
    cluster_id: str
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    concurrency: int = Field(ge=1, strict=True)
    treatment_system: Literal["treatment"]
    fault_kind: Literal[
        "timeout",
        "rate_limit",
        "server_error",
        "disconnect",
        "malformed_response",
        "state_loss",
    ]
    fault_sequence: int = Field(ge=0, strict=True)
    failure_turn: int = Field(ge=0, strict=True)
    fault_plan_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    fault_injection_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    baseline_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    treatment_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    injection_observed: bool
    recovered: bool
    state_preserved: bool
    baseline_terminal_success: bool
    treatment_terminal_success: bool
    baseline_recovery_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    treatment_recovery_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    baseline_retry_count: int = Field(ge=0, strict=True)
    treatment_retry_count: int = Field(ge=0, strict=True)
    maximum_recovery_latency_ms: float = Field(gt=0, allow_inf_nan=False)
    maximum_retry_amplification: float = Field(ge=1, allow_inf_nan=False)
    side_effect_scope: Literal["none", "observed"]
    side_effect_count: int = Field(ge=0, strict=True)
    duplicate_side_effect_count: int = Field(ge=0, strict=True)
    observed_at: datetime

    _fault_id = field_validator(
        "ledger_id",
        "source_id",
        "target_id",
        "fault_id",
        "cohort_pair_id",
        "repetition_id",
        "conversation_id",
        "cluster_id",
    )(_validate_id)

    @model_validator(mode="after")
    def validate_scope(self) -> RecoveryMethodEvidence:
        if self.duplicate_side_effect_count > self.side_effect_count:
            raise ValueError("duplicate side effects cannot exceed side effects")
        if self.side_effect_scope == "none" and (
            self.side_effect_count or self.duplicate_side_effect_count
        ):
            raise ValueError("no-side-effect scope cannot contain side effects")
        if self.recovered and not self.injection_observed:
            raise ValueError("recovery cannot be claimed without an observed injection")
        if self.state_preserved and not self.recovered:
            raise ValueError("state preservation requires a recovered trajectory")
        if self.recovered != self.treatment_terminal_success:
            raise ValueError("recovery must bind the treatment terminal outcome")
        if self.observed_at.tzinfo is None:
            raise ValueError(
                "fault-recovery observation timestamp must be timezone-aware"
            )
        return self


class ExperimentPolicyArm(StrictModel):
    """One immutable policy arm in a randomized production experiment."""

    id: str
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    assignment_probability: float = Field(gt=0, lt=1, allow_inf_nan=False)
    target_policy_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    reference_policy_probability: float = Field(ge=0, le=1, allow_inf_nan=False)

    _id = field_validator("id")(_validate_id)


class ProductionExperimentMethodEvidence(StrictModel):
    """One randomized production assignment with operational safety controls."""

    contract_version: Literal["evaluation-production-experiment.v1"]
    experiment_id: str
    ledger_id: str
    ledger_total_assignment_count: int = Field(ge=1, strict=True)
    ledger_total_outcome_count: int = Field(ge=0, strict=True)
    source_id: str
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment: Literal["production"]
    assignment_scheme: Literal["randomized"]
    assignment_id: str
    exposure_id: str
    participant_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    segment_id: str
    policy_arms: tuple[ExperimentPolicyArm, ...] = Field(min_length=2)
    assigned_policy_arm_id: str
    selected_model_id: str | None = None
    assignment_probability: float = Field(gt=0, le=1, allow_inf_nan=False)
    exposure_probability: float = Field(gt=0, le=1, allow_inf_nan=False)
    behavior_propensity: float = Field(gt=0, le=1, allow_inf_nan=False)
    target_policy_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    minimum_effective_sample_size: float = Field(
        ge=MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_SIZE, allow_inf_nan=False
    )
    minimum_effective_sample_ratio: float = Field(
        ge=MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_RATIO, le=1, allow_inf_nan=False
    )
    minimum_segment_sample_size: int = Field(
        ge=MINIMUM_PRODUCTION_SEGMENT_SAMPLE_SIZE, strict=True
    )
    minimum_assignment_count: int = Field(
        ge=MINIMUM_PRODUCTION_ASSIGNMENT_COUNT, strict=True
    )
    minimum_reward_lift: float = Field(
        ge=MINIMUM_PRODUCTION_REWARD_LIFT, le=1, allow_inf_nan=False
    )
    confidence_level: Literal[0.95]
    risk_event: bool
    risk_budget_max_rate: float = Field(
        ge=0, le=MAXIMUM_PRODUCTION_RISK_BUDGET_RATE, allow_inf_nan=False
    )
    stop_rule_id: str
    stop_rule_evaluated_at: datetime
    stop_triggered: bool
    rollback_receipt_id: str
    rollback_validated_at: datetime
    rollback_ready: bool
    rollback_executed_at: datetime | None = None
    rollback_succeeded: bool | None = None
    assigned_at: datetime
    exposed_at: datetime
    ledger_sealed_at: datetime

    _portable_ids = field_validator(
        "experiment_id",
        "ledger_id",
        "source_id",
        "target_id",
        "assignment_id",
        "exposure_id",
        "segment_id",
        "stop_rule_id",
        "rollback_receipt_id",
    )(_validate_id)

    @model_validator(mode="after")
    def validate_assignment(self) -> ProductionExperimentMethodEvidence:
        if self.ledger_total_outcome_count > self.ledger_total_assignment_count:
            raise ValueError("production outcome count cannot exceed assignment count")
        assigned = self._validate_policy_arms()
        self._validate_policy_probabilities(assigned)
        self._validate_timestamps()
        self._validate_rollback_receipt()
        return self

    def _validate_policy_arms(self) -> ExperimentPolicyArm:
        arm_ids = [arm.id for arm in self.policy_arms]
        if len(self.policy_arms) != PRODUCTION_POLICY_ARM_COUNT:
            raise ValueError(
                "production experiment v1 requires exactly two policy arms"
            )
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("production policy arms must be unique")
        by_id = {arm.id: arm for arm in self.policy_arms}
        assigned = by_id.get(self.assigned_policy_arm_id)
        if assigned is None:
            raise ValueError("assigned policy arm must be experiment-eligible")
        return assigned

    def _validate_policy_probabilities(self, assigned: ExperimentPolicyArm) -> None:
        if (
            abs(sum(arm.assignment_probability for arm in self.policy_arms) - 1)
            > PROBABILITY_BINDING_TOLERANCE
        ):
            raise ValueError("policy-arm assignment probabilities must sum to one")
        if (
            abs(sum(arm.target_policy_probability for arm in self.policy_arms) - 1)
            > PROBABILITY_BINDING_TOLERANCE
        ):
            raise ValueError("target-policy probabilities must sum to one")
        if (
            abs(sum(arm.reference_policy_probability for arm in self.policy_arms) - 1)
            > PROBABILITY_BINDING_TOLERANCE
        ):
            raise ValueError("reference-policy probabilities must sum to one")
        if (
            abs(self.assignment_probability - assigned.assignment_probability)
            > PROBABILITY_BINDING_TOLERANCE
        ):
            raise ValueError("assignment probability must bind the assigned policy arm")
        if (
            abs(self.target_policy_probability - assigned.target_policy_probability)
            > PROBABILITY_BINDING_TOLERANCE
        ):
            raise ValueError("target probability must bind the assigned policy arm")
        expected = self.assignment_probability * self.exposure_probability
        if abs(self.behavior_propensity - expected) > PROBABILITY_BINDING_TOLERANCE:
            raise ValueError(
                "behavior propensity must bind assignment and exposure probabilities"
            )

    def _validate_timestamps(self) -> None:
        if not (
            self.assigned_at.tzinfo
            and self.exposed_at.tzinfo
            and self.stop_rule_evaluated_at.tzinfo
            and self.rollback_validated_at.tzinfo
            and self.ledger_sealed_at.tzinfo
            and (self.rollback_executed_at is None or self.rollback_executed_at.tzinfo)
            and self.assigned_at
            <= self.exposed_at
            <= self.stop_rule_evaluated_at
            <= self.rollback_validated_at
            <= self.ledger_sealed_at
        ):
            raise ValueError(
                "production experiment timestamps must be aware and ordered"
            )

    def _validate_rollback_receipt(self) -> None:
        if self.stop_triggered:
            if (
                not self.rollback_ready
                or self.rollback_executed_at is None
                or self.rollback_succeeded is not True
                or self.rollback_executed_at < self.stop_rule_evaluated_at
                or self.rollback_executed_at > self.rollback_validated_at
            ):
                raise ValueError(
                    "triggered stop requires a successful rollback receipt"
                )
        elif (
            self.rollback_executed_at is not None or self.rollback_succeeded is not None
        ):
            raise ValueError("untriggered stop cannot claim a rollback execution")


class OnlinePreferenceOutcome(StrictModel):
    """Observed reward for one exposed assignment."""

    contract_version: Literal["evaluation-online-preference-ledger.v1"]
    outcome_id: str
    assignment_id: str
    exposure_id: str
    participant_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    segment_id: str
    reward: float = Field(ge=0, le=1, allow_inf_nan=False)
    observed_at: datetime

    _ids = field_validator("outcome_id", "assignment_id", "exposure_id", "segment_id")(
        _validate_id
    )

    @model_validator(mode="after")
    def validate_timestamp(self) -> OnlinePreferenceOutcome:
        if self.observed_at.tzinfo is None:
            raise ValueError("preference outcome timestamp must be timezone-aware")
        return self


class OnlinePreferenceMethodEvidence(StrictModel):
    """Preference outcome layered on a production experiment assignment."""

    contract_version: Literal["evaluation-online-preference-method.v1"]
    experiment: ProductionExperimentMethodEvidence
    outcome: OnlinePreferenceOutcome

    @model_validator(mode="after")
    def validate_outcome(self) -> OnlinePreferenceMethodEvidence:
        if (
            self.outcome.assignment_id != self.experiment.assignment_id
            or self.outcome.exposure_id != self.experiment.exposure_id
            or self.outcome.participant_digest != self.experiment.participant_digest
            or self.outcome.segment_id != self.experiment.segment_id
        ):
            raise ValueError("preference outcome does not bind its exposed assignment")
        if (
            self.outcome.observed_at < self.experiment.exposed_at
            or self.outcome.observed_at > self.experiment.stop_rule_evaluated_at
        ):
            raise ValueError("preference outcome timestamp is outside evaluation")
        return self


class HardPolicyEnforcementBinding(StrictModel):
    """One rule at one concrete runtime enforcement point."""

    rule_id: str
    enforcement_point: str

    @model_validator(mode="after")
    def validate_binding(self) -> HardPolicyEnforcementBinding:
        if (
            not self.rule_id
            or self.rule_id.strip() != self.rule_id
            or not self.enforcement_point
            or self.enforcement_point.strip() != self.enforcement_point
        ):
            raise ValueError("hard-policy binding values must be trimmed and non-empty")
        return self


class HardPolicyStaticProof(StrictModel):
    """Router-owned proof for the exact runtime policy/config snapshot."""

    contract_version: Literal["evaluation-hard-policy-proof.v1"]
    proof_id: str
    source_id: str
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_instance_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_total_observation_count: int = Field(ge=1, strict=True)
    required_bindings: tuple[HardPolicyEnforcementBinding, ...] = Field(min_length=1)
    verified_at: datetime

    _ids = field_validator("proof_id", "source_id", "target_id")(_validate_id)

    @model_validator(mode="after")
    def validate_proof(self) -> HardPolicyStaticProof:
        pairs = [
            (binding.rule_id, binding.enforcement_point)
            for binding in self.required_bindings
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError("hard-policy required bindings must be unique")
        if self.verified_at.tzinfo is None:
            raise ValueError("hard-policy proof timestamp must be timezone-aware")
        return self


class HardPolicyMethodEvidence(StrictModel):
    """One brokered dynamic attack/block row bound to a static runtime proof."""

    contract_version: Literal["evaluation-hard-policy-observation.v1"]
    proof: HardPolicyStaticProof
    observation_id: str
    attack_id: str
    rule_id: str
    enforcement_point: str
    decision_receipt_id: str
    should_block: bool
    blocked: bool
    violations: int = Field(ge=0, strict=True)
    observed_at: datetime

    _ids = field_validator("observation_id", "attack_id", "decision_receipt_id")(
        _validate_id
    )

    @model_validator(mode="after")
    def validate_observation(self) -> HardPolicyMethodEvidence:
        required = {
            (binding.rule_id, binding.enforcement_point)
            for binding in self.proof.required_bindings
        }
        if (self.rule_id, self.enforcement_point) not in required:
            raise ValueError("dynamic safety row references an unproved binding")
        if self.observed_at.tzinfo is None or self.observed_at < self.proof.verified_at:
            raise ValueError("dynamic safety row predates the static proof")
        if self.violations and self.blocked:
            raise ValueError("blocked hard-policy attacks cannot contain violations")
        return self
