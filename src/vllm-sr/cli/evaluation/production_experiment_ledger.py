"""Strict ingestion of sealed randomized production experiment ledgers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.contract_primitives import Message, StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import (
    LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.http_client import EvaluationHTTPClient
from cli.evaluation.method_evidence import (
    MAXIMUM_PRODUCTION_RISK_BUDGET_RATE,
    MINIMUM_PRODUCTION_ASSIGNMENT_COUNT,
    MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_RATIO,
    MINIMUM_PRODUCTION_EFFECTIVE_SAMPLE_SIZE,
    MINIMUM_PRODUCTION_REWARD_LIFT,
    MINIMUM_PRODUCTION_SEGMENT_SAMPLE_SIZE,
    OnlinePreferenceMethodEvidence,
    OnlinePreferenceOutcome,
    ProductionExperimentMethodEvidence,
)
from cli.evaluation.method_ledger_identity import (
    MethodMixtureBinding,
    method_mixture_binding,
    validate_method_ledger_freshness,
)
from cli.evaluation.target_contracts import EvaluationTargetArm, ManifestMixture

PRODUCTION_EXPERIMENT_LEDGER_VERSION = "evaluation-production-experiment-ledger.v1"


class ProductionExperimentLedger(StrictModel):
    """One sealed production window; preference outcomes are an optional layer."""

    contract_version: Literal[PRODUCTION_EXPERIMENT_LEDGER_VERSION]
    experiment_id: str
    ledger_id: str
    source_id: str
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture: MethodMixtureBinding
    environment: Literal["production"]
    assignment_scheme: Literal["randomized"]
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
    window_started_at: datetime
    window_ended_at: datetime
    sealed_at: datetime
    assignments: tuple[ProductionExperimentMethodEvidence, ...] = Field(min_length=1)
    preference_outcomes: tuple[OnlinePreferenceOutcome, ...] = ()

    _portable_ids = field_validator(
        "experiment_id",
        "ledger_id",
        "source_id",
        "target_id",
        "stop_rule_id",
        "rollback_receipt_id",
    )(_validate_id)

    def _validate_window_timestamps(self) -> None:
        timestamps = (
            self.window_started_at,
            self.window_ended_at,
            self.stop_rule_evaluated_at,
            self.rollback_validated_at,
            self.sealed_at,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("production ledger timestamps must be timezone-aware")
        if not (
            self.window_started_at
            < self.window_ended_at
            <= self.stop_rule_evaluated_at
            <= self.rollback_validated_at
            <= self.sealed_at
        ):
            raise ValueError("production ledger window is not ordered")

    @model_validator(mode="after")
    def validate_window(self) -> ProductionExperimentLedger:
        self._validate_window_timestamps()
        assignments: dict[str, ProductionExperimentMethodEvidence] = {}
        exposures: set[str] = set()
        participants: set[str] = set()
        policy_arms = self.assignments[0].policy_arms
        total_assignments = len(self.assignments)
        total_outcomes = len(self.preference_outcomes)
        for row in self.assignments:
            if (
                row.experiment_id != self.experiment_id
                or row.ledger_id != self.ledger_id
                or row.ledger_total_assignment_count != total_assignments
                or row.ledger_total_outcome_count != total_outcomes
                or row.source_id != self.source_id
                or row.policy_snapshot_digest != self.policy_snapshot_digest
                or row.config_digest != self.config_digest
                or row.target_id != self.target_id
                or row.backend_topology_digest != self.backend_topology_digest
                or row.mixture_snapshot_digest != self.mixture.snapshot_digest
                or row.environment != self.environment
                or row.assignment_scheme != self.assignment_scheme
                or row.risk_budget_max_rate != self.risk_budget_max_rate
                or row.stop_rule_id != self.stop_rule_id
                or row.stop_rule_evaluated_at != self.stop_rule_evaluated_at
                or row.stop_triggered != self.stop_triggered
                or row.rollback_receipt_id != self.rollback_receipt_id
                or row.rollback_validated_at != self.rollback_validated_at
                or row.rollback_ready != self.rollback_ready
                or row.rollback_executed_at != self.rollback_executed_at
                or row.rollback_succeeded != self.rollback_succeeded
                or row.policy_arms != policy_arms
                or row.minimum_effective_sample_size
                != self.minimum_effective_sample_size
                or row.minimum_effective_sample_ratio
                != self.minimum_effective_sample_ratio
                or row.minimum_segment_sample_size != self.minimum_segment_sample_size
                or row.minimum_assignment_count != self.minimum_assignment_count
                or row.minimum_reward_lift != self.minimum_reward_lift
                or row.confidence_level != self.confidence_level
                or row.ledger_sealed_at != self.sealed_at
            ):
                raise ValueError(
                    "production assignment does not bind the sealed ledger"
                )
            if not (
                self.window_started_at
                <= row.assigned_at
                <= row.exposed_at
                <= self.window_ended_at
            ):
                raise ValueError("production assignment lies outside the sealed window")
            if (
                row.assignment_id in assignments
                or row.exposure_id in exposures
                or row.participant_digest in participants
            ):
                raise ValueError(
                    "assignment, exposure, and participant identities must be unique"
                )
            assignments[row.assignment_id] = row
            exposures.add(row.exposure_id)
            participants.add(row.participant_digest)
        outcome_ids: set[str] = set()
        outcome_assignments: set[str] = set()
        for outcome in self.preference_outcomes:
            assignment = assignments.get(outcome.assignment_id)
            if assignment is None:
                raise ValueError("preference outcome references an unknown assignment")
            OnlinePreferenceMethodEvidence(
                contract_version="evaluation-online-preference-method.v1",
                experiment=assignment,
                outcome=outcome,
            )
            if (
                outcome.outcome_id in outcome_ids
                or outcome.assignment_id in outcome_assignments
                or outcome.observed_at > self.window_ended_at
            ):
                raise ValueError(
                    "preference outcomes must be unique and inside the window"
                )
            outcome_ids.add(outcome.outcome_id)
            outcome_assignments.add(outcome.assignment_id)
        return self


@dataclass(frozen=True)
class ProductionExperimentExecution:
    visible: VisibleCaseSet
    grading: GradingCaseSet
    records: list[ExecutionRecord]


def _case_id(ledger_id: str, assignment_id: str) -> str:
    digest = hashlib.sha256(f"{ledger_id}\x00{assignment_id}".encode()).hexdigest()[:24]
    return f"experiment-{digest}"


def _selected_assignments(
    ledger: ProductionExperimentLedger, *, sample_limit: int, seed: int
) -> tuple[ProductionExperimentMethodEvidence, ...]:
    if sample_limit < len(ledger.assignments):
        raise ValueError(
            "sample_limit must cover every assignment in the sealed production window"
        )
    ranked = sorted(
        ledger.assignments,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}\x00{ledger.ledger_id}\x00{row.assignment_id}".encode()
            ).digest(),
            row.assignment_id,
        ),
    )
    return tuple(ranked)


def execute_production_experiment_ledger(
    client: EvaluationHTTPClient,
    endpoint: str,
    *,
    policy_snapshot_digest: str,
    config_digest: str,
    target_id: str,
    backend_topology_digest: str,
    mixture: ManifestMixture,
    model_arms: tuple[EvaluationTargetArm, ...],
    sample_limit: int,
    seed: int,
) -> ProductionExperimentExecution:
    """Fetch a production ledger through the broker and emit assignment rows."""

    result = client.get(
        endpoint,
        track_id="preference",
        case_id="production-ledger",
        attempt_id="ledger-fetch",
        broker_operation="production.experiment-ledger",
    )
    if not result.success or result.payload is None or result.broker_receipt is None:
        raise ValueError("production experiment ledger could not be read")
    ledger = ProductionExperimentLedger.model_validate(result.payload)
    validate_method_ledger_freshness(ledger.sealed_at, result.fetched_at)
    expected_mixture = method_mixture_binding(mixture)
    if (
        ledger.policy_snapshot_digest != policy_snapshot_digest
        or ledger.config_digest != config_digest
        or ledger.target_id != target_id
        or ledger.backend_topology_digest != backend_topology_digest
        or ledger.mixture != expected_mixture
    ):
        raise ValueError("production ledger belongs to a different runtime snapshot")
    selected = _selected_assignments(ledger, sample_limit=sample_limit, seed=seed)
    if not selected:
        raise ValueError("production experiment sampling selected no assignments")
    model_ids = {arm.id for arm in model_arms}
    for row in selected:
        if row.selected_model_id is not None and row.selected_model_id not in model_ids:
            raise ValueError(
                "production ledger references an undeclared selected model"
            )
    outcomes = {row.assignment_id: row for row in ledger.preference_outcomes}
    visible: list[CaseVisible] = []
    grading: list[CaseGrading] = []
    records: list[ExecutionRecord] = []
    for assignment in selected:
        case_id = _case_id(ledger.ledger_id, assignment.assignment_id)
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=("preference",),
                messages=(
                    Message(
                        role="user",
                        content="Sealed randomized production assignment",
                    ),
                ),
                tags=("production-experiment", assignment.segment_id),
            )
        )
        grading.append(CaseGrading(case_id=case_id))
        outcome = outcomes.get(assignment.assignment_id)
        preference = (
            OnlinePreferenceMethodEvidence(
                contract_version="evaluation-online-preference-method.v1",
                experiment=assignment,
                outcome=outcome,
            )
            if outcome is not None
            else None
        )
        records.append(
            ExecutionRecord(
                id=f"preference-{case_id}",
                track_id="preference",
                case_id=case_id,
                attempt_id=f"preference-{case_id}",
                status="succeeded",
                selected_arm_id=assignment.assigned_policy_arm_id,
                success=True,
                quality=outcome.reward if outcome is not None else None,
                behavior_propensity=assignment.behavior_propensity,
                production_experiment=assignment,
                online_preference=preference,
                evidence_kind=LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
            )
        )
    return ProductionExperimentExecution(
        visible=VisibleCaseSet(cases=tuple(visible)),
        grading=GradingCaseSet(cases=tuple(grading)),
        records=records,
    )
