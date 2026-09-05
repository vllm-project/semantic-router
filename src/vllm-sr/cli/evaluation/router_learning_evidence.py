"""Per-round evidence for the deterministic Router Learning benchmark."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id

ROUTER_LEARNING_EVIDENCE_VERSION = "evaluation-router-learning-method.v1"
ROUTER_LEARNING_CORPUS_REVISION = "router-learning-core-v1"
ROUTER_LEARNING_POLICY_IDS = (
    "static-base",
    "routing-sampling",
    "beta-bernoulli",
)


class RouterLearningMethodEvidence(StrictModel):
    contract_version: Literal["evaluation-router-learning-method.v1"] = (
        ROUTER_LEARNING_EVIDENCE_VERSION
    )
    corpus_revision: Literal["router-learning-core-v1"] = (
        ROUTER_LEARNING_CORPUS_REVISION
    )
    policy_id: Literal["static-base", "routing-sampling", "beta-bernoulli"]
    trial_id: str
    trial_seed: int = Field(ge=0, le=2**32 - 1)
    round_index: int = Field(ge=0)
    candidate_arm_ids: tuple[str, ...] = Field(min_length=2)
    eligible_arm_ids: tuple[str, ...] = Field(min_length=1)
    protected_arm_id: str | None = None
    proposed_arm_id: str
    selected_arm_id: str
    outcome_success: bool
    feedback_delay_rounds: int = Field(ge=0)
    feedback_observed: bool
    protection_required: bool
    protection_violation: bool
    hard_constraint_violation: bool
    call_count: int = Field(ge=1)
    lifecycle_cost_usd: float = Field(ge=0)
    propensity_status: Literal["unsupported"] = "unsupported"

    _portable_ids = field_validator("trial_id", "proposed_arm_id", "selected_arm_id")(
        validate_portable_id
    )

    @field_validator("protected_arm_id")
    @classmethod
    def validate_optional_arm_id(cls, value: str | None) -> str | None:
        return validate_portable_id(value) if value is not None else None

    @field_validator("candidate_arm_ids", "eligible_arm_ids")
    @classmethod
    def validate_arm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Router Learning arm IDs must be unique")
        for arm_id in value:
            validate_portable_id(arm_id)
        return value

    @model_validator(mode="after")
    def validate_relations(self) -> RouterLearningMethodEvidence:
        if self.policy_id not in ROUTER_LEARNING_POLICY_IDS:
            raise ValueError("Router Learning policy is unsupported")
        if not set(self.eligible_arm_ids).issubset(self.candidate_arm_ids):
            raise ValueError("eligible Router Learning arms must be candidates")
        if self.proposed_arm_id not in self.candidate_arm_ids:
            raise ValueError("Router Learning proposal must be a candidate")
        if self.selected_arm_id not in self.candidate_arm_ids:
            raise ValueError("Router Learning selection must be a candidate")
        if self.protection_required != (self.protected_arm_id is not None):
            raise ValueError("Router Learning protection coordinates disagree")
        if self.protected_arm_id is not None and (
            self.protected_arm_id not in self.eligible_arm_ids
        ):
            raise ValueError("Router Learning protected arm must be eligible")
        if self.protection_violation != (
            self.protected_arm_id is not None
            and self.selected_arm_id != self.protected_arm_id
        ):
            raise ValueError("Router Learning protection outcome disagrees")
        if self.hard_constraint_violation != (
            self.selected_arm_id not in self.eligible_arm_ids
        ):
            raise ValueError("Router Learning eligibility outcome disagrees")
        return self
