"""Strict, versioned workload and run contracts for evaluation execution."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from cli.evaluation import target_contracts as _target_contracts
from cli.evaluation.capacity_load_contract import (
    CAPACITY_LOAD_CONFIDENCE_LEVEL,
    CAPACITY_LOAD_KIND,
    CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE,
    MAX_CAPACITY_MEASUREMENT_REQUESTS,
    MAX_CAPACITY_REPETITIONS,
    MAX_CAPACITY_STABILITY_CV,
    MAX_CAPACITY_WARMUP_MULTIPLIER,
    MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL,
    MIN_CAPACITY_MEASUREMENT_REQUESTS,
    MIN_CAPACITY_REPETITIONS,
    MIN_CAPACITY_WARMUP_MULTIPLIER,
    capacity_concurrency_levels,
)
from cli.evaluation.constants import BUILTIN_SUITE_IDS, SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contract_primitives import (
    ArtifactRef as _ArtifactRef,
)
from cli.evaluation.contract_primitives import (
    Message as _Message,
)
from cli.evaluation.contract_primitives import (
    StrictModel as _StrictModel,
)
from cli.evaluation.contract_validation import (
    is_portable_id,
    is_valid_suite_revision,
    validate_canonical_uuid,
    validate_portable_id,
    validate_run_description,
    validate_run_name,
)
from cli.evaluation.gate_contract import GATE_CONTRACT_VERSION, ChangeProfile
from cli.evaluation.manifest_identity import (
    require_manifest_digest,
    seal_manifest_fields,
)

_MAX_SUITE_EXECUTOR_ID_LENGTH = 128
_MINIMUM_MIXTURE_ARM_COUNT = 2
_MINIMUM_LIVE_CAPACITY_CONCURRENCY = 2


class CaseVisible(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    track_ids: tuple[str, ...] = Field(min_length=1)
    messages: tuple[_Message, ...] = Field(min_length=1)
    modality: Literal["text", "image", "document", "audio", "video"] = "text"
    tags: tuple[str, ...] = ()
    trajectory_id: str | None = None

    _id = field_validator("id")(validate_portable_id)

    @field_validator("track_ids")
    @classmethod
    def validate_track_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("case track ids must be unique")
        canonical = tuple(track_id for track_id in TRACK_IDS if track_id in value)
        if value != canonical:
            raise ValueError("case track ids must be known and use canonical order")
        return value

    @model_validator(mode="after")
    def validate_track_applicability(self) -> CaseVisible:
        if self.modality == "text" and "multimodal" in self.track_ids:
            raise ValueError("text cases cannot plan multimodal evidence")
        return self


class CaseGrading(_StrictModel):
    """Hidden labels loaded only after policy/model execution."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    expected_route: str | None = None
    expected_answer: str | None = None
    preferred_arm_id: str | None = None
    expected_tools: tuple[str, ...] = ()
    should_block: bool | None = None
    weight: float = Field(default=1.0, gt=0)

    _case_id = field_validator("case_id")(validate_portable_id)


class VisibleCaseSet(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    cases: tuple[CaseVisible, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> VisibleCaseSet:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("visible case ids must be unique")
        return self


class GradingCaseSet(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    cases: tuple[CaseGrading, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> GradingCaseSet:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("grading case ids must be unique")
        return self


class WorkloadSnapshot(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    visible_cases: _ArtifactRef
    grading_cases: _ArtifactRef

    _id = field_validator("id")(validate_portable_id)

    @model_validator(mode="after")
    def physically_separated(self) -> WorkloadSnapshot:
        if self.visible_cases.digest == self.grading_cases.digest:
            raise ValueError("visible and grading cases must be separate artifacts")
        return self


class CapacitySLO(_StrictModel):
    """Frozen service-level objective for a repeated live load profile."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    required_concurrency: int = Field(ge=1, le=128)
    max_latency_p95_ms: float = Field(gt=0, allow_inf_nan=False)
    max_error_rate: float = Field(ge=0, lt=1, allow_inf_nan=False)
    min_throughput_rps: float = Field(gt=0, allow_inf_nan=False)
    min_throughput_scaling_efficiency: float = Field(gt=0, le=1, allow_inf_nan=False)


class CapacityLoadProtocol(_StrictModel):
    """Frozen repeated closed-loop measurement protocol for a live load claim."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    kind: Literal[CAPACITY_LOAD_KIND] = CAPACITY_LOAD_KIND
    concurrency_levels: tuple[int, ...] = Field(min_length=2, max_length=8)
    warmup_request_multiplier: int = Field(
        ge=MIN_CAPACITY_WARMUP_MULTIPLIER,
        le=MAX_CAPACITY_WARMUP_MULTIPLIER,
    )
    measurement_requests_per_repetition: int = Field(
        ge=MIN_CAPACITY_MEASUREMENT_REQUESTS,
        le=MAX_CAPACITY_MEASUREMENT_REQUESTS,
    )
    repetitions_per_level: int = Field(
        ge=MIN_CAPACITY_REPETITIONS,
        le=MAX_CAPACITY_REPETITIONS,
    )
    minimum_measurement_clusters_per_level: Literal[
        MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL
    ] = MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL
    confidence_level: Literal[CAPACITY_LOAD_CONFIDENCE_LEVEL] = (
        CAPACITY_LOAD_CONFIDENCE_LEVEL
    )
    max_error_rate_cluster_range: Literal[CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE] = (
        CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE
    )
    max_throughput_cv: float = Field(
        gt=0, le=MAX_CAPACITY_STABILITY_CV, allow_inf_nan=False
    )
    max_latency_p95_cv: float = Field(
        gt=0, le=MAX_CAPACITY_STABILITY_CV, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_load_ladder(self) -> CapacityLoadProtocol:
        if self.concurrency_levels != capacity_concurrency_levels(
            self.concurrency_levels[-1]
        ):
            raise ValueError(
                "capacity concurrency_levels must use the geometric platform ladder"
            )
        if self.minimum_measurement_clusters_per_level > self.repetitions_per_level:
            raise ValueError(
                "capacity repetitions_per_level must cover the minimum independent clusters"
            )
        return self


class RunManifest(_StrictModel):
    """Fixed public worker manifest shared with the Dashboard backend."""

    schema_version: Literal[SCHEMA_VERSION]
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str
    name: str
    description: str
    mode: Literal["replay", "live"]
    target: _target_contracts.EvaluationTarget
    change_profile: ChangeProfile
    gate_contract_version: Literal[GATE_CONTRACT_VERSION]
    suite_ids: tuple[str, ...] = Field(min_length=1)
    suite_revisions: dict[str, str] = Field(min_length=1)
    suite_executors: dict[str, str] = Field(min_length=1)
    track_ids: tuple[str, ...] = Field(min_length=1)
    sample_limit: int = Field(gt=0, le=100000)
    concurrency: int = Field(ge=1, le=128)
    capacity_slo: CapacitySLO | None = None
    capacity_load_protocol: CapacityLoadProtocol | None = None
    seed: int = Field(ge=0, le=2**32 - 1)
    baseline_run_id: str | None = None
    created_at: datetime
    code_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    redaction_policy: str = Field(min_length=1, max_length=160)

    _run_id = field_validator("run_id")(validate_canonical_uuid)
    _name = field_validator("name")(validate_run_name)
    _description = field_validator("description")(validate_run_description)

    @classmethod
    def from_semantic_fields(cls, **fields: object) -> Self:
        if "manifest_digest" in fields:
            raise ValueError("manifest_digest is derived from semantic fields")
        semantic_fields = {"schema_version": SCHEMA_VERSION, **fields}
        return cls.model_validate(seal_manifest_fields(semantic_fields))

    def with_semantic_updates(self, **updates: object) -> Self:
        if "manifest_digest" in updates:
            raise ValueError("manifest_digest is derived from semantic fields")
        fields = self.model_dump(mode="python", exclude={"manifest_digest"})
        fields.update(updates)
        return type(self).model_validate(seal_manifest_fields(fields))

    @field_validator("baseline_run_id")
    @classmethod
    def validate_baseline_run_id(cls, value: str | None) -> str | None:
        return validate_canonical_uuid(value) if value is not None else None

    @field_validator("track_ids")
    @classmethod
    def validate_tracks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(TRACK_IDS))
        if unknown:
            raise ValueError(f"unknown track ids: {', '.join(unknown)}")
        if len(value) != len(set(value)):
            raise ValueError("track ids must be unique")
        canonical = tuple(track_id for track_id in TRACK_IDS if track_id in value)
        if value != canonical:
            raise ValueError("track ids must use canonical catalog order")
        return value

    @field_validator("suite_ids")
    @classmethod
    def validate_suites(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("suite ids must be unique")
        for suite_id in value:
            validate_portable_id(suite_id)
        builtin = set(value).intersection(BUILTIN_SUITE_IDS)
        if builtin:
            if len(builtin) != len(value):
                raise ValueError("builtin and installed suite ids cannot be mixed")
            canonical = tuple(
                suite_id for suite_id in BUILTIN_SUITE_IDS if suite_id in value
            )
            if value != canonical:
                raise ValueError("builtin suite ids must use canonical catalog order")
        elif value != tuple(sorted(value)):
            raise ValueError("installed suite ids must use lexical canonical order")
        return value

    @model_validator(mode="after")
    def validate_frozen_manifest_structure(self) -> RunManifest:
        self._validate_suite_bindings()
        self._validate_mixture_binding()
        self._validate_capacity_binding()
        require_manifest_digest(self)
        return self

    def _validate_suite_bindings(self) -> None:
        if set(self.suite_revisions) != set(self.suite_ids) or any(
            not is_valid_suite_revision(revision)
            for revision in self.suite_revisions.values()
        ):
            raise ValueError(
                "suite_revisions must contain one non-empty immutable identity per suite_id"
            )
        if set(self.suite_executors) != set(self.suite_ids) or any(
            not executor_id.strip()
            or len(executor_id) > _MAX_SUITE_EXECUTOR_ID_LENGTH
            or not is_portable_id(executor_id)
            for executor_id in self.suite_executors.values()
        ):
            raise ValueError(
                "suite_executors must contain one portable identity per suite_id"
            )
        if len(set(self.suite_executors.values())) != 1:
            raise ValueError("one evaluation run cannot mix executor implementations")

    def _validate_mixture_binding(self) -> None:
        # The wire contract can validate frozen mixture facts, but executor
        # capability belongs to the registry that resolves suite_executors.
        mixture = self.target.mixture
        if mixture is not None:
            if self.policy_snapshot_digest != mixture.recipe_digest:
                raise ValueError(
                    "policy_snapshot_digest must equal the selected mixture recipe digest"
                )
            if len(mixture.model_arms) < _MINIMUM_MIXTURE_ARM_COUNT:
                raise ValueError(
                    "Mixture-of-Models evaluation requires at least two arms"
                )
        elif self.mode == "live":
            raise ValueError("live evaluation requires a frozen target mixture")

    def _validate_capacity_binding(self) -> None:
        capacity_selected = "capacity" in self.track_ids
        if capacity_selected and self.mode == "live":
            if self.concurrency < _MINIMUM_LIVE_CAPACITY_CONCURRENCY:
                raise ValueError(
                    "live capacity track requires concurrency of at least 2"
                )
            if self.capacity_slo is None:
                raise ValueError("live capacity track requires capacity_slo")
            if self.capacity_load_protocol is None:
                raise ValueError("live capacity track requires capacity_load_protocol")
            if self.capacity_slo.required_concurrency > self.concurrency:
                raise ValueError(
                    "capacity_slo required_concurrency cannot exceed run concurrency"
                )
            if self.capacity_load_protocol.concurrency_levels[-1] != self.concurrency:
                raise ValueError(
                    "capacity_load_protocol must terminate at run concurrency"
                )
        elif self.capacity_slo is not None or self.capacity_load_protocol is not None:
            raise ValueError(
                "capacity_slo and capacity_load_protocol are valid only for a live capacity track"
            )


class ExecutorMetadata(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    track_id: str
    executor_id: str
    mode: Literal["replay", "live"]

    @field_validator("track_id")
    @classmethod
    def validate_track(cls, value: str) -> str:
        if value not in TRACK_IDS:
            raise ValueError("unknown track id")
        return value


class ResolvedRunSnapshot(_StrictModel):
    """Content-addressed run graph resolved from the fixed public manifest."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workload: WorkloadSnapshot
    policy: _target_contracts.PolicySnapshot
    binding: _target_contracts.BindingSnapshot
    pool: _target_contracts.PoolDefinition
    arms: tuple[_target_contracts.EvaluationTargetArm, ...]
    environment: _target_contracts.RunEnvironment
    fixture_ref: _ArtifactRef | None = None
    discovered_entrypoints: tuple[str, ...] = ()
    executors: tuple[ExecutorMetadata, ...]
