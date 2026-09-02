"""Canonical IR shared by pinned external benchmark adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from cli.evaluation.benchmark_registry import ADAPTER_CONTRACT_VERSION
from cli.evaluation.canonical import digest_value
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import ArtifactRef, StrictModel
from cli.evaluation.reporting import TrackID
from cli.evaluation.suite_qualification import (
    NORMALIZED_REPLAY_EXECUTOR_DIGEST,
    BenchmarkQualificationReceipt,
)

SUITE_CONTRACT_VERSION = "evaluation-suite.v1"


def _subject_field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def qualification_manifest_subject_digest(value: object) -> str:
    """Digest manifest semantics that exist before qualification is issued."""

    return digest_value(
        {
            "schema_version": SUITE_CONTRACT_VERSION,
            "id": _subject_field(value, "id"),
            "name": _subject_field(value, "name"),
            "adapter_id": _subject_field(value, "adapter_id"),
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "source_receipt": _subject_field(value, "source_receipt"),
            "decision_unit": _subject_field(value, "decision_unit"),
            "action_space": _subject_field(value, "action_space"),
            "track_ids": _subject_field(value, "track_ids"),
            "split_protocol": _subject_field(value, "split_protocol"),
            "case_count": _subject_field(value, "case_count"),
            "arm_ids": _subject_field(value, "arm_ids"),
            "data_classification": _subject_field(value, "data_classification"),
            "redistribution": _subject_field(value, "redistribution"),
            "artifacts": _subject_field(value, "artifacts"),
            "limitations": _subject_field(value, "limitations"),
        }
    )


class NormalizedOutcome(StrictModel):
    """One observed outcome for an arm and optional budget/action variant."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    arm_id: str
    action_id: str | None = None
    budget_tokens: int | None = Field(default=None, gt=0)
    success: bool | None = None
    quality: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    runtime_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    grader_id: str | None = None
    grader_revision: str | None = None
    split: str
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NormalizedPreference(StrictModel):
    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    left_action_id: str
    right_action_id: str
    preference: Literal["left", "right", "tie", "skip"]
    chosen_action_id: str | None = None
    reward: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    segment_id: str | None = None
    assignment_id: str | None = None
    exposure_id: str | None = None
    exposure_probability: float | None = Field(default=None, gt=0, le=1)
    behavior_propensity: float | None = Field(default=None, gt=0, le=1)
    participant_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NormalizedTrajectoryStep(StrictModel):
    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    trajectory_id: str
    step_id: str
    sequence: int = Field(ge=0)
    case_id: str
    selected_action_id: str | None = None
    tool_name: str | None = None
    tool_call_valid: bool | None = None
    side_effect_id: str | None = None
    state_digest_before: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    state_digest_after: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    terminal: bool = False
    terminal_success: bool | None = None
    task_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    privacy_exposures: int | None = Field(default=None, ge=0)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NormalizedPerturbation(StrictModel):
    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    pair_id: str
    source_case_id: str
    perturbed_case_id: str
    relation: Literal["invariant", "expected_change"]
    expected_action_id: str | None = None
    slice_ids: tuple[str, ...] = ()
    native_pair_count: int = Field(ge=1, strict=True)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def expected_change_has_an_action(self) -> NormalizedPerturbation:
        if self.source_case_id == self.perturbed_case_id:
            raise ValueError("perturbation source and target cases must differ")
        if self.relation == "expected_change" and not self.expected_action_id:
            raise ValueError("expected-change perturbations require an action")
        if self.relation == "invariant" and self.expected_action_id is not None:
            raise ValueError(
                "invariant perturbations cannot override the source action"
            )
        if len(self.slice_ids) != len(set(self.slice_ids)) or any(
            not value or value.strip() != value for value in self.slice_ids
        ):
            raise ValueError("perturbation slice ids must be unique and non-empty")
        return self


class NormalizedFault(StrictModel):
    """Continuity source diagnostic; this is not a real runtime fault receipt."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    id: str
    trajectory_id: str
    sequence: int = Field(ge=0)
    kind: Literal[
        "timeout",
        "rate_limit",
        "server_error",
        "disconnect",
        "malformed_response",
        "state_loss",
    ]
    diagnostic_scope: Literal["provider_fallback_label_and_context_preservation"]
    method_id: Literal["continuity.labeled-failover.v1"]
    cohort_pair_id: str
    conversation_id: str
    system_role: Literal["treatment"]
    concurrency: int = Field(ge=1, strict=True)
    failure_turn: int = Field(ge=0, strict=True)
    native_repetition_count: Literal[1]
    repeated_seed_evidence: Literal[False]
    native_pair_count: int = Field(ge=1, strict=True)
    failover_labeled: bool
    context_preserved: bool
    experiment_manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    baseline_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    treatment_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    baseline_terminal_success: bool
    treatment_terminal_success: bool
    baseline_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    treatment_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def labeled_failover_is_bound(self) -> NormalizedFault:
        if not self.failover_labeled:
            raise ValueError("continuity diagnostic requires a source failover label")
        if self.context_preserved != self.treatment_terminal_success:
            raise ValueError(
                "context-preservation label must bind the treatment terminal outcome"
            )
        return self


class NormalizedDecision(StrictModel):
    """One observed router decision, separate from hidden grading labels."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    selected_arm_id: str | None = None
    selected_action_id: str | None = None
    selection_status: Literal[
        "selected", "abstained", "fallback", "error", "unavailable"
    ]
    selection_method: str | None = None
    success: bool
    fallback: bool = False
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def selected_decision_has_an_action(self) -> NormalizedDecision:
        if self.selection_status in {"selected", "fallback"} and not (
            self.selected_arm_id or self.selected_action_id
        ):
            raise ValueError("selected and fallback decisions require an action")
        if self.fallback != (self.selection_status == "fallback"):
            raise ValueError("fallback flag and selection status must agree")
        if self.selection_status in {"error", "unavailable"} and self.success:
            raise ValueError("error and unavailable decisions cannot be successful")
        return self


class NormalizedMultimodalObservation(StrictModel):
    """Observed capability and grading result for one non-text case."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    modality: Literal["image", "document", "audio", "video"]
    supported: bool
    quality: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    privacy_violations: int = Field(ge=0, strict=True)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NormalizedSafetyObservation(StrictModel):
    """Observed policy enforcement; expected blocking remains in grading cases."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    violations: int = Field(ge=0, strict=True)
    blocked: bool
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NormalizedCapacityObservation(StrictModel):
    """One bounded replayed load observation, never an inferred SLO claim."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    case_id: str
    concurrency: int = Field(ge=1, strict=True)
    success: bool
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    throughput_rps: float = Field(ge=0, allow_inf_nan=False)
    runtime_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    capacity_tco_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    energy_kwh: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_record_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class SuiteArtifactSet(StrictModel):
    visible_cases: ArtifactRef
    grading_cases: ArtifactRef
    outcomes: ArtifactRef | None = None
    decisions: ArtifactRef | None = None
    preferences: ArtifactRef | None = None
    trajectories: ArtifactRef | None = None
    perturbations: ArtifactRef | None = None
    faults: ArtifactRef | None = None
    multimodal_observations: ArtifactRef | None = None
    safety_observations: ArtifactRef | None = None
    capacity_observations: ArtifactRef | None = None
    media_manifest: ArtifactRef | None = None
    license_manifest: ArtifactRef

    @model_validator(mode="after")
    def labels_are_separate(self) -> SuiteArtifactSet:
        if self.visible_cases.digest == self.grading_cases.digest:
            raise ValueError(
                "visible and grading artifacts must be physically separate"
            )
        return self


class BenchmarkSourceReceipt(StrictModel):
    schema_version: Literal[ADAPTER_CONTRACT_VERSION] = ADAPTER_CONTRACT_VERSION
    adapter_id: str
    expected_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    observed_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_dataset_revision: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )
    observed_dataset_revision: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )
    source_clean: bool
    dataset_clean: bool | None = None
    verified: bool


class BenchmarkSuiteManifest(StrictModel):
    """Immutable installed suite; executable code is never embedded."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    id: str
    name: str
    adapter_id: str
    adapter_contract_version: Literal[ADAPTER_CONTRACT_VERSION] = (
        ADAPTER_CONTRACT_VERSION
    )
    source_receipt: BenchmarkSourceReceipt
    revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision_unit: str
    action_space: str
    track_ids: tuple[TrackID, ...]
    qualification_receipt: BenchmarkQualificationReceipt
    split_protocol: str
    case_count: int = Field(gt=0)
    arm_ids: tuple[str, ...] = ()
    data_classification: Literal["public", "internal", "confidential", "restricted"]
    redistribution: Literal["allowed", "metadata_only", "prohibited"]
    artifacts: SuiteArtifactSet
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def source_must_be_verified(self) -> BenchmarkSuiteManifest:
        if self.source_receipt.adapter_id != self.adapter_id:
            raise ValueError("source receipt belongs to a different adapter")
        if not self.source_receipt.verified:
            raise ValueError("suite source receipt must be verified")
        if len(self.track_ids) != len(set(self.track_ids)):
            raise ValueError("suite track ids must be unique")
        canonical_tracks = tuple(
            track_id for track_id in TRACK_IDS if track_id in self.track_ids
        )
        if self.track_ids != canonical_tracks:
            raise ValueError("suite track ids must use canonical catalog order")
        if len(self.arm_ids) != len(set(self.arm_ids)):
            raise ValueError("suite arm ids must be unique")
        if self.qualification_receipt.source_receipt_digest != digest_value(
            self.source_receipt
        ):
            raise ValueError("qualification receipt does not bind the source receipt")
        if self.qualification_receipt.artifact_set_digest != digest_value(
            self.artifacts
        ):
            raise ValueError("qualification receipt does not bind the artifact set")
        if (
            self.qualification_receipt.manifest_subject_digest
            != qualification_manifest_subject_digest(self)
        ):
            raise ValueError("qualification receipt does not bind the manifest")
        if (
            self.qualification_receipt.executor_digest
            != NORMALIZED_REPLAY_EXECUTOR_DIGEST
        ):
            raise ValueError("qualification receipt does not bind the executor")
        return self
