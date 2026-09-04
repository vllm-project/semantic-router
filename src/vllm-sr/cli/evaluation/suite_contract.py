"""Canonical IR shared by pinned external benchmark adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from cli.evaluation.benchmark_registry import ADAPTER_CONTRACT_VERSION
from cli.evaluation.contracts import ArtifactRef, StrictModel
from cli.evaluation.reporting import EvidenceLevel, TrackID

SUITE_CONTRACT_VERSION = "evaluation-suite.v1"


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


class NormalizedFault(StrictModel):
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
        "labeled_proxy",
    ]
    expected_recovery: str
    is_real_fault: bool


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
    evidence_level_ceiling: EvidenceLevel
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
        if len(self.arm_ids) != len(set(self.arm_ids)):
            raise ValueError("suite arm ids must be unique")
        return self
