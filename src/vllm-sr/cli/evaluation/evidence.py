"""Strict normalized evidence contracts used by all executors."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contracts import StrictModel, _validate_id


class RouteEvidence(StrictModel):
    selected_model: str | None = None
    selection_status: str = "unknown"
    success: bool
    latency_ms: float | None = Field(default=None, ge=0)
    fallback: bool = False


class ArmEvidence(StrictModel):
    arm_id: str
    success: bool
    quality: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    runtime_cost: float | None = Field(default=None, ge=0)

    _arm_id = field_validator("arm_id")(_validate_id)


class TrajectoryEvidence(StrictModel):
    success: bool
    task_score: float | None = Field(default=None, ge=0, le=1)
    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    invalid_tool_calls: int = Field(default=0, ge=0)


class MultimodalEvidence(StrictModel):
    supported: bool
    quality: float | None = Field(default=None, ge=0, le=1)
    privacy_violations: int = Field(default=0, ge=0)


class PreferenceEvidence(StrictModel):
    chosen_arm_id: str
    preferred_arm_id: str
    reward: float | None = None
    behavior_propensity: float | None = Field(default=None, gt=0, le=1)


class SafetyEvidence(StrictModel):
    violations: int = Field(ge=0)
    should_block: bool
    blocked: bool


class CapacityEvidence(StrictModel):
    concurrency: int = Field(ge=1)
    success: bool
    latency_ms: float = Field(ge=0)
    throughput_rps: float = Field(ge=0)
    capacity_tco: float | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    energy_kwh: float | None = Field(default=None, ge=0)


class FixtureCaseEvidence(StrictModel):
    case_id: str
    route: RouteEvidence
    arms: tuple[ArmEvidence, ...]
    trajectory: TrajectoryEvidence
    multimodal: MultimodalEvidence
    preference: PreferenceEvidence
    safety: SafetyEvidence
    capacity: CapacityEvidence

    _case_id = field_validator("case_id")(_validate_id)


class ReplayFixture(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    cases: tuple[FixtureCaseEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> ReplayFixture:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("fixture case ids must be unique")
        return self


class RoutingTraceNode(StrictModel):
    node_type: str = Field(min_length=1, max_length=64)
    signal_type: str | None = Field(default=None, max_length=128)
    signal_name: str | None = Field(default=None, max_length=128)
    label: str | None = Field(default=None, max_length=128)
    state: str | None = Field(default=None, max_length=32)
    matched: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    has_signal_error: bool = False
    confidence_scored: bool = False
    children: tuple[RoutingTraceNode, ...] = ()


class RoutingDecisionTrace(StrictModel):
    decision_name: str = Field(min_length=1, max_length=128)
    state: str | None = Field(default=None, max_length=32)
    matched: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    on_unknown: str | None = Field(default=None, max_length=32)
    root_trace: RoutingTraceNode | None = None


class RoutingSignalDiagnostic(StrictModel):
    key: str = Field(min_length=1, max_length=160)
    confidence: float | None = None
    value: float | None = None
    has_error: bool = False


class RoutingDiagnostic(StrictModel):
    """Content-minimized route trace; raw prompts and free-form errors are excluded."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    recipe: str | None = Field(default=None, max_length=160)
    decision_name: str | None = Field(default=None, max_length=160)
    algorithm: str | None = Field(default=None, max_length=160)
    plugins: tuple[str, ...] = ()
    recommended_models: tuple[str, ...] = ()
    selected_model: str | None = Field(default=None, max_length=256)
    selection_status: str | None = Field(default=None, max_length=64)
    selection_method: str | None = Field(default=None, max_length=128)
    routing_decision: str | None = Field(default=None, max_length=160)
    traces: tuple[RoutingDecisionTrace, ...] = ()
    signals: tuple[RoutingSignalDiagnostic, ...] = ()
    applied_unknown_policies: tuple[tuple[str, str], ...] = ()

    _case_id = field_validator("case_id")(_validate_id)


class ExecutionRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    track_id: str
    case_id: str
    attempt_id: str
    status: Literal["succeeded", "failed", "unavailable"]
    arm_id: str | None = None
    selected_arm_id: str | None = None
    selection_status: str | None = None
    selection_method: str | None = None
    recipe: str | None = None
    decision_name: str | None = None
    algorithm: str | None = None
    trace_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    success: bool | None = None
    quality: float | None = Field(default=None, ge=0, le=1)
    fallback: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    runtime_cost: float | None = Field(default=None, ge=0)
    evaluation_cost: float | None = Field(default=None, ge=0)
    capacity_tco: float | None = Field(default=None, ge=0)
    trajectory_steps: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    invalid_tool_calls: int | None = Field(default=None, ge=0)
    modality: str | None = None
    privacy_violations: int | None = Field(default=None, ge=0)
    preference_match: bool | None = None
    behavior_propensity: float | None = Field(default=None, gt=0, le=1)
    safety_violations: int | None = Field(default=None, ge=0)
    should_block: bool | None = None
    blocked: bool | None = None
    concurrency: int | None = Field(default=None, ge=1)
    throughput_rps: float | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    energy_kwh: float | None = Field(default=None, ge=0)
    load_elapsed_seconds: float | None = Field(default=None, ge=0)
    grader: str | None = None
    evidence_kind: str | None = None
    error: str | None = None

    _id = field_validator("id", "case_id", "attempt_id")(_validate_id)

    @field_validator("track_id")
    @classmethod
    def validate_track(cls, value: str) -> str:
        if value not in TRACK_IDS:
            raise ValueError("unknown track id")
        return value
