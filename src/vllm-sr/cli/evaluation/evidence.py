"""Strict normalized evidence contracts used by all executors."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.agent_task_evidence import (
    AgentTaskMethodEvidence,
    binary64_equal,
)
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id
from cli.evaluation.evidence_source_ids import (
    DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
    LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.method_evidence import (
    HardPolicyMethodEvidence,
    OnlinePreferenceMethodEvidence,
    ProductionExperimentMethodEvidence,
    RecoveryMethodEvidence,
    RobustnessMethodEvidence,
)


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
    truncated: bool = False
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
    decision_error: str | None = Field(default=None, max_length=200)

    _case_id = field_validator("case_id")(_validate_id)


class ExecutionRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    track_id: str
    case_id: str
    attempt_id: str
    status: Literal["succeeded", "failed", "unavailable"]
    arm_id: str | None = None
    method_id: str | None = None
    action_id: str | None = None
    budget_tokens: int | None = Field(default=None, gt=0)
    slice_ids: tuple[str, ...] = ()
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
    robustness: RobustnessMethodEvidence | None = None
    agent_task: AgentTaskMethodEvidence | None = None
    recovery: RecoveryMethodEvidence | None = None
    production_experiment: ProductionExperimentMethodEvidence | None = None
    online_preference: OnlinePreferenceMethodEvidence | None = None
    hard_policy: HardPolicyMethodEvidence | None = None
    safety_violations: int | None = Field(default=None, ge=0)
    should_block: bool | None = None
    blocked: bool | None = None
    concurrency: int | None = Field(default=None, ge=1)
    throughput_rps: float | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    energy_kwh: float | None = Field(default=None, ge=0)
    load_elapsed_seconds: float | None = Field(default=None, ge=0)
    load_phase: Literal["warmup", "measurement"] | None = None
    load_repetition: int | None = Field(default=None, ge=0)
    load_request_index: int | None = Field(default=None, ge=0)
    grader: str | None = None
    evidence_kind: str | None = None
    broker_receipt: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    error: str | None = None

    _id = field_validator("id", "case_id", "attempt_id")(_validate_id)

    @model_validator(mode="after")
    def validate_method_coordinates(self) -> ExecutionRecord:
        if self.method_id is None:
            if (
                self.action_id is not None
                or self.budget_tokens is not None
                or self.slice_ids
            ):
                raise ValueError("method coordinates require a method identity")
            return self
        _validate_id(self.method_id)
        if self.method_id == "r2.compound-model-budget.v2" and (
            self.track_id != "model_pool"
            or self.action_id is None
            or self.budget_tokens is None
            or self.quality is None
            or not self.slice_ids
        ):
            raise ValueError("R2 evidence requires complete compound coordinates")
        if len(self.slice_ids) != len(set(self.slice_ids)):
            raise ValueError("method slice ids must be unique")
        for slice_id in self.slice_ids:
            _validate_id(slice_id)
        return self

    @field_validator("track_id")
    @classmethod
    def validate_track(cls, value: str) -> str:
        if value not in TRACK_IDS:
            raise ValueError("unknown track id")
        return value

    @model_validator(mode="after")
    def validate_normalized_replay_diagnostics(self) -> ExecutionRecord:
        """Fail closed on the typed facts used by source-bound diagnostics."""

        self._validate_capacity_load_record()
        kind = self.evidence_kind or ""
        executor_id, separator, ceiling = kind.rpartition(";ceiling=")
        if not separator:
            return self._validate_live_method_binding()
        _validate_id(executor_id)
        if ceiling not in {"E0", "E1", "E2", "E3", "E4", "E5"}:
            raise ValueError("normalized replay evidence_kind has an invalid ceiling")
        if self.track_id not in {"routing", "agentic", "preference"}:
            return self
        if self.status == "unavailable":
            if not self.error or not self.error.strip():
                raise ValueError(
                    "unavailable normalized replay evidence requires a reason"
                )
            if self._normalized_diagnostic_values():
                raise ValueError(
                    "unavailable normalized replay evidence cannot carry diagnostic values"
                )
            return self
        if self.error is not None:
            raise ValueError(
                "evaluated normalized replay evidence cannot carry an error"
            )
        if self.track_id == "routing":
            self._validate_normalized_routing(executor_id)
        elif self.track_id == "agentic":
            self._validate_normalized_agentic()
        else:
            self._validate_normalized_preference(ceiling)
        return self

    def _validate_capacity_load_record(self) -> None:
        load_values = (
            self.load_phase,
            self.load_repetition,
            self.load_request_index,
        )
        if self.track_id != "capacity":
            if any(value is not None for value in load_values):
                raise ValueError("load coordinates are valid only for capacity rows")
            return
        if any(value is not None for value in load_values):
            if any(value is None for value in load_values):
                raise ValueError("capacity load coordinates must be complete")
            if self.concurrency is None or self.throughput_rps is None:
                raise ValueError(
                    "capacity load rows require concurrency and throughput"
                )
            if self.load_phase == "warmup" and self.load_repetition != 0:
                raise ValueError("capacity warmup rows require repetition zero")
            if self.load_phase == "measurement" and self.load_repetition == 0:
                raise ValueError(
                    "capacity measurement rows require a positive repetition"
                )

    def _normalized_diagnostic_values(self) -> tuple[object, ...]:
        if self.track_id == "routing":
            return tuple(
                value
                for value in (
                    self.selected_arm_id,
                    self.selection_status,
                    self.selection_method,
                    self.success,
                    self.quality,
                    self.fallback,
                    self.latency_ms,
                    self.robustness,
                )
                if value is not None
            )
        if self.track_id == "agentic":
            return tuple(
                value
                for value in (
                    self.selected_arm_id,
                    self.success,
                    self.quality,
                    self.trajectory_steps,
                    self.tool_calls,
                    self.invalid_tool_calls,
                    self.privacy_violations,
                    self.agent_task,
                    self.recovery,
                )
                if value is not None
            )
        return tuple(
            value
            for value in (
                self.selected_arm_id,
                self.success,
                self.quality,
                self.preference_match,
                self.behavior_propensity,
                self.production_experiment,
                self.online_preference,
            )
            if value is not None
        )

    def _validate_success_status(self) -> None:
        if self.success is None:
            raise ValueError("evaluated normalized replay evidence requires success")
        expected = "succeeded" if self.success else "failed"
        if self.status != expected:
            raise ValueError("normalized replay status must agree with success")

    def _validate_normalized_routing(self, executor_id: str) -> None:
        self._validate_success_status()
        if (
            self.selection_status is None
            or self.selection_method != executor_id
            or self.fallback is None
        ):
            raise ValueError("normalized routing evidence lacks typed decision facts")
        if self.fallback != (self.selection_status == "fallback"):
            raise ValueError("normalized routing fallback facts disagree")
        if (
            self.selection_status in {"selected", "fallback"}
            and not self.selected_arm_id
        ):
            raise ValueError("normalized selected routing evidence lacks an action")
        if self.robustness is not None:
            if not self.selected_arm_id:
                raise ValueError("robustness evidence requires a perturbed action")
            if self.robustness.source_case_id == self.case_id:
                raise ValueError("robustness source and perturbed cases must differ")
            if self.robustness.target_case_id != self.case_id:
                raise ValueError("robustness target must bind the perturbed record")

    def _validate_normalized_agentic(self) -> None:
        self._validate_success_status()
        if any(
            value is None
            for value in (
                self.quality,
                self.trajectory_steps,
                self.tool_calls,
                self.invalid_tool_calls,
                self.privacy_violations,
            )
        ):
            raise ValueError("normalized agentic evidence lacks typed trajectory facts")
        if self.invalid_tool_calls > self.tool_calls:
            raise ValueError("invalid tool calls cannot exceed tool calls")
        if self.tool_calls > self.trajectory_steps:
            raise ValueError("tool-call steps cannot exceed trajectory steps")
        if self.agent_task is not None or self.recovery is not None:
            raise ValueError(
                "live agent-task and fault-recovery evidence cannot come from normalized replay"
            )

    def _validate_normalized_preference(self, ceiling: str) -> None:
        if self.status != "succeeded" or self.success is not True:
            raise ValueError("normalized preference observations must be successful")
        if (
            not self.selected_arm_id
            or self.quality is None
            or self.preference_match is None
        ):
            raise ValueError("normalized preference evidence lacks typed outcome facts")
        if ceiling == "E5" and self.behavior_propensity is None:
            raise ValueError("E5 normalized preference evidence requires propensity")
        if self.production_experiment is not None or self.online_preference is not None:
            raise ValueError("production experiment evidence cannot come from replay")

    def _validate_live_method_binding(self) -> ExecutionRecord:
        if self.agent_task is not None and self.recovery is not None:
            raise ValueError(
                "agent-task outcomes and fault-recovery continuity are independent methods"
            )
        self._validate_live_robustness_binding()
        self._validate_production_experiment_binding()
        self._validate_agent_task_binding()
        self._validate_hard_policy_binding()
        self._validate_recovery_binding()
        return self

    def _validate_live_robustness_binding(self) -> None:
        if (
            self.robustness is not None
            and self.robustness.method_id == "declared-shift.server-live.v1"
        ):
            method = self.robustness
            if (
                self.track_id != "routing"
                or self.status != "succeeded"
                or self.success is not True
                or not self.selected_arm_id
                or not self.broker_receipt
                or method.target_case_id != self.case_id
                or self.evidence_kind != DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID
            ):
                raise ValueError(
                    "server-live declared-shift evidence lacks its brokered routing binding"
                )

    def _validate_production_experiment_binding(self) -> None:
        if self.production_experiment is not None:
            experiment = self.production_experiment
            if self.track_id != "preference" or self.status != "succeeded":
                raise ValueError(
                    "production experiment evidence requires a successful preference row"
                )
            if self.behavior_propensity != experiment.behavior_propensity:
                raise ValueError("record propensity must bind production assignment")
            if self.selected_arm_id != experiment.assigned_policy_arm_id:
                raise ValueError("record policy arm must bind production assignment")
            if (
                self.online_preference is not None
                and self.online_preference.experiment != experiment
            ):
                raise ValueError(
                    "online preference outcome must bind its production experiment"
                )
        elif self.online_preference is not None:
            raise ValueError("online outcome requires production experiment evidence")

    def _validate_agent_task_binding(self) -> None:
        if self.agent_task is not None:
            method = self.agent_task
            expected_status = "succeeded" if method.task_success else "failed"
            if (
                self.track_id != "agentic"
                or self.status != expected_status
                or self.success is not method.task_success
                or self.selected_arm_id != method.selected_arm_id
                or self.quality != method.task_score
                or self.trajectory_steps != method.trajectory_steps
                or self.tool_calls != method.tool_call_count
                or self.invalid_tool_calls != method.invalid_tool_call_count
                or self.privacy_violations != method.privacy_exposure_count
                or self.input_tokens != method.input_tokens
                or self.output_tokens != method.output_tokens
                or self.runtime_cost is None
                or not binary64_equal(self.runtime_cost, method.runtime_cost_usd)
                or self.evaluation_cost is None
                or not binary64_equal(self.evaluation_cost, method.evaluation_cost_usd)
                or self.grader != method.grader_id
                or self.evidence_kind != LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID
                or not self.broker_receipt
            ):
                raise ValueError(
                    "agentic row must bind its exact provider-observed task attempt"
                )

    def _validate_hard_policy_binding(self) -> None:
        if self.hard_policy is not None:
            method = self.hard_policy
            if self.track_id != "safety" or self.status != "succeeded":
                raise ValueError(
                    "hard-policy evidence requires a successful safety row"
                )
            if (
                self.should_block != method.should_block
                or self.blocked != method.blocked
                or self.safety_violations != method.violations
            ):
                raise ValueError("safety row must bind hard-policy method evidence")

    def _validate_recovery_binding(self) -> None:
        if self.recovery is not None:
            method = self.recovery
            if self.track_id != "agentic" or self.status == "unavailable":
                raise ValueError(
                    "fault-recovery evidence requires an evaluated agentic row"
                )
            retry_amplification = (method.treatment_retry_count + 1) / (
                method.baseline_retry_count + 1
            )
            passed = (
                method.injection_observed
                and method.recovered
                and method.state_preserved
                and method.treatment_terminal_success
                and method.duplicate_side_effect_count == 0
                and method.treatment_recovery_latency_ms
                <= method.maximum_recovery_latency_ms
                and retry_amplification <= method.maximum_retry_amplification
            )
            if self.success is not passed or self.status != (
                "succeeded" if passed else "failed"
            ):
                raise ValueError(
                    "agentic row must bind live fault-recovery method evidence"
                )
