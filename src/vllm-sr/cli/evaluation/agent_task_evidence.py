"""Strict evidence for sealed, provider-observed agent task trajectories."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id

AGENT_TASK_LEDGER_VERSION = "evaluation-agent-task-ledger.v1"
AGENT_TASK_ATTEMPT_VERSION = "evaluation-agent-task-attempt.v1"
AGENT_TASK_METHOD_ID = "live-agent-task.v1"
AGENT_TASK_EXECUTION_SEMANTICS = "provider-observed-explicit-tool-policy"
AGENT_TASK_BENCHMARK_PARITY_CLAIM = "none"
MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT = 20
MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK = 2
MAXIMUM_AGENT_TASK_TOOL_CALLS_PER_ATTEMPT = 128


def binary64_equal(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if left == right:
        return True
    spacing = max(math.ulp(left), math.ulp(right))
    return abs(left - right) <= 8 * spacing


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


class AgentTaskToolCallEvidence(StrictModel):
    """One observed tool call; this contract never asks the worker to execute it."""

    sequence: int = Field(ge=1, strict=True)
    tool_call_id: str
    tool_name: str
    arguments_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    outcome: Literal["executed", "rejected_invalid"]
    result_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    execution_receipt_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    cost_usd: float = Field(ge=0, allow_inf_nan=False)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    _ids = field_validator("tool_call_id", "tool_name")(_validate_id)

    @model_validator(mode="after")
    def validate_execution_receipt(self) -> AgentTaskToolCallEvidence:
        if self.outcome == "executed":
            if (
                self.result_digest is None
                or self.execution_receipt_digest is None
                or self.started_at is None
                or self.completed_at is None
                or not _aware(self.started_at)
                or not _aware(self.completed_at)
                or self.completed_at < self.started_at
            ):
                raise ValueError("executed tool call requires a real ordered receipt")
        elif (
            self.result_digest is not None
            or self.execution_receipt_digest is not None
            or self.started_at is not None
            or self.completed_at is not None
            or self.cost_usd != 0
        ):
            raise ValueError("invalid tool call cannot claim execution")
        return self


class AgentTaskToolPolicy(StrictModel):
    """Frozen per-task policy distinguishing required tools from pure reasoning."""

    requires_tools: bool
    expected_tools: tuple[str, ...]

    _tool_ids = field_validator("expected_tools")(
        lambda values: tuple(_validate_id(value) for value in values)
    )

    @model_validator(mode="after")
    def validate_policy(self) -> AgentTaskToolPolicy:
        if tuple(sorted(set(self.expected_tools))) != self.expected_tools:
            raise ValueError("expected tools must be unique and sorted")
        if self.requires_tools != bool(self.expected_tools):
            raise ValueError("requires_tools contradicts expected_tools")
        return self


class AgentTaskMethodEvidence(StrictModel):
    """One repeated task attempt bound to a sealed Mixture and real tool receipts."""

    contract_version: Literal[AGENT_TASK_ATTEMPT_VERSION]
    method_id: Literal[AGENT_TASK_METHOD_ID]
    ledger_id: str
    source_id: str
    suite_id: str
    suite_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    task_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    benchmark_parity_claim: Literal[AGENT_TASK_BENCHMARK_PARITY_CLAIM]
    execution_semantics: Literal[AGENT_TASK_EXECUTION_SEMANTICS]
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_total_attempt_count: int = Field(ge=1, strict=True)
    ledger_total_distinct_task_count: int = Field(ge=1, strict=True)
    minimum_distinct_task_count: int = Field(
        ge=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT, strict=True
    )
    minimum_attempts_per_task: int = Field(
        ge=MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK, strict=True
    )
    task_id: str
    task_spec_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    tool_policy: AgentTaskToolPolicy
    attempt_id: str
    repetition_id: str
    trajectory_id: str
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    selected_arm_id: str
    task_success: bool
    task_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    success_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    grader_id: str
    grader_revision_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    grading_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    privacy_audit_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    execution_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    trajectory_steps: int = Field(ge=1, strict=True)
    tool_call_count: int = Field(
        ge=0, le=MAXIMUM_AGENT_TASK_TOOL_CALLS_PER_ATTEMPT, strict=True
    )
    invalid_tool_call_count: int = Field(ge=0, strict=True)
    privacy_exposure_count: int = Field(ge=0, strict=True)
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    model_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    tool_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    evaluation_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    total_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    started_at: datetime
    completed_at: datetime
    graded_at: datetime
    privacy_audited_at: datetime
    tool_calls: tuple[AgentTaskToolCallEvidence, ...] = Field(
        max_length=MAXIMUM_AGENT_TASK_TOOL_CALLS_PER_ATTEMPT
    )

    _ids = field_validator(
        "ledger_id",
        "source_id",
        "suite_id",
        "target_id",
        "task_id",
        "attempt_id",
        "repetition_id",
        "trajectory_id",
        "selected_arm_id",
        "grader_id",
    )(_validate_id)

    @model_validator(mode="after")
    def validate_trajectory(self) -> AgentTaskMethodEvidence:
        self._validate_outcome_and_counts()
        self._validate_tool_policy()
        self._validate_tool_observations()
        self._validate_costs_and_receipts()
        self._validate_timestamps()
        return self

    def _validate_outcome_and_counts(self) -> None:
        if self.task_success != (self.task_score >= self.success_threshold):
            raise ValueError("task outcome contradicts its success threshold")
        if self.invalid_tool_call_count > self.tool_call_count:
            raise ValueError("invalid tool calls cannot exceed all tool calls")
        if self.tool_call_count > self.trajectory_steps:
            raise ValueError("tool calls cannot exceed trajectory steps")
        if len(self.tool_calls) != self.tool_call_count:
            raise ValueError("tool-call count does not match observed tool calls")

    def _validate_tool_policy(self) -> None:
        if not self.tool_policy.requires_tools:
            if self.tool_calls:
                raise ValueError("pure-reasoning agent task cannot contain tool calls")
            return
        expected = set(self.tool_policy.expected_tools)
        executed = [call for call in self.tool_calls if call.outcome == "executed"]
        if any(call.tool_name not in expected for call in executed):
            raise ValueError("agent task executed outside its expected-tool policy")
        if not executed:
            raise ValueError(
                "tool-required agent-task attempt lacks a provider-executed receipt"
            )

    def _validate_tool_observations(self) -> None:
        if [call.sequence for call in self.tool_calls] != list(
            range(1, len(self.tool_calls) + 1)
        ):
            raise ValueError("tool-call sequence must be dense and ordered")
        tool_ids = [call.tool_call_id for call in self.tool_calls]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("tool-call identities must be unique")
        invalid = sum(call.outcome == "rejected_invalid" for call in self.tool_calls)
        if invalid != self.invalid_tool_call_count:
            raise ValueError("invalid-tool counter differs from observations")

    def _validate_costs_and_receipts(self) -> None:
        tool_cost = sum(call.cost_usd for call in self.tool_calls)
        if not binary64_equal(tool_cost, self.tool_cost_usd) or not binary64_equal(
            self.model_cost_usd + self.tool_cost_usd + self.evaluation_cost_usd,
            self.total_cost_usd,
        ):
            raise ValueError("complete cost does not match trajectory components")
        receipts = [
            self.execution_receipt_digest,
            self.grading_receipt_digest,
            self.privacy_audit_receipt_digest,
            *(
                call.execution_receipt_digest
                for call in self.tool_calls
                if call.execution_receipt_digest is not None
            ),
        ]
        if len(receipts) != len(set(receipts)):
            raise ValueError("attempt execution and grading receipts must be unique")

    def _validate_timestamps(self) -> None:
        timestamps = (
            self.started_at,
            self.completed_at,
            self.graded_at,
            self.privacy_audited_at,
        )
        if any(not _aware(value) for value in timestamps) or not (
            self.started_at <= self.completed_at <= self.graded_at
            and self.completed_at <= self.privacy_audited_at
        ):
            raise ValueError("agent-task attempt timestamps must be aware and ordered")
        for call in self.tool_calls:
            if call.outcome == "executed" and (
                call.started_at is None
                or call.completed_at is None
                or call.started_at < self.started_at
                or call.completed_at > self.completed_at
            ):
                raise ValueError("tool execution is outside the task attempt window")

    @property
    def runtime_cost_usd(self) -> float:
        return self.model_cost_usd + self.tool_cost_usd

    @property
    def receipts(self) -> tuple[str, ...]:
        return (
            self.execution_receipt_digest,
            self.grading_receipt_digest,
            self.privacy_audit_receipt_digest,
            *(
                call.execution_receipt_digest
                for call in self.tool_calls
                if call.execution_receipt_digest is not None
            ),
        )
