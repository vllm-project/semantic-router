"""Ingest a complete sealed agent-task ledger through the server broker.

This module only normalizes provider-observed trajectories. It never executes
tools and never claims parity with an upstream benchmark's native runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.agent_task_evidence import (
    AGENT_TASK_BENCHMARK_PARITY_CLAIM,
    AGENT_TASK_EXECUTION_SEMANTICS,
    AGENT_TASK_LEDGER_VERSION,
    MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK,
    MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
    AgentTaskMethodEvidence,
    binary64_equal,
)
from cli.evaluation.contract_primitives import Message, StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID
from cli.evaluation.http_client import EvaluationHTTPClient
from cli.evaluation.method_ledger_identity import (
    MethodMixtureBinding,
    method_mixture_binding,
    validate_method_ledger_freshness,
)
from cli.evaluation.target_contracts import ManifestMixture


def agent_task_set_digest(attempts: tuple[AgentTaskMethodEvidence, ...]) -> str:
    tasks: dict[str, tuple[str, bool, tuple[str, ...]]] = {}
    for attempt in attempts:
        identity = (
            attempt.task_spec_digest,
            attempt.tool_policy.requires_tools,
            attempt.tool_policy.expected_tools,
        )
        prior = tasks.setdefault(attempt.task_id, identity)
        if prior != identity:
            raise ValueError("agent-task identity maps to multiple task specs")
    rows: list[str] = []
    for task_id in sorted(tasks):
        spec_digest, requires_tools, expected_tools = tasks[task_id]
        policy = "requires_tools" if requires_tools else "pure_reasoning"
        rows.append(
            f"{task_id}\x00{spec_digest}\x00{policy}"
            + "".join(f"\x00{tool_name}" for tool_name in expected_tools)
            + "\n"
        )
    canonical = "agent-task-set.v1\n" + "".join(rows)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class AgentTaskLedger(StrictModel):
    contract_version: Literal[AGENT_TASK_LEDGER_VERSION]
    ledger_id: str
    source_id: str
    environment: Literal["production"]
    suite_id: str
    suite_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    task_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    benchmark_parity_claim: Literal[AGENT_TASK_BENCHMARK_PARITY_CLAIM]
    execution_semantics: Literal[AGENT_TASK_EXECUTION_SEMANTICS]
    provider_attestation_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture: MethodMixtureBinding
    ledger_total_attempt_count: int = Field(ge=1, strict=True)
    ledger_total_distinct_task_count: int = Field(ge=1, strict=True)
    minimum_distinct_task_count: int = Field(
        ge=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT, strict=True
    )
    minimum_attempts_per_task: int = Field(
        ge=MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK, strict=True
    )
    window_started_at: datetime
    window_ended_at: datetime
    sealed_at: datetime
    attempts: tuple[AgentTaskMethodEvidence, ...] = Field(min_length=1)

    _ids = field_validator("ledger_id", "source_id", "suite_id", "target_id")(
        _validate_id
    )

    @model_validator(mode="after")
    def validate_sealed_membership(self) -> AgentTaskLedger:
        _validate_agent_task_ledger_header(self)
        task_contracts, attempt_counts = _validate_agent_task_attempts(self)
        _validate_agent_task_cohort(self, task_contracts, attempt_counts)
        return self


AgentTaskContract = tuple[str, str, str, float, bool, tuple[str, ...]]


def _validate_agent_task_ledger_header(ledger: AgentTaskLedger) -> None:
    timestamps = (
        ledger.window_started_at,
        ledger.window_ended_at,
        ledger.sealed_at,
    )
    if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
        raise ValueError("agent-task ledger window is invalid")
    if not ledger.window_started_at < ledger.window_ended_at <= ledger.sealed_at:
        raise ValueError("agent-task ledger window is invalid")
    if ledger.ledger_total_attempt_count != len(ledger.attempts):
        raise ValueError("agent-task attempt count does not bind membership")
    if agent_task_set_digest(ledger.attempts) != ledger.task_set_digest:
        raise ValueError("agent-task set digest does not bind membership")


def _agent_task_attempt_binds_ledger(
    ledger: AgentTaskLedger, attempt: AgentTaskMethodEvidence
) -> bool:
    return (
        attempt.ledger_id == ledger.ledger_id
        and attempt.source_id == ledger.source_id
        and attempt.suite_id == ledger.suite_id
        and attempt.suite_revision == ledger.suite_revision
        and attempt.task_set_digest == ledger.task_set_digest
        and attempt.benchmark_parity_claim == ledger.benchmark_parity_claim
        and attempt.execution_semantics == ledger.execution_semantics
        and attempt.policy_snapshot_digest == ledger.policy_snapshot_digest
        and attempt.config_digest == ledger.config_digest
        and attempt.target_id == ledger.target_id
        and attempt.backend_topology_digest == ledger.backend_topology_digest
        and attempt.mixture_snapshot_digest == ledger.mixture.snapshot_digest
        and attempt.ledger_total_attempt_count == ledger.ledger_total_attempt_count
        and attempt.ledger_total_distinct_task_count
        == ledger.ledger_total_distinct_task_count
        and attempt.minimum_distinct_task_count == ledger.minimum_distinct_task_count
        and attempt.minimum_attempts_per_task == ledger.minimum_attempts_per_task
        and ledger.window_started_at
        <= attempt.started_at
        <= attempt.completed_at
        <= ledger.window_ended_at
        and attempt.graded_at <= ledger.window_ended_at
        and attempt.privacy_audited_at <= ledger.window_ended_at
    )


def _agent_task_contract(attempt: AgentTaskMethodEvidence) -> AgentTaskContract:
    return (
        attempt.task_spec_digest,
        attempt.grader_id,
        attempt.grader_revision_digest,
        attempt.success_threshold,
        attempt.tool_policy.requires_tools,
        attempt.tool_policy.expected_tools,
    )


def _validate_agent_task_attempts(
    ledger: AgentTaskLedger,
) -> tuple[dict[str, AgentTaskContract], dict[str, int]]:
    attempts: set[str] = set()
    trajectories: set[str] = set()
    repetitions: set[tuple[str, str]] = set()
    seeds: set[tuple[str, int]] = set()
    receipts: set[str] = {ledger.provider_attestation_digest}
    task_contracts: dict[str, AgentTaskContract] = {}
    attempt_counts: dict[str, int] = {}
    for attempt in ledger.attempts:
        if not _agent_task_attempt_binds_ledger(ledger, attempt):
            raise ValueError("agent-task attempt does not bind sealed ledger")
        repetition = (attempt.task_id, attempt.repetition_id)
        seed = (attempt.task_id, attempt.seed)
        if (
            attempt.attempt_id in attempts
            or attempt.trajectory_id in trajectories
            or repetition in repetitions
            or seed in seeds
        ):
            raise ValueError("agent-task attempt identities must be unique")
        attempts.add(attempt.attempt_id)
        trajectories.add(attempt.trajectory_id)
        repetitions.add(repetition)
        seeds.add(seed)
        contract = _agent_task_contract(attempt)
        prior = task_contracts.get(attempt.task_id)
        if prior is not None and (
            prior[:3] != contract[:3]
            or not binary64_equal(prior[3], contract[3])
            or prior[4:] != contract[4:]
        ):
            raise ValueError("task repetitions changed task or grader contract")
        task_contracts[attempt.task_id] = contract
        attempt_counts[attempt.task_id] = attempt_counts.get(attempt.task_id, 0) + 1
        for receipt in attempt.receipts:
            if receipt in receipts:
                raise ValueError("agent-task ledger receipts must be unique")
            receipts.add(receipt)
    return task_contracts, attempt_counts


def _validate_agent_task_cohort(
    ledger: AgentTaskLedger,
    task_contracts: dict[str, AgentTaskContract],
    attempt_counts: dict[str, int],
) -> None:
    if (
        len(task_contracts) != ledger.ledger_total_distinct_task_count
        or len(task_contracts) < ledger.minimum_distinct_task_count
        or any(
            count < ledger.minimum_attempts_per_task
            for count in attempt_counts.values()
        )
    ):
        raise ValueError("agent-task ledger lacks a decision-grade task cohort")


@dataclass(frozen=True)
class AgentTaskExecution:
    visible: VisibleCaseSet
    grading: GradingCaseSet
    records: list[ExecutionRecord]


def _case_id(ledger_id: str, attempt_id: str) -> str:
    digest = hashlib.sha256(f"{ledger_id}\x00{attempt_id}".encode()).hexdigest()[:24]
    return f"agent-task-{digest}"


def _fetch_agent_task_ledger(
    client: EvaluationHTTPClient,
    endpoint: str,
    *,
    policy_snapshot_digest: str,
    config_digest: str,
    target_id: str,
    backend_topology_digest: str,
    mixture: ManifestMixture,
) -> tuple[AgentTaskLedger, str]:
    result = client.get(
        endpoint,
        track_id="agentic",
        case_id="agent-task-ledger",
        attempt_id="ledger-fetch",
        broker_operation="agent-task.ledger",
    )
    if not result.success or result.payload is None or result.broker_receipt is None:
        raise ValueError("agent-task ledger could not be read")
    ledger = AgentTaskLedger.model_validate(result.payload)
    validate_method_ledger_freshness(ledger.sealed_at, result.fetched_at)
    if (
        ledger.policy_snapshot_digest != policy_snapshot_digest
        or ledger.config_digest != config_digest
        or ledger.target_id != target_id
        or ledger.backend_topology_digest != backend_topology_digest
        or ledger.mixture != method_mixture_binding(mixture)
    ):
        raise ValueError("agent-task ledger belongs to another Mixture snapshot")
    return ledger, result.broker_receipt


def execute_agent_task_ledger(
    client: EvaluationHTTPClient,
    endpoint: str,
    *,
    policy_snapshot_digest: str,
    config_digest: str,
    target_id: str,
    backend_topology_digest: str,
    mixture: ManifestMixture,
    sample_limit: int,
    seed: int,
) -> AgentTaskExecution:
    """Fetch and normalize one whole sealed task window; execute no tools."""

    ledger, broker_receipt = _fetch_agent_task_ledger(
        client,
        endpoint,
        policy_snapshot_digest=policy_snapshot_digest,
        config_digest=config_digest,
        target_id=target_id,
        backend_topology_digest=backend_topology_digest,
        mixture=mixture,
    )
    if sample_limit < len(ledger.attempts):
        raise ValueError(
            "sample_limit must cover every attempt in the sealed agent-task window"
        )
    arms = {arm.id: arm for arm in mixture.model_arms}
    for attempt in ledger.attempts:
        arm = arms.get(attempt.selected_arm_id)
        if arm is None:
            raise ValueError("agent-task ledger selected outside the frozen Mixture")
        expected_model_cost = (
            attempt.input_tokens * arm.input_cost_per_million_tokens_usd
            + attempt.output_tokens * arm.output_cost_per_million_tokens_usd
        ) / 1_000_000
        if not binary64_equal(attempt.model_cost_usd, expected_model_cost):
            raise ValueError(
                "agent-task model cost differs from frozen Mixture pricing"
            )

    selected = sorted(
        ledger.attempts,
        key=lambda attempt: (
            hashlib.sha256(
                f"{seed}\x00{ledger.ledger_id}\x00{attempt.attempt_id}".encode()
            ).digest(),
            attempt.attempt_id,
        ),
    )
    visible: list[CaseVisible] = []
    grading: list[CaseGrading] = []
    records: list[ExecutionRecord] = []
    for attempt in selected:
        case_id = _case_id(ledger.ledger_id, attempt.attempt_id)
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=("agentic",),
                messages=(
                    Message(
                        role="user",
                        content="Sealed provider-observed agent-task trajectory receipt",
                    ),
                ),
                tags=("live-agent-task", ledger.suite_id),
                trajectory_id=attempt.trajectory_id,
            )
        )
        grading.append(CaseGrading(case_id=case_id))
        records.append(
            ExecutionRecord(
                id=f"agentic-{case_id}",
                track_id="agentic",
                case_id=case_id,
                attempt_id=f"agentic-{case_id}",
                status="succeeded" if attempt.task_success else "failed",
                selected_arm_id=attempt.selected_arm_id,
                success=attempt.task_success,
                quality=attempt.task_score,
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                runtime_cost=attempt.runtime_cost_usd,
                evaluation_cost=attempt.evaluation_cost_usd,
                trajectory_steps=attempt.trajectory_steps,
                tool_calls=attempt.tool_call_count,
                invalid_tool_calls=attempt.invalid_tool_call_count,
                privacy_violations=attempt.privacy_exposure_count,
                agent_task=attempt,
                grader=attempt.grader_id,
                evidence_kind=LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID,
                broker_receipt=broker_receipt,
            )
        )
    return AgentTaskExecution(
        visible=VisibleCaseSet(cases=tuple(visible)),
        grading=GradingCaseSet(cases=tuple(grading)),
        records=records,
    )
