"""Reduction for complete sealed provider-observed agent-task ledgers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cli.evaluation.agent_task_evidence import AgentTaskMethodEvidence, binary64_equal
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, build_metric
from cli.evaluation.metric_recovery import one_sided_wilson_lower_bound


@dataclass(frozen=True)
class AgentTaskReduction:
    attempt_count: int
    ledger_total_attempt_count: int
    distinct_task_count: int
    ledger_total_distinct_task_count: int
    minimum_distinct_task_count: int
    minimum_attempts_per_task: int
    successful_attempt_count: int
    reliable_task_count: int
    tool_call_count: int
    tool_required_attempt_count: int
    pure_reasoning_attempt_count: int
    required_tool_receipt_coverage: float | None
    task_success_rate: float | None
    task_success_rate_lower_95: float | None
    task_reliability: float | None
    task_reliability_lower_95: float | None
    mean_task_score: float | None
    mean_trajectory_steps: float | None
    invalid_tool_call_rate: float | None
    privacy_exposures_per_attempt: float | None
    total_cost_usd: float | None
    cost_per_successful_attempt_usd: float | None
    policy_snapshot_digest: str | None
    config_digest: str | None
    mixture_snapshot_digest: str | None
    complete: bool


def _same_contract(
    left: AgentTaskMethodEvidence, right: AgentTaskMethodEvidence
) -> bool:
    return all(
        (
            left.ledger_id == right.ledger_id,
            left.source_id == right.source_id,
            left.suite_id == right.suite_id,
            left.suite_revision == right.suite_revision,
            left.task_set_digest == right.task_set_digest,
            left.benchmark_parity_claim == right.benchmark_parity_claim,
            left.execution_semantics == right.execution_semantics,
            left.policy_snapshot_digest == right.policy_snapshot_digest,
            left.config_digest == right.config_digest,
            left.target_id == right.target_id,
            left.backend_topology_digest == right.backend_topology_digest,
            left.mixture_snapshot_digest == right.mixture_snapshot_digest,
            left.ledger_total_attempt_count == right.ledger_total_attempt_count,
            left.ledger_total_distinct_task_count
            == right.ledger_total_distinct_task_count,
            left.minimum_distinct_task_count == right.minimum_distinct_task_count,
            left.minimum_attempts_per_task == right.minimum_attempts_per_task,
        )
    )


def _empty_reduction() -> AgentTaskReduction:
    return AgentTaskReduction(
        attempt_count=0,
        ledger_total_attempt_count=0,
        distinct_task_count=0,
        ledger_total_distinct_task_count=0,
        minimum_distinct_task_count=0,
        minimum_attempts_per_task=0,
        successful_attempt_count=0,
        reliable_task_count=0,
        tool_call_count=0,
        tool_required_attempt_count=0,
        pure_reasoning_attempt_count=0,
        required_tool_receipt_coverage=None,
        task_success_rate=None,
        task_success_rate_lower_95=None,
        task_reliability=None,
        task_reliability_lower_95=None,
        mean_task_score=None,
        mean_trajectory_steps=None,
        invalid_tool_call_rate=None,
        privacy_exposures_per_attempt=None,
        total_cost_usd=None,
        cost_per_successful_attempt_usd=None,
        policy_snapshot_digest=None,
        config_digest=None,
        mixture_snapshot_digest=None,
        complete=False,
    )


_TaskContract = tuple[str, str, str, float, bool, tuple[str, ...]]


@dataclass
class _AgentTaskAccumulator:
    first: AgentTaskMethodEvidence | None = None
    attempts: set[str] = field(default_factory=set)
    trajectories: set[str] = field(default_factory=set)
    receipts: set[str] = field(default_factory=set)
    task_attempts: dict[str, int] = field(default_factory=dict)
    task_all_successful: dict[str, bool] = field(default_factory=dict)
    task_contracts: dict[str, _TaskContract] = field(default_factory=dict)
    successful: int = 0
    total_score: float = 0.0
    total_steps: int = 0
    total_tool_calls: int = 0
    total_invalid_calls: int = 0
    total_privacy: int = 0
    total_cost: float = 0.0
    tool_required_attempts: int = 0
    pure_reasoning_attempts: int = 0
    tool_required_attempts_with_receipt: int = 0

    def add(self, method: AgentTaskMethodEvidence) -> None:
        self._bind_ledger_contract(method)
        self._bind_unique_identities(method)
        self._bind_task_contract(method)
        self._accumulate_outcome(method)

    def _bind_ledger_contract(self, method: AgentTaskMethodEvidence) -> None:
        if self.first is None:
            self.first = method
        elif not _same_contract(self.first, method):
            raise ValueError("agent-task rows mix sealed ledger contracts")

    def _bind_unique_identities(self, method: AgentTaskMethodEvidence) -> None:
        if method.attempt_id in self.attempts:
            raise ValueError("agent-task attempt identities must be unique")
        if method.trajectory_id in self.trajectories:
            raise ValueError("agent-task trajectory identities must be unique")
        self.attempts.add(method.attempt_id)
        self.trajectories.add(method.trajectory_id)
        for receipt in method.receipts:
            if receipt in self.receipts:
                raise ValueError("agent-task receipts must be unique")
            self.receipts.add(receipt)

    def _bind_task_contract(self, method: AgentTaskMethodEvidence) -> None:
        contract: _TaskContract = (
            method.task_spec_digest,
            method.grader_id,
            method.grader_revision_digest,
            method.success_threshold,
            method.tool_policy.requires_tools,
            method.tool_policy.expected_tools,
        )
        prior = self.task_contracts.setdefault(method.task_id, contract)
        if (
            prior[:3] != contract[:3]
            or not binary64_equal(prior[3], contract[3])
            or prior[4:] != contract[4:]
        ):
            raise ValueError("agent-task repetitions mix task or grader contracts")

    def _accumulate_outcome(self, method: AgentTaskMethodEvidence) -> None:
        self.task_attempts[method.task_id] = (
            self.task_attempts.get(method.task_id, 0) + 1
        )
        self.task_all_successful[method.task_id] = (
            self.task_all_successful.get(method.task_id, True) and method.task_success
        )
        self.successful += int(method.task_success)
        self.total_score += method.task_score
        self.total_steps += method.trajectory_steps
        self.total_tool_calls += method.tool_call_count
        self._accumulate_tool_policy(method)
        self.total_invalid_calls += method.invalid_tool_call_count
        self.total_privacy += method.privacy_exposure_count
        self.total_cost += method.total_cost_usd
        if not math.isfinite(self.total_score) or not math.isfinite(self.total_cost):
            raise ValueError("agent-task metric aggregate is not finite")

    def _accumulate_tool_policy(self, method: AgentTaskMethodEvidence) -> None:
        if method.tool_policy.requires_tools:
            self.tool_required_attempts += 1
            if any(
                call.outcome == "executed" and call.execution_receipt_digest is not None
                for call in method.tool_calls
            ):
                self.tool_required_attempts_with_receipt += 1
        else:
            self.pure_reasoning_attempts += 1

    def build(self) -> AgentTaskReduction:
        first = self.first
        if first is None:
            return _empty_reduction()
        count = len(self.attempts)
        distinct_tasks = len(self.task_attempts)
        reliable = sum(self.task_all_successful.values())
        minimum_attempts_satisfied = all(
            count >= first.minimum_attempts_per_task
            for count in self.task_attempts.values()
        )
        return AgentTaskReduction(
            attempt_count=count,
            ledger_total_attempt_count=first.ledger_total_attempt_count,
            distinct_task_count=distinct_tasks,
            ledger_total_distinct_task_count=first.ledger_total_distinct_task_count,
            minimum_distinct_task_count=first.minimum_distinct_task_count,
            minimum_attempts_per_task=first.minimum_attempts_per_task,
            successful_attempt_count=self.successful,
            reliable_task_count=reliable,
            tool_call_count=self.total_tool_calls,
            tool_required_attempt_count=self.tool_required_attempts,
            pure_reasoning_attempt_count=self.pure_reasoning_attempts,
            required_tool_receipt_coverage=(
                self.tool_required_attempts_with_receipt / self.tool_required_attempts
                if self.tool_required_attempts
                else None
            ),
            task_success_rate=self.successful / count,
            task_success_rate_lower_95=one_sided_wilson_lower_bound(
                self.successful, count
            ),
            task_reliability=reliable / distinct_tasks,
            task_reliability_lower_95=one_sided_wilson_lower_bound(
                reliable, distinct_tasks
            ),
            mean_task_score=self.total_score / count,
            mean_trajectory_steps=self.total_steps / count,
            invalid_tool_call_rate=(
                self.total_invalid_calls / self.total_tool_calls
                if self.total_tool_calls
                else None
            ),
            privacy_exposures_per_attempt=self.total_privacy / count,
            total_cost_usd=self.total_cost,
            cost_per_successful_attempt_usd=(
                self.total_cost / self.successful if self.successful else None
            ),
            policy_snapshot_digest=first.policy_snapshot_digest,
            config_digest=first.config_digest,
            mixture_snapshot_digest=first.mixture_snapshot_digest,
            complete=(
                count == first.ledger_total_attempt_count
                and distinct_tasks == first.ledger_total_distinct_task_count
                and distinct_tasks >= first.minimum_distinct_task_count
                and minimum_attempts_satisfied
                and self.tool_required_attempts_with_receipt
                == self.tool_required_attempts
                and self.tool_required_attempts + self.pure_reasoning_attempts == count
            ),
        )


def reduce_agent_tasks(records: list[ExecutionRecord]) -> AgentTaskReduction:
    accumulator = _AgentTaskAccumulator()
    for row in records:
        method = row.agent_task
        if row.track_id == "agentic" and method is not None:
            accumulator.add(method)
    return accumulator.build()


_MetricValue = tuple[str, str, float | None, str, str, int]


def _task_outcome_metric_values(
    reduced: AgentTaskReduction,
) -> tuple[_MetricValue, ...]:
    return (
        (
            "agentic.task_attempt_count",
            "Sealed agent-task attempt count",
            float(reduced.attempt_count) if reduced.attempt_count else None,
            "attempts",
            "higher_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_distinct_count",
            "Distinct sealed agent tasks",
            float(reduced.distinct_task_count) if reduced.attempt_count else None,
            "tasks",
            "higher_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_attempt_success_rate",
            "Agent-task attempt success rate",
            reduced.task_success_rate,
            "fraction",
            "higher_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_attempt_success_rate_lower_95",
            "One-sided 95% agent-task attempt success lower bound",
            reduced.task_success_rate_lower_95,
            "fraction",
            "higher_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_reliability",
            "Repeated-task all-attempt reliability",
            reduced.task_reliability,
            "fraction",
            "higher_is_better",
            reduced.distinct_task_count,
        ),
        (
            "agentic.task_reliability_lower_95",
            "One-sided 95% repeated-task reliability lower bound",
            reduced.task_reliability_lower_95,
            "fraction",
            "higher_is_better",
            reduced.distinct_task_count,
        ),
        (
            "agentic.task_mean_score",
            "Mean sealed agent-task score",
            reduced.mean_task_score,
            "score",
            "higher_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_mean_steps",
            "Mean sealed agent-task trajectory steps",
            reduced.mean_trajectory_steps,
            "steps",
            "target",
            reduced.attempt_count,
        ),
        (
            "agentic.task_invalid_tool_rate",
            "Invalid tool-call rate in sealed agent tasks",
            reduced.invalid_tool_call_rate,
            "fraction",
            "lower_is_better",
            reduced.tool_call_count,
        ),
    )


def _task_policy_metric_values(
    reduced: AgentTaskReduction,
) -> tuple[_MetricValue, ...]:
    return (
        (
            "agentic.task_tool_required_attempt_count",
            "Tool-required agent-task attempts",
            (
                float(reduced.tool_required_attempt_count)
                if reduced.attempt_count
                else None
            ),
            "attempts",
            "target",
            reduced.attempt_count,
        ),
        (
            "agentic.task_pure_reasoning_attempt_count",
            "Pure-reasoning agent-task attempts",
            (
                float(reduced.pure_reasoning_attempt_count)
                if reduced.attempt_count
                else None
            ),
            "attempts",
            "target",
            reduced.attempt_count,
        ),
        (
            "agentic.task_required_tool_receipt_coverage",
            "Provider-executed required-tool receipt coverage",
            reduced.required_tool_receipt_coverage,
            "fraction",
            "higher_is_better",
            reduced.tool_required_attempt_count,
        ),
        (
            "agentic.task_privacy_exposures_per_attempt",
            "Privacy exposures per sealed agent-task attempt",
            reduced.privacy_exposures_per_attempt,
            "exposures/attempt",
            "lower_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_total_cost_usd",
            "Complete sealed agent-task cost",
            reduced.total_cost_usd,
            "USD",
            "lower_is_better",
            reduced.attempt_count,
        ),
        (
            "agentic.task_cost_per_success_usd",
            "Complete agent-task cost per successful attempt",
            reduced.cost_per_successful_attempt_usd,
            "USD/success",
            "lower_is_better",
            reduced.successful_attempt_count,
        ),
    )


def agent_task_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    reduced = reduce_agent_tasks(records)
    values = _task_outcome_metric_values(reduced) + _task_policy_metric_values(reduced)
    return [
        build_metric(metric_id, name, "agentic", value, unit, direction, count)
        for metric_id, name, value, unit, direction, count in values
    ]
