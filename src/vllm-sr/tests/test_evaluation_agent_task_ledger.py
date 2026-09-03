from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from cli.evaluation import live_runtime_collection
from cli.evaluation.agent_task_evidence import (
    AGENT_TASK_ATTEMPT_VERSION,
    AGENT_TASK_BENCHMARK_PARITY_CLAIM,
    AGENT_TASK_EXECUTION_SEMANTICS,
    AGENT_TASK_METHOD_ID,
    MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK,
    MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
    AgentTaskMethodEvidence,
)
from cli.evaluation.agent_task_ledger import (
    AGENT_TASK_LEDGER_VERSION,
    AgentTaskExecution,
    AgentTaskLedger,
    agent_task_set_digest,
    execute_agent_task_ledger,
)
from cli.evaluation.builtin_executors import LiveRuntimeExecutor
from cli.evaluation.catalog import get_catalog
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import track_evidence_level
from cli.evaluation.gates import compute_gates
from cli.evaluation.http_client import HTTPResult
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.method_ledger_identity import method_mixture_binding
from cli.evaluation.metric_agent_task import agent_task_metrics, reduce_agent_tasks
from cli.evaluation.target_capabilities import DEFAULT_TARGET_REGISTRY
from cli.evaluation.target_contracts import (
    EvaluationTargetArm,
    HTTPServiceEndpoint,
    ManifestMixture,
    MixtureDecisionBinding,
)
from evaluation_contract_test_support import _live_manifest, build_routing_recipe_plan
from pydantic import ValidationError


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


_START = datetime(2026, 8, 31, 1, tzinfo=UTC)
_POLICY = _digest("agent-task-policy")
_CONFIG = _digest("agent-task-config")
_TOPOLOGY = _digest("agent-task-topology")
_BROKER_RECEIPT = _digest("agent-task-broker")
_MIXTURE_SNAPSHOT_GOLDEN = (
    "sha256:8d229b7c78bbf7865ae1b4c3dd9f6709d6afa36cbb1118274302cf03b23021d3"
)


class _LedgerClient:
    def __init__(
        self, payload: dict[str, object], *, fetched_at: datetime | None = None
    ):
        self.payload = payload
        self.fetched_at = fetched_at or _START + timedelta(minutes=12)
        self.calls: list[dict[str, object]] = []

    def get(self, endpoint: str, **kwargs: object) -> HTTPResult:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return HTTPResult(
            success=True,
            status_code=200,
            payload=self.payload,
            latency_ms=1.0,
            headers={},
            broker_receipt=_BROKER_RECEIPT,
            fetched_at=self.fetched_at,
        )


def _mixture() -> ManifestMixture:
    arm = EvaluationTargetArm(
        id="agent-arm",
        model="provider/agent-model",
        provider_model_id_digest=_digest("agent-model"),
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )
    arms = (arm,)
    recipe_name = "agent-task-recipe"
    selector_policy_digest = _digest("agent-selector-policy")
    recipe_digest = _digest("agent-recipe")
    pool_digest = model_pool_snapshot_digest(arms)
    selector_digest = selector_snapshot_digest(selector_policy_digest, ())
    adaptation_digest = _digest("agent-adaptation")
    binding_digest = _digest("agent-binding")
    return ManifestMixture(
        id=mixture_target_id(recipe_name),
        entrypoint_model="agent-entrypoint",
        aliases=("agent-entrypoint",),
        recipe_name=recipe_name,
        recipe_description="Frozen provider-observed agent-task subject",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy_digest,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=(),
        fallback_arm_id=arm.id,
        decisions=(
            MixtureDecisionBinding(
                name="default", algorithm="single", arm_ids=(arm.id,)
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy_digest,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=(arm.id,),
            fallback_arm_id=arm.id,
            signals=(),
            projections=(),
        ),
    )


def _attempt(index: int, *, mixture: ManifestMixture) -> AgentTaskMethodEvidence:
    task_index = index // MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK
    repetition = index % MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK
    started = _START + timedelta(seconds=index * 10 + 1)
    completed = started + timedelta(seconds=2)
    return AgentTaskMethodEvidence(
        contract_version=AGENT_TASK_ATTEMPT_VERSION,
        method_id=AGENT_TASK_METHOD_ID,
        ledger_id="agent-task-ledger",
        source_id="provider-agent-runtime",
        suite_id="provider-agent-tasks",
        suite_revision=_digest("agent-task-suite"),
        task_set_digest=_digest("pending-task-set"),
        benchmark_parity_claim=AGENT_TASK_BENCHMARK_PARITY_CLAIM,
        execution_semantics=AGENT_TASK_EXECUTION_SEMANTICS,
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=mixture.id,
        backend_topology_digest=_TOPOLOGY,
        mixture_snapshot_digest=method_mixture_binding(mixture).snapshot_digest,
        ledger_total_attempt_count=(
            MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT
            * MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK
        ),
        ledger_total_distinct_task_count=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
        minimum_distinct_task_count=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
        minimum_attempts_per_task=MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK,
        task_id=f"task-{task_index}",
        task_spec_digest=_digest(f"task-spec-{task_index}"),
        tool_policy={"requires_tools": True, "expected_tools": ("provider-tool",)},
        attempt_id=f"attempt-{index}",
        repetition_id=f"repetition-{repetition}",
        trajectory_id=f"trajectory-{index}",
        seed=index,
        selected_arm_id="agent-arm",
        task_success=True,
        task_score=1.0,
        success_threshold=0.8,
        grader_id="provider-task-grader",
        grader_revision_digest=_digest("provider-task-grader-v1"),
        grading_receipt_digest=_digest(f"grading-{index}"),
        privacy_audit_receipt_digest=_digest(f"privacy-{index}"),
        execution_receipt_digest=_digest(f"attempt-execution-{index}"),
        trajectory_steps=3,
        tool_call_count=1,
        invalid_tool_call_count=0,
        privacy_exposure_count=index % 2,
        input_tokens=10,
        output_tokens=5,
        model_cost_usd=0.00002,
        tool_cost_usd=2.0,
        evaluation_cost_usd=3.0,
        total_cost_usd=5.00002,
        started_at=started,
        completed_at=completed,
        graded_at=completed + timedelta(seconds=1),
        privacy_audited_at=completed + timedelta(seconds=2),
        tool_calls=(
            {
                "sequence": 1,
                "tool_call_id": f"tool-call-{index}",
                "tool_name": "provider-tool",
                "arguments_digest": _digest(f"arguments-{index}"),
                "outcome": "executed",
                "result_digest": _digest(f"result-{index}"),
                "execution_receipt_digest": _digest(f"tool-execution-{index}"),
                "cost_usd": 2.0,
                "started_at": started + timedelta(milliseconds=100),
                "completed_at": started + timedelta(milliseconds=500),
            },
        ),
    )


def _ledger() -> AgentTaskLedger:
    mixture = _mixture()
    attempts = tuple(
        _attempt(index, mixture=mixture)
        for index in range(
            MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT
            * MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK
        )
    )
    task_set_digest = agent_task_set_digest(attempts)
    attempts = tuple(
        attempt.model_copy(update={"task_set_digest": task_set_digest})
        for attempt in attempts
    )
    return AgentTaskLedger(
        contract_version=AGENT_TASK_LEDGER_VERSION,
        ledger_id="agent-task-ledger",
        source_id="provider-agent-runtime",
        environment="production",
        suite_id="provider-agent-tasks",
        suite_revision=_digest("agent-task-suite"),
        task_set_digest=task_set_digest,
        benchmark_parity_claim=AGENT_TASK_BENCHMARK_PARITY_CLAIM,
        execution_semantics=AGENT_TASK_EXECUTION_SEMANTICS,
        provider_attestation_digest=_digest("provider-attestation"),
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=mixture.id,
        backend_topology_digest=_TOPOLOGY,
        mixture=method_mixture_binding(mixture),
        ledger_total_attempt_count=len(attempts),
        ledger_total_distinct_task_count=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
        minimum_distinct_task_count=MINIMUM_AGENT_TASK_DISTINCT_TASK_COUNT,
        minimum_attempts_per_task=MINIMUM_AGENT_TASK_ATTEMPTS_PER_TASK,
        window_started_at=_START,
        window_ended_at=_START + timedelta(minutes=10),
        sealed_at=_START + timedelta(minutes=11),
        attempts=attempts,
    )


def _execute(
    ledger: AgentTaskLedger,
    *,
    mixture: ManifestMixture | None = None,
    target_id: str | None = None,
    topology_digest: str | None = None,
    fetched_at: datetime | None = None,
):
    client = _LedgerClient(ledger.model_dump(mode="json"), fetched_at=fetched_at)
    execution = execute_agent_task_ledger(
        client,  # type: ignore[arg-type]
        "https://agent-runtime.example.test/sealed-window",
        policy_snapshot_digest=_POLICY,
        config_digest=_CONFIG,
        target_id=target_id or _mixture().id,
        backend_topology_digest=topology_digest or _TOPOLOGY,
        mixture=mixture or _mixture(),
        sample_limit=len(ledger.attempts),
        seed=7,
    )
    assert client.calls == [
        {
            "endpoint": "https://agent-runtime.example.test/sealed-window",
            "track_id": "agentic",
            "case_id": "agent-task-ledger",
            "attempt_id": "ledger-fetch",
            "broker_operation": "agent-task.ledger",
        }
    ]
    return execution


def test_method_mixture_binding_matches_go_golden() -> None:
    assert method_mixture_binding(_mixture()).snapshot_digest == (
        _MIXTURE_SNAPSHOT_GOLDEN
    )


def test_agent_task_catalog_method_never_claims_g6() -> None:
    catalog = get_catalog(
        generated_at=False,
        agent_task_ledger=HTTPServiceEndpoint(url="https://agent-task.example.test"),
    )
    task_suite = next(
        suite for suite in catalog.suites if suite.id == "live-agent-tasks"
    )
    recovery_suite = next(
        suite for suite in catalog.suites if suite.id == "live-fault-recovery"
    )

    assert task_suite.methods[0].status == "configured"
    assert task_suite.methods[0].qualified_gate_ids == ()
    assert recovery_suite.methods[0].status == "data_required"
    assert recovery_suite.methods[0].qualified_gate_ids == ("G6",)


def test_complete_provider_observed_task_ledger_reduces_decision_metrics() -> None:
    ledger = _ledger()
    execution = _execute(ledger)
    reduced = reduce_agent_tasks(execution.records)

    assert len(execution.records) == 40
    assert reduced.complete is True
    assert reduced.attempt_count == 40
    assert reduced.distinct_task_count == 20
    assert reduced.task_success_rate == 1.0
    assert reduced.task_reliability == 1.0
    assert reduced.invalid_tool_call_rate == 0.0
    assert reduced.tool_required_attempt_count == 40
    assert reduced.pure_reasoning_attempt_count == 0
    assert reduced.required_tool_receipt_coverage == 1.0
    assert reduced.privacy_exposures_per_attempt == 0.5
    assert reduced.total_cost_usd == pytest.approx(200.0008)
    assert reduced.cost_per_successful_attempt_usd == pytest.approx(5.00002)
    assert all(record.recovery is None for record in execution.records)
    values = {
        metric.id: metric.value for metric in agent_task_metrics(execution.records)
    }
    assert values["agentic.task_attempt_count"] == 40.0
    assert values["agentic.task_distinct_count"] == 20.0
    assert values["agentic.task_reliability"] == 1.0
    assert values["agentic.task_tool_required_attempt_count"] == 40.0
    assert values["agentic.task_pure_reasoning_attempt_count"] == 0.0
    assert values["agentic.task_required_tool_receipt_coverage"] == 1.0
    assert values["agentic.task_total_cost_usd"] == pytest.approx(200.0008)
    assert (
        track_evidence_level(
            "live",
            LiveRuntimeExecutor.contract,
            "agentic",
            execution.records,
        )
        == "E5"
    )
    forged = [
        execution.records[0].model_copy(update={"quality": 0.25}),
        *execution.records[1:],
    ]
    assert (
        track_evidence_level("live", LiveRuntimeExecutor.contract, "agentic", forged)
        == "E0"
    )
    split_receipt = [
        execution.records[0].model_copy(
            update={"broker_receipt": "sha256:" + "f" * 64}
        ),
        *execution.records[1:],
    ]
    assert (
        track_evidence_level(
            "live", LiveRuntimeExecutor.contract, "agentic", split_receipt
        )
        == "E0"
    )

    # Task success does not substitute for G6's injected-fault continuity.
    g6 = next(
        gate
        for gate in compute_gates(
            agent_task_metrics(execution.records),
            has_records=True,
            change_profile="agent_multimodal",
            records=execution.records,
        )
        if gate.id == "G6"
    )
    assert (g6.disposition, g6.verdict) == ("required", "unavailable")


def test_agent_task_reliability_requires_every_repeated_attempt() -> None:
    execution = _execute(_ledger())
    failed_method = execution.records[0].agent_task.model_copy(
        update={"task_success": False, "task_score": 0.5}
    )
    execution.records[0] = execution.records[0].model_copy(
        update={
            "status": "failed",
            "success": False,
            "quality": 0.5,
            "agent_task": failed_method,
        }
    )

    reduced = reduce_agent_tasks(execution.records)
    assert reduced.task_success_rate == pytest.approx(39 / 40)
    assert reduced.task_reliability == pytest.approx(19 / 20)


def test_agent_task_record_rejects_a_forged_normalized_outcome() -> None:
    record = _execute(_ledger()).records[0].model_dump(mode="json")
    record["quality"] = 0.25
    with pytest.raises(
        ValidationError,
        match="must bind its exact provider-observed task attempt",
    ):
        ExecutionRecord.model_validate(record)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda value: value.update({"unexpected": True}),
            "Extra inputs are not permitted",
        ),
        (
            lambda value: value.update({"benchmark_parity_claim": "native"}),
            "benchmark_parity_claim",
        ),
        (
            lambda value: value.update({"attempts": value["attempts"][:-1]}),
            "attempt count does not bind membership",
        ),
    ),
)
def test_agent_task_ledger_rejects_unknown_parity_and_truncated_membership(
    mutation, match: str
) -> None:
    value = _ledger().model_dump(mode="json")
    mutation(value)
    with pytest.raises(ValidationError, match=match):
        AgentTaskLedger.model_validate(value)


def test_agent_task_ledger_rejects_snapshot_substitution_and_receipt_reuse() -> None:
    value = _ledger().model_dump(mode="json")
    value["mixture"]["snapshot_digest"] = _digest("another-mixture")
    with pytest.raises(ValidationError, match="does not bind sealed ledger"):
        AgentTaskLedger.model_validate(value)

    value = _ledger().model_dump(mode="json")
    value["attempts"][1]["grading_receipt_digest"] = value["attempts"][0][
        "grading_receipt_digest"
    ]
    with pytest.raises(ValidationError, match="receipts must be unique"):
        AgentTaskLedger.model_validate(value)


def test_agent_task_ledger_rejects_fake_tool_execution_and_task_spec_drift() -> None:
    value = _ledger().model_dump(mode="json")
    value["attempts"][0]["tool_calls"][0]["outcome"] = "rejected_invalid"
    with pytest.raises(
        ValidationError, match="invalid tool call cannot claim execution"
    ):
        AgentTaskLedger.model_validate(value)

    value = _ledger().model_dump(mode="json")
    value["attempts"][1]["task_spec_digest"] = _digest("forged-task-spec")
    with pytest.raises(ValidationError, match="multiple task specs"):
        AgentTaskLedger.model_validate(value)


def test_agent_task_tool_policy_is_enforced_for_every_attempt() -> None:
    value = _ledger().model_dump(mode="json")
    attempt = value["attempts"][0]
    attempt["tool_calls"] = []
    attempt["tool_call_count"] = 0
    attempt["tool_cost_usd"] = 0.0
    attempt["total_cost_usd"] -= 2.0
    with pytest.raises(ValidationError, match="lacks a provider-executed receipt"):
        AgentTaskLedger.model_validate(value)

    value = _ledger().model_dump(mode="json")
    value["attempts"][0]["tool_policy"]["expected_tools"] = ["other-tool"]
    with pytest.raises(ValidationError, match="outside its expected-tool policy"):
        AgentTaskLedger.model_validate(value)


def test_pure_reasoning_tasks_are_explicit_and_need_no_fake_tool_receipt() -> None:
    source = _ledger().model_dump(mode="json")
    for attempt in source["attempts"]:
        attempt["tool_policy"] = {"requires_tools": False, "expected_tools": []}
        attempt["tool_calls"] = []
        attempt["tool_call_count"] = 0
        attempt["tool_cost_usd"] = 0.0
        attempt["total_cost_usd"] -= 2.0
    parsed_attempts = tuple(
        AgentTaskMethodEvidence.model_validate(attempt)
        for attempt in source["attempts"]
    )
    task_set_digest = agent_task_set_digest(parsed_attempts)
    source["task_set_digest"] = task_set_digest
    for attempt in source["attempts"]:
        attempt["task_set_digest"] = task_set_digest
    ledger = AgentTaskLedger.model_validate(source)
    reduced = reduce_agent_tasks(_execute(ledger).records)
    assert reduced.complete is True
    assert reduced.tool_required_attempt_count == 0
    assert reduced.pure_reasoning_attempt_count == len(ledger.attempts)
    assert reduced.required_tool_receipt_coverage is None


def test_agent_task_execution_rejects_mixture_or_partial_window() -> None:
    ledger = _ledger()
    client = _LedgerClient(ledger.model_dump(mode="json"))
    with pytest.raises(ValueError, match="sample_limit must cover every attempt"):
        execute_agent_task_ledger(
            client,  # type: ignore[arg-type]
            "https://agent-runtime.example.test/sealed-window",
            policy_snapshot_digest=_POLICY,
            config_digest=_CONFIG,
            target_id=_mixture().id,
            backend_topology_digest=_TOPOLOGY,
            mixture=_mixture(),
            sample_limit=1,
            seed=7,
        )

    different = _mixture().model_copy(
        update={"binding_digest": _digest("another-binding")}
    )
    with pytest.raises(ValueError, match="another Mixture snapshot"):
        _execute(ledger, mixture=different)

    with pytest.raises(ValueError, match="another Mixture snapshot"):
        _execute(ledger, target_id="different-target")

    with pytest.raises(ValueError, match="another Mixture snapshot"):
        _execute(ledger, topology_digest=_digest("different-topology"))


@pytest.mark.parametrize(
    ("fetched_at", "message"),
    (
        (_START + timedelta(minutes=10, seconds=59), "future"),
        (_START + timedelta(hours=24, minutes=11, seconds=1), "freshness"),
    ),
)
def test_agent_task_execution_rejects_future_and_stale_ledger(
    fetched_at: datetime, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _execute(_ledger(), fetched_at=fetched_at)


def test_agent_task_execution_rejects_forged_model_cost() -> None:
    value = _ledger().model_dump(mode="json")
    value["attempts"][0]["model_cost_usd"] += 1.0
    value["attempts"][0]["total_cost_usd"] += 1.0
    ledger = AgentTaskLedger.model_validate(value)

    with pytest.raises(ValueError, match="frozen Mixture pricing"):
        _execute(ledger)


def test_live_runtime_dispatches_agent_tasks_without_recovery_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _live_manifest("agent-task-runtime-dispatch")
    manifest = base.with_semantic_updates(
        target=base.target.model_copy(
            update={
                "agent_task_ledger": HTTPServiceEndpoint(
                    url="https://agent-task.example.test"
                ),
                "fault_recovery_ledger": None,
            }
        ),
        suite_ids=("live-agent-tasks",),
        suite_revisions={"live-agent-tasks": "executor-v1"},
        suite_executors={"live-agent-tasks": "live-runtime.v1"},
        track_ids=("agentic",),
        sample_limit=40,
    )
    DEFAULT_TARGET_REGISTRY.resolve(manifest, LiveRuntimeExecutor.contract)

    recovery_only = manifest.with_semantic_updates(
        target=manifest.target.model_copy(
            update={
                "agent_task_ledger": None,
                "fault_recovery_ledger": HTTPServiceEndpoint(
                    url="https://recovery.example.test"
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="agent_task_ledger"):
        DEFAULT_TARGET_REGISTRY.resolve(recovery_only, LiveRuntimeExecutor.contract)
    captured: dict[str, object] = {}
    observed = _execute(_ledger())

    def execute_task(*args: object, **kwargs: object) -> AgentTaskExecution:
        captured["args"] = args
        captured.update(kwargs)
        return observed

    def execute_recovery(*args: object, **kwargs: object) -> None:
        pytest.fail("agent-task suite dispatched the G6 fault-recovery ledger")

    monkeypatch.setattr(
        live_runtime_collection, "execute_agent_task_ledger", execute_task
    )
    monkeypatch.setattr(
        live_runtime_collection, "execute_fault_recovery_ledger", execute_recovery
    )
    live_runtime_collection._append_ledger_evidence(manifest, [], [], [])

    assert captured["target_id"] == manifest.target.id
    assert captured["mixture"] == manifest.target.mixture
    assert captured["sample_limit"] == 40
