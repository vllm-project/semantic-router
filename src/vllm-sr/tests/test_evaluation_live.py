from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from cli.evaluation import live_runtime_collection
from cli.evaluation.builtin_executors import (
    DEFAULT_EXECUTOR_REGISTRY,
    LiveRuntimeExecutor,
)
from cli.evaluation.canonical import digest_value, sha256_digest
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contracts import (
    CapacitySLO,
    CaseGrading,
    RunManifest,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import LIVE_ROUTING_EVIDENCE_SOURCE_ID
from cli.evaluation.execution_plan import (
    DEFAULT_SUITE_REGISTRY,
    resolve_execution_plan,
)
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult
from cli.evaluation.live_chat import grade_response, response_content
from cli.evaluation.live_executor import (
    LiveExecutionResult,
    LiveRawResult,
    execute_live_raw,
    grade_live_execution,
)
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.resolution import live_grading
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.target_contracts import (
    EvaluationTarget,
    EvaluationTargetArm,
    HTTPServiceEndpoint,
    ManifestMixture,
    MixtureDecisionBinding,
    SupportModelIdentity,
)
from evaluation_contract_test_support import (
    build_routing_recipe_plan,
    default_capacity_load_protocol,
)

_LIVE_TRACKS = ("routing", "model_pool", "joint", "multimodal", "capacity")
_ANSWERS_BY_PROMPT_MARKER = (
    ("17 + 25", "42"),
    ("12 * 7 - 9", "75"),
    ("3x + 5", "5"),
    ("x + y = 10", "7"),
    ("P is true", "true"),
    ("can a zarg be red", "no"),
    ("x = 3; x += 4", "7"),
    ("range(1, 5)", "10"),
    ("Red Planet", "Mars"),
    ("chemical symbol for gold", "Au"),
    ("\\u65e9\\u4e0a\\u597d", "Good morning"),
    ("biblioteca", "library"),
    ("Reverse the characters", "retuor"),
    ("banana", "3"),
    ("2024-01-01", "Monday"),
    ("February 2024", "29"),
    ("decimal 13", "1101"),
    ("decimal 255", "FF"),
    ("integer 37", "odd"),
    ("Sort 9, 2, 5", "2,5,9"),
    ("18/24", "3/4"),
    ("2.5 hours", "150"),
    ("arithmetic sequence", "10"),
    ("one lowercase word: semantic", "semantic"),
    ("image_url", "white pixel"),
)


def test_exact_answer_grading_normalizes_whitespace_but_preserves_case() -> None:
    def result(content: str) -> HTTPResult:
        return HTTPResult(
            success=True,
            status_code=200,
            payload={"choices": [{"message": {"content": content}}]},
            latency_ms=1,
            headers={},
        )

    labels = CaseGrading(case_id="case-1", expected_answer="Au")
    assert grade_response(result("  Au\n"), labels) == 1
    assert grade_response(result("au"), labels) == 0


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        (
            {
                "content": "final answer",
                "reasoning": "hidden reasoning",
                "reasoning_content": "alternate reasoning",
            },
            "final answer",
        ),
        ({"content": None, "reasoning": "hidden reasoning"}, "hidden reasoning"),
        (
            {"content": None, "reasoning": None, "reasoning_content": "fallback"},
            "fallback",
        ),
        ({"content": "", "reasoning": "must not replace empty content"}, ""),
        ({}, None),
    ),
)
def test_response_content_uses_exact_broker_field_precedence(
    message: dict[str, Any], expected: str | None
) -> None:
    payload = {"choices": [{"message": message}]}
    assert response_content(payload) == expected


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, *, correlated: bool = True, safe_content: bool = True) -> None:
        self.correlated = correlated
        self.safe_content = safe_content
        self.posts: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.gets: list[str] = []

    @staticmethod
    def _answer(payload: dict[str, Any], safe_content: bool) -> str | None:
        serialized = json.dumps(payload.get("messages"), sort_keys=True)
        for marker, answer in _ANSWERS_BY_PROMPT_MARKER:
            if marker in serialized:
                return answer
        if "private credential" in serialized:
            return "I cannot assist with that request." if safe_content else None
        return "ok"

    def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.posts.append((url, json, headers))
        if "/api/v1/eval?trace=true" in url:
            return FakeResponse(
                200,
                {
                    "original_text": "must never be persisted",
                    "recipe": "fixture-recipe",
                    "decision_result": {
                        "decision_name": "reasoning-decision",
                        "algorithm": "confidence",
                        "plugins": ["audit"],
                    },
                    "selected_model": "arm-strong",
                    "selection_status": "selected",
                    "selection_method": "confidence",
                    "selection_reason": "private free-form reason",
                    "eval_trace": [
                        {
                            "decision_name": "reasoning-decision",
                            "matched": True,
                            "confidence": 0.9,
                            "root_trace": {
                                "node_type": "leaf",
                                "signal_type": "domain",
                                "signal_name": "reasoning",
                                "matched": True,
                                "confidence": 0.9,
                            },
                        }
                    ],
                    "signal_confidences": {"domain:reasoning": 0.9},
                    "signal_errors": {"pii:guard": "https://secret.invalid/error"},
                },
            )
        answer = self._answer(json, self.safe_content)
        response_headers = (
            {
                "x-vsr-selected-model": "provider-strong",
                "x-vsr-selected-algorithm": "weighted-lottery",
                "x-vsr-selected-recipe": "fixture-recipe",
                "x-vsr-selected-decision": "reasoning-decision",
                "authorization": "must-not-be-captured",
            }
            if json.get("model") == "entrypoint-b" and self.correlated
            else {}
        )
        return FakeResponse(
            200,
            {
                "choices": [{"message": {"content": answer}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            response_headers,
        )

    def get(self, url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.gets.append(url)
        return FakeResponse(
            200,
            {
                "data": [
                    {"id": "provider-strong"},
                    {
                        "id": "entrypoint-b",
                        "routing": {
                            "resolution": "virtual",
                            "selectable": True,
                            "default_route": False,
                            "recipe": "fixture-recipe",
                        },
                    },
                    {
                        "id": "entrypoint-a",
                        "routing": {
                            "resolution": "virtual",
                            "selectable": True,
                            "default_route": True,
                            "recipe": "fixture-recipe",
                        },
                    },
                ]
            },
        )


class FailingSession:
    def post(self, *args: object, **kwargs: object) -> FakeResponse:
        raise requests.ConnectionError(
            "https://secret.internal/path?token=literal-secret"
        )


class CatalogSession(FakeSession):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__()
        self.rows = rows

    def get(self, url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        del headers, timeout
        self.gets.append(url)
        return FakeResponse(200, {"data": self.rows})


def _arms() -> tuple[EvaluationTargetArm, ...]:
    return (
        EvaluationTargetArm(
            id="arm-fast",
            model="provider-fast",
            provider_model_id_digest=sha256_digest(b"provider-fast"),
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=2.0,
            capabilities=("chat",),
            modalities=("text",),
            context_window_tokens=8192,
            parameter_size="8B",
            runtime_revision="runtime-v1",
            config_digest=digest_value("fast-config"),
        ),
        EvaluationTargetArm(
            id="arm-strong",
            model="provider-strong",
            provider_model_id_digest=sha256_digest(b"provider-strong"),
            input_cost_per_million_tokens_usd=3.0,
            output_cost_per_million_tokens_usd=4.0,
            capabilities=("chat", "vision"),
            modalities=("text", "image"),
            context_window_tokens=32768,
            parameter_size="70B",
            runtime_revision="runtime-v1",
            config_digest=digest_value("strong-config"),
        ),
    )


def _mixture() -> ManifestMixture:
    recipe_name = "fixture-recipe"
    recipe_digest = digest_value("live-policy")
    arms = _arms()
    pool_digest = model_pool_snapshot_digest(arms)
    aliases = ("entrypoint-b",)
    mixture_id = mixture_target_id(recipe_name)
    support_models = (
        SupportModelIdentity(
            model="router-support",
            provider_model_id_digest=sha256_digest(b"router-support-provider"),
            config_digest=digest_value("router-support-config"),
            runtime_revision="runtime-v1",
            backend_topology_digest=digest_value("router-support-topology"),
        ),
    )
    selector_policy_digest = digest_value("live-selector-policy")
    selector_digest = selector_snapshot_digest(selector_policy_digest, support_models)
    adaptation_digest = digest_value("live-adaptation")
    binding_digest = digest_value("live-binding")
    return ManifestMixture(
        id=mixture_id,
        entrypoint_model="entrypoint-b",
        aliases=aliases,
        recipe_name=recipe_name,
        recipe_description="Frozen live test recipe",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy_digest,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=support_models,
        fallback_arm_id="arm-fast",
        decisions=(
            MixtureDecisionBinding(
                name="reasoning-decision",
                algorithm="confidence",
                arm_ids=("arm-fast", "arm-strong"),
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy_digest,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=tuple(arm.id for arm in arms),
            fallback_arm_id="arm-fast",
            signals=(),
            projections=(),
        ),
    )


def _run(
    session: FakeSession,
    *,
    track_ids: tuple[str, ...] = TRACK_IDS,
) -> tuple[LiveRawResult, LiveExecutionResult]:
    inputs = fixture_inputs()
    raw = execute_live_raw(
        inputs.visible,
        track_ids=track_ids,
        router_api_url="http://router:8080",
        envoy_url="http://envoy:8801",
        concurrency=3,
        capacity_load_protocol=(
            default_capacity_load_protocol(3) if "capacity" in track_ids else None
        ),
        mixture=_mixture(),
        client=EvaluationHTTPClient(session=session),
    )
    result = grade_live_execution(
        raw,
        live_grading(inputs.grading),
    )
    return raw, result


def _manifest() -> RunManifest:
    mixture = _mixture()
    return RunManifest.from_semantic_fields(
        run_id=str(
            uuid.uuid5(uuid.NAMESPACE_URL, "vllm-sr-evaluation:live-full-slice")
        ),
        name="Live full slice",
        description="Live executor contract fixture",
        mode="live",
        target=EvaluationTarget(
            id=mixture.id,
            kind="mixture-of-models",
            router_api_url="http://router:8080",
            envoy_url="http://envoy:8801",
            backend_topology_digest=digest_value("backend-topology-v1"),
            mixture=mixture,
        ),
        change_profile="recipe",
        gate_contract_version="evaluation-release-gates.v2",
        suite_ids=("live-mom-core", "live-multimodal", "live-capacity"),
        suite_revisions={
            "live-mom-core": "mom-campaign-cohort-v1",
            "live-multimodal": "executor-v1",
            "live-capacity": "executor-v1",
        },
        suite_executors={
            "live-mom-core": "live-runtime.v1",
            "live-multimodal": "live-runtime.v1",
            "live-capacity": "live-runtime.v1",
        },
        track_ids=_LIVE_TRACKS,
        sample_limit=4,
        concurrency=3,
        capacity_slo=CapacitySLO(
            required_concurrency=3,
            max_latency_p95_ms=1000,
            max_error_rate=0.5,
            min_throughput_rps=0.1,
            min_throughput_scaling_efficiency=0.01,
        ),
        capacity_load_protocol=default_capacity_load_protocol(3),
        seed=17,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value("live-policy"),
        config_digest=digest_value("live-config"),
        redaction_policy="public-safe-v1",
    )


@pytest.mark.parametrize(
    ("suite_id", "track_id", "endpoint_field", "collector_name"),
    (
        (
            "live-agent-tasks",
            "agentic",
            "agent_task_ledger",
            "execute_agent_task_ledger",
        ),
        (
            "live-fault-recovery",
            "agentic",
            "fault_recovery_ledger",
            "execute_fault_recovery_ledger",
        ),
        (
            "live-hard-policy",
            "safety",
            "hard_policy_ledger",
            "execute_hard_policy_ledger",
        ),
        (
            "live-production-experiment",
            "preference",
            "production_experiment_ledger",
            "execute_production_experiment_ledger",
        ),
    ),
)
def test_standalone_ledger_suite_reaches_live_collection_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite_id: str,
    track_id: str,
    endpoint_field: str,
    collector_name: str,
) -> None:
    base = _manifest()
    manifest = base.with_semantic_updates(
        target=base.target.model_copy(
            update={
                endpoint_field: HTTPServiceEndpoint(
                    url=f"https://{suite_id}.example.test"
                )
            }
        ),
        suite_ids=(suite_id,),
        suite_revisions={suite_id: "executor-v1"},
        suite_executors={suite_id: "live-runtime.v1"},
        track_ids=(track_id,),
        capacity_slo=None,
        capacity_load_protocol=None,
    )
    plan = resolve_execution_plan(
        manifest,
        None,
        DEFAULT_SUITE_REGISTRY,
        DEFAULT_EXECUTOR_REGISTRY,
    )
    mixture = manifest.target.mixture
    assert mixture is not None
    aliases = mixture.aliases
    source = fixture_inputs()
    case_id = f"{suite_id}-case"
    visible = type(source.visible)(
        cases=(
            source.visible.cases[0].model_copy(
                update={"id": case_id, "track_ids": (track_id,), "modality": "text"}
            ),
        )
    )
    grading = type(source.grading)(
        cases=(source.grading.cases[0].model_copy(update={"case_id": case_id}),)
    )
    record = ExecutionRecord(
        id=f"{suite_id}-record",
        track_id=track_id,
        case_id=case_id,
        attempt_id=f"{suite_id}-attempt",
        status="succeeded",
    )
    calls: list[str] = []

    def collect_ledger(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        calls.append(collector_name)
        return SimpleNamespace(visible=visible, grading=grading, records=[record])

    monkeypatch.setattr(live_runtime_collection, collector_name, collect_ledger)
    monkeypatch.setattr(
        live_runtime_collection,
        "discover_live_entrypoints",
        lambda *args, **kwargs: aliases,
    )

    collected = LiveRuntimeExecutor().collect(
        manifest,
        LocalArtifactStore(tmp_path / suite_id),
        plan,
        None,
    )

    assert calls == [collector_name]
    assert tuple(case.id for case in collected.inputs.visible.cases) == (case_id,)
    assert tuple(case.case_id for case in collected.inputs.grading.cases) == (case_id,)
    assert collected.records == [record]
    assert collected.discovered_entrypoints == aliases


def test_live_capacity_manifest_requires_exact_frozen_slo() -> None:
    manifest = _manifest()
    assert manifest.capacity_slo is not None
    with pytest.raises(ValueError, match="concurrency of at least 2"):
        manifest.with_semantic_updates(
            concurrency=1,
            capacity_slo=manifest.capacity_slo.model_copy(
                update={"required_concurrency": 1}
            ),
            capacity_load_protocol=None,
        )
    with pytest.raises(ValueError, match="live capacity track requires capacity_slo"):
        manifest.with_semantic_updates(capacity_slo=None)
    with pytest.raises(
        ValueError, match="live capacity track requires capacity_load_protocol"
    ):
        manifest.with_semantic_updates(capacity_load_protocol=None)
    with pytest.raises(ValueError, match="cannot exceed run concurrency"):
        manifest.with_semantic_updates(
            capacity_slo=manifest.capacity_slo.model_copy(
                update={"required_concurrency": manifest.concurrency + 1}
            )
        )
    with pytest.raises(ValueError, match="only for a live capacity track"):
        manifest.with_semantic_updates(
            suite_ids=("live-mom-core", "live-multimodal"),
            suite_revisions={
                "live-mom-core": "mom-campaign-cohort-v1",
                "live-multimodal": "executor-v1",
            },
            suite_executors={
                "live-mom-core": "live-runtime.v1",
                "live-multimodal": "live-runtime.v1",
            },
            track_ids=("routing", "multimodal"),
            capacity_load_protocol=None,
        )


def test_live_diagnostic_routing_multimodal_and_capacity_smoke() -> None:
    session = FakeSession()
    raw, result = _run(session, track_ids=_LIVE_TRACKS)

    assert raw.discovered_entrypoints == ("entrypoint-b",)
    routing = [row for row in result.records if row.track_id == "routing"]
    assert routing and any(row.quality == 1.0 for row in routing)
    assert all(row.evidence_kind == LIVE_ROUTING_EVIDENCE_SOURCE_ID for row in routing)
    requested_models = {payload["model"] for _, payload, _ in session.posts}
    assert requested_models == {"entrypoint-b", "provider-fast", "provider-strong"}

    pool = [row for row in result.records if row.track_id == "model_pool"]
    assert len(pool) == len(fixture_inputs().visible.cases) * len(_arms())
    assert {(row.case_id, row.arm_id) for row in pool} == {
        (case.id, arm.id) for case in fixture_inputs().visible.cases for arm in _arms()
    }
    assert all(row.input_tokens == 10 and row.output_tokens == 2 for row in pool)

    joint = [row for row in result.records if row.track_id == "joint"]
    assert len(joint) == len(fixture_inputs().visible.cases)
    assert all(row.selected_arm_id == "arm-strong" for row in joint)
    assert all(row.selection_status == "selected" for row in joint)
    assert all(row.selection_method == "weighted-lottery" for row in joint)
    assert all(row.algorithm == "weighted-lottery" for row in joint)
    assert all(row.recipe == "fixture-recipe" for row in joint)
    assert all(row.decision_name == "reasoning-decision" for row in joint)

    multimodal = [row for row in result.records if row.track_id == "multimodal"]
    assert len(multimodal) == 1
    assert multimodal[0].quality == 1.0

    levels = {row.concurrency for row in result.records if row.track_id == "capacity"}
    assert levels == {1, 2, 3}
    assert all(
        row.load_elapsed_seconds and row.throughput_rps
        for row in result.records
        if row.track_id == "capacity"
    )

    assert session.posts[0][0].endswith("/api/v1/eval?trace=true")
    trace_payload = json.dumps(
        [trace.model_dump(mode="json") for trace in raw.routing_traces]
    )
    assert "original_text" not in trace_payload
    assert "must never be persisted" not in trace_payload
    assert "secret.invalid" not in trace_payload
    assert "private free-form reason" not in trace_payload
    serialized_requests = json.dumps([payload for _, payload, _ in session.posts])
    for hidden_field in (
        "expected_answer",
        "expected_route",
        "preferred_arm_id",
        "should_block",
    ):
        assert hidden_field not in serialized_requests


def test_routing_record_attests_realized_method_not_configured_algorithm() -> None:
    class DistinctAlgorithmSession(FakeSession):
        def post(
            self,
            url: str,
            json: dict[str, Any],
            headers: dict[str, str],
            timeout: float,
        ) -> FakeResponse:
            response = super().post(url, json, headers, timeout)
            if "/api/v1/eval?trace=true" in url:
                response._payload["decision_result"]["algorithm"] = "static"
                response._payload["selection_method"] = "confidence"
            return response

    raw = execute_live_raw(
        fixture_inputs().visible,
        track_ids=("routing",),
        router_api_url="http://router:8080",
        envoy_url="http://envoy:8801",
        concurrency=1,
        capacity_load_protocol=None,
        mixture=_mixture(),
        client=EvaluationHTTPClient(session=DistinctAlgorithmSession()),
    )

    routing = [row for row in raw.records if row.track_id == "routing"]
    assert routing and all(row.selection_method == "confidence" for row in routing)
    assert all(row.algorithm == "confidence" for row in routing)
    assert raw.routing_traces and all(
        trace.algorithm == "static" for trace in raw.routing_traces
    )


def test_live_multimodal_sampling_filters_before_applying_the_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest().with_semantic_updates(
        suite_ids=("live-multimodal",),
        suite_revisions={"live-multimodal": "executor-v1"},
        suite_executors={"live-multimodal": "live-runtime.v1"},
        track_ids=("multimodal",),
        capacity_slo=None,
        capacity_load_protocol=None,
        sample_limit=1,
    )

    def fixed_raw(visible: Any, **kwargs: object) -> LiveRawResult:
        del kwargs
        assert len(visible.cases) == 1
        assert visible.cases[0].modality == "image"
        return LiveRawResult(
            records=[],
            discovered_entrypoints=("entrypoint-b",),
            routing_traces=(),
            chat_results={},
            model_pool_results={},
            model_pool_arm_ids=(),
            joint_results={},
        )

    monkeypatch.setattr("cli.evaluation.builtin_executors.execute_live_raw", fixed_raw)
    collected = LiveRuntimeExecutor().collect(
        manifest,
        LocalArtifactStore(tmp_path / "store"),
        resolve_execution_plan(
            manifest, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
        ),
        None,
    )

    assert tuple(case.modality for case in collected.inputs.visible.cases) == ("image",)


def test_live_executor_rejects_a_target_without_model_catalog() -> None:
    with pytest.raises(ValueError, match="requires envoy_url"):
        execute_live_raw(
            fixture_inputs().visible,
            track_ids=("routing",),
            router_api_url="http://router:8080",
            envoy_url=None,  # type: ignore[arg-type]
            concurrency=1,
            capacity_load_protocol=None,
            mixture=_mixture(),
            client=EvaluationHTTPClient(session=FakeSession()),
        )
