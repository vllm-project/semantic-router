from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import requests
from cli.evaluation import live_executor as live_executor_module
from cli.evaluation.canonical import digest_value, sha256_digest
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contracts import EvaluationTarget, EvaluationTargetArm, RunManifest
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.http_client import EvaluationHTTPClient
from cli.evaluation.live_executor import (
    LiveExecutionResult,
    LiveRawResult,
    execute_live_raw,
    grade_live_execution,
)
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.resolution import live_grading
from cli.evaluation.store import LocalArtifactStore

_LIVE_TRACKS = ("routing", "multimodal", "capacity")


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
        if "17 + 25" in serialized:
            return "42"
        if "Red Planet" in serialized:
            return "Mars"
        if "image_url" in serialized:
            return "white pixel"
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
                "authorization": "must-not-be-captured",
            }
            if json.get("model") == "entrypoint-a" and self.correlated
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
                        },
                    },
                    {
                        "id": "entrypoint-a",
                        "routing": {
                            "resolution": "virtual",
                            "selectable": True,
                            "default_route": True,
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
        model_arms=_arms(),
        client=EvaluationHTTPClient(session=session),
    )
    result = grade_live_execution(
        raw,
        inputs.visible,
        live_grading(inputs.grading),
        track_ids=track_ids,
        model_arms=_arms(),
    )
    return raw, result


def _manifest() -> RunManifest:
    return RunManifest(
        manifest_digest="sha256:" + "0" * 64,
        run_id="live-full-slice",
        mode="live",
        target=EvaluationTarget(
            id="runtime",
            kind="runtime",
            router_api_url="http://router:8080",
            envoy_url="http://envoy:8801",
            backend_topology_digest=digest_value("backend-topology-v1"),
            model_arms=_arms(),
        ),
        change_profile="recipe",
        gate_contract_version="evaluation-release-gates.v1",
        suite_ids=("live-routing-core", "live-multimodal", "live-capacity"),
        suite_revisions={
            "live-routing-core": "executor-v1",
            "live-multimodal": "executor-v1",
            "live-capacity": "executor-v1",
        },
        track_ids=_LIVE_TRACKS,
        sample_limit=4,
        concurrency=3,
        seed=17,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value("live-policy"),
        config_digest=digest_value("live-config"),
        redaction_policy="public-safe-v1",
    )


def test_live_diagnostic_routing_multimodal_and_capacity_smoke() -> None:
    session = FakeSession()
    raw, result = _run(session, track_ids=_LIVE_TRACKS)

    assert raw.discovered_entrypoints == ("entrypoint-a", "entrypoint-b")
    routing = [row for row in result.records if row.track_id == "routing"]
    assert routing and all(row.quality is None for row in routing)
    assert all(row.evidence_kind == "live-routing-diagnostic-smoke" for row in routing)
    assert {payload["model"] for _, payload, _ in session.posts} == {"entrypoint-a"}

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


def test_routing_only_target_uses_the_canonical_entrypoint_fallback() -> None:
    session = FakeSession()
    raw = execute_live_raw(
        fixture_inputs().visible,
        track_ids=("routing",),
        router_api_url="http://router:8080",
        envoy_url=None,
        concurrency=1,
        model_arms=_arms(),
        client=EvaluationHTTPClient(session=session),
    )

    assert raw.discovered_entrypoints == ()
    assert {payload["model"] for _, payload, _ in session.posts} == {"vllm-sr/auto"}


def test_unattested_live_model_pool_and_joint_are_rejected() -> None:
    with pytest.raises(ValueError, match="attested direct-arm"):
        _run(FakeSession(), track_ids=("model_pool", "joint"))


@pytest.mark.parametrize("track_id", ("agentic", "preference", "safety"))
def test_generic_live_target_rejects_unqualified_tracks(track_id: str) -> None:
    with pytest.raises(ValueError, match="cannot produce qualified track evidence"):
        _run(FakeSession(), track_ids=(track_id,))


def test_http_client_separates_env_credentials_and_redacts_request_exceptions(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_SR_EVAL_API_KEY", "implicit-secret")
    uncredentialed_session = FakeSession()
    EvaluationHTTPClient(session=uncredentialed_session).post(
        "http://router:8080/api/v1/eval", {"model": "auto"}
    )
    assert "Authorization" not in uncredentialed_session.posts[0][2]

    monkeypatch.setenv("ROUTER_EVAL_KEY", "router-secret")
    session = FakeSession()
    client = EvaluationHTTPClient(session=session, credential_env="ROUTER_EVAL_KEY")
    client.post("http://router:8080/api/v1/eval", {"model": "auto"})
    assert session.posts[0][2]["Authorization"] == "Bearer router-secret"

    failed = EvaluationHTTPClient(session=FailingSession()).post(
        "http://private-host:8080/v1/chat/completions", {"model": "auto"}
    )
    assert failed.error == "request_error:ConnectionError"
    assert "private-host" not in (failed.error or "")
    assert "literal-secret" not in (failed.error or "")

    monkeypatch.setenv("ENVOY_EVAL_KEY", "envoy-secret")
    created: list[str | None] = []

    def client_factory(*, credential_env: str | None = None) -> EvaluationHTTPClient:
        created.append(credential_env)
        return EvaluationHTTPClient(session=session, credential_env=credential_env)

    monkeypatch.setattr(live_executor_module, "EvaluationHTTPClient", client_factory)
    execute_live_raw(
        fixture_inputs().visible,
        track_ids=("routing", "multimodal"),
        router_api_url="http://router:8080",
        envoy_url="http://envoy:8801",
        concurrency=1,
        model_arms=_arms(),
        router_api_key_env="ROUTER_EVAL_KEY",
        envoy_api_key_env="ENVOY_EVAL_KEY",
    )
    assert created == ["ROUTER_EVAL_KEY", "ENVOY_EVAL_KEY"]
    authorization_values = {
        headers.get("Authorization") for _, _, headers in session.posts
    }
    assert {"Bearer router-secret", "Bearer envoy-secret"} <= authorization_values


def test_live_run_writes_trace_capacity_and_rich_metric_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    inputs = fixture_inputs()
    raw = execute_live_raw(
        inputs.visible,
        track_ids=_LIVE_TRACKS,
        router_api_url="http://router:8080",
        envoy_url="http://envoy:8801",
        concurrency=3,
        model_arms=_arms(),
        client=EvaluationHTTPClient(session=FakeSession()),
    )

    def fixed_raw(*args: object, **kwargs: object) -> LiveRawResult:
        assert "grading" not in kwargs
        return raw

    monkeypatch.setattr("cli.evaluation.execution.execute_live_raw", fixed_raw)
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)
    artifacts = {artifact.name for artifact in report.artifacts}
    assert {"routing-traces.jsonl", "capacity-profile.json"} <= artifacts

    traces = store.read_run_text(report.run.id, "routing-traces.jsonl")
    records = store.read_run_text(report.run.id, "records.jsonl")
    for private_value in (
        "original_text",
        "must never be persisted",
        "secret.invalid",
        "private free-form reason",
    ):
        assert private_value not in traces
        assert private_value not in records
    capacity = store.read_run_json(report.run.id, "capacity-profile.json")
    assert [level["concurrency"] for level in capacity["levels"]] == [1, 2, 3]
    assert capacity["kind"] == "bounded-concurrency-sweep"
    assert capacity["slo"] is None

    metrics = {metric.id: metric for metric in report.metrics}
    assert metrics["routing.accuracy"].value is None
    assert metrics["capacity.latency_p99_ms"].value is not None
    assert metrics["capacity.level.1.throughput_rps"].value is not None
    assert metrics["capacity.slo_headroom"].value is None
    assert metrics["capacity.cost_per_successful_request"].value is not None
    assert report.provenance.binding_snapshot_digest is not None
    lineage = store.read_run_json(report.run.id, "lineage.json")
    assert lineage["policy"]["entrypoint_model"] == "entrypoint-a"
    levels = {track.track_id: track.evidence_level for track in report.tracks}
    assert levels == dict.fromkeys(_LIVE_TRACKS, "E0")
    assert report.run.evidence_level == "E0"
    assert report.summary.quality_score is None

    persisted_metrics = store.read_run_json(report.run.id, "metrics.json")["metrics"]
    assert persisted_metrics == [
        metric.model_dump(mode="json", exclude_none=False) for metric in report.metrics
    ]
    assert {metric["track_id"] for metric in persisted_metrics} == set(_LIVE_TRACKS)
