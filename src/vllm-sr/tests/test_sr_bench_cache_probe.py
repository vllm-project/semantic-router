from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench import cache_probe
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, SESSION_AWARE, plan
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.grading import CACHE_OBSERVATION_RESULT, validate_result
from cli.sr_bench.offline import export_training, regrade, replay
from cli.sr_bench.report import compare, make_report
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CACHE_OBSERVATION_VERSION, normalize_usage
from test_sr_bench import wait_run

from bench import cache_token_probe as native

ROOT: Path = Path(__file__).resolve().parents[3]
PREFIX_SHA256: str = "7e257ee0907293229e728fe17941c2469a20e12096a303fea393d37f302dbad7"


def usage(*, source: str = "chat", cached: object = 40) -> dict[str, Any]:
    value: dict[str, Any] = (
        {"input_tokens": 100, "output_tokens": 3}
        if source == "input"
        else {"prompt_tokens": 100, "completion_tokens": 3}
    )
    if cached is None:
        return value
    if source == "chat":
        value["prompt_tokens_details"] = {"cached_tokens": cached}
    elif source == "input":
        value = {
            "input_tokens": 100,
            "output_tokens": 3,
            "input_tokens_details": {"cached_tokens": cached},
        }
    elif source == "deepseek":
        value["prompt_cache_hit_tokens"] = cached
    elif source == "root":
        value["cached_tokens"] = cached
    return value


class Provider(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict[str, Any]] = []
        self.usage_events: list[dict[str, Any] | None] = [usage()]
        self.usage_by_request: list[list[dict[str, Any] | None]] | None = None
        self.fault: str | None = None
        self.fault_at: int = 1
        self.omit_model: bool = False
        self.config_hash: str | None = "fixture-config"
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        provider = self.server
        assert isinstance(provider, Provider)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        index = len(provider.requests)
        usage_events = (
            provider.usage_by_request[index]
            if provider.usage_by_request is not None
            else provider.usage_events
        )
        provider.requests.append(
            {
                "body": body,
                "session_id": self.headers.get("x-session-id"),
                "path": self.path,
            }
        )
        fault = provider.fault if index == provider.fault_at else None
        if fault == "connection":
            self.close_connection = True
            return
        if fault == "http":
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        streamed = body.get("stream") is True and fault != "buffered"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "text/event-stream" if streamed else "application/json"
        )
        if provider.config_hash is not None:
            self.send_header("x-sr-bench-config-hash", provider.config_hash)
        self.send_header("x-vsr-inference-call-count", "1")
        self.send_header("x-vsr-session-phase", "user_turn")
        if not provider.omit_model:
            self.send_header("x-vsr-selected-model", "reported-model")
        self.end_headers()
        if fault == "cancel":
            self.wfile.write(b": waiting for cancellation\n\n")
            self.wfile.flush()
            provider.entered.set()
            assert provider.release.wait(timeout=10)
            return
        event: dict[str, Any] = {
            **({"model": "reported-model"} if not provider.omit_model else {}),
            "choices": [
                {
                    "index": 0,
                    "delta" if streamed else "message": {
                        "content": (
                            "A"
                            if body["messages"][-1]["content"] == "Return A"
                            else "A short response."
                        ),
                        "reasoning_content": "This is not cache evidence.",
                    },
                    "finish_reason": "length" if fault == "length" else "stop",
                }
            ],
        }
        with suppress(BrokenPipeError, ConnectionResetError):
            if not streamed:
                if usage_events:
                    event["usage"] = usage_events[-1]
                self.wfile.write(json.dumps(event).encode())
                return
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            for raw in usage_events:
                self.wfile.write(
                    (
                        "data: " + json.dumps({"choices": [], "usage": raw}) + "\n\n"
                    ).encode()
                )
            if fault != "truncated":
                self.wfile.write(b"data: [DONE]\n\n")


@contextmanager
def running_provider() -> Iterator[Provider]:
    server = Provider()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def provider() -> Iterator[Provider]:
    with running_provider() as server:
        yield server


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Server]:
    server = Server(("127.0.0.1", 0), Store(tmp_path / "store"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        for worker in server.engine.threads.values():
            worker.join(timeout=5)
            assert not worker.is_alive()
        server.server_close()
        thread.join(timeout=5)
        server.store.db.close()


def document(*, root: Path, provider: Provider, repeats: int = 6) -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "profile": "quick",
        "seed": 0,
        "cost_policy": "capability_only",
        "dataset": cache_probe.prepare(store=root, repeats=repeats, timeout=3),
        "targets": [
            {
                "id": "subject",
                "kind": "single",
                "model": "requested-model",
                "base_url": provider.base_url,
                "session_mode": SESSION_AWARE,
            }
        ],
        "sampling": {"temperature": 0, "max_tokens": 16},
        "limits": {
            "max_calls_per_case": repeats,
            "total_timeout_s": 3,
            "idle_timeout_s": 2,
            "case_timeout_s": 20,
            "max_run_seconds": 30,
        },
    }


def benchmark_cli(
    *, service: Server, args: list[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.main",
            "benchmark",
            "--url",
            f"http://127.0.0.1:{service.server_port}",
            "--store",
            str(service.store.root),
            "--no-autostart",
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )


def run_cli(
    *, service: Server, value: dict[str, Any], detach: bool = False
) -> subprocess.CompletedProcess[str]:
    path = service.store.root / "probe.json"
    path.write_text(json.dumps(value))
    return benchmark_cli(
        service=service,
        args=[
            "run",
            "--manifest",
            str(path),
            "--idempotency-key",
            "cache-probe-once",
            *(["--detach"] if detach else []),
        ],
    )


def native_cli(
    *,
    provider: Provider,
    output: Path,
    repeats: int = 6,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/cache_token_probe.py"),
            "--base-url",
            provider.base_url,
            "--model",
            "requested-model",
            "--repeats",
            str(repeats),
            "--timeout",
            "3",
            "--output-dir",
            str(output),
            *(extra_args or []),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cache_probe_has_no_answer_quality_contract() -> None:
    adapter = get_adapter(cache_probe.BENCHMARK_ID)
    assert adapter.result_kind == CACHE_OBSERVATION_RESULT
    assert adapter.id not in BENCHMARK_WEIGHTS
    assert not adapter.requires_answer
    result = {
        "answer": None,
        "correct": None,
        "score": None,
        "details": {
            "cache_observation": {
                "version": CACHE_OBSERVATION_RESULT,
                "planned_repeats": 6,
                "completed_repeats": 6,
            }
        },
    }
    validate_result(kind=CACHE_OBSERVATION_RESULT, result=result)
    for name in ("answer", "correct", "score"):
        invalid = {**result, name: "A" if name == "answer" else 0}
        with pytest.raises(ValueError):
            validate_result(kind=CACHE_OBSERVATION_RESULT, result=invalid)


@pytest.mark.parametrize("source", ["chat", "input", "deepseek"])
@pytest.mark.parametrize("cached", [None, 0, 40])
def test_native_and_cli_preserve_requests_and_qualified_observations(
    tmp_path: Path, provider: Provider, service: Server, source: str, cached: int | None
) -> None:
    provider.usage_events = [usage(source=source, cached=cached)]
    legacy = native_cli(provider=provider, output=tmp_path / "native")
    assert legacy.returncode == 0, legacy.stderr
    native_summary = json.loads(legacy.stdout)["summary"]
    native_requests = copy.deepcopy(provider.requests)
    provider.requests.clear()

    value = document(root=tmp_path, provider=provider)
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    target = report["summary"]["targets"][0]
    probe = target["cache_observation"]
    assert target["scored"] == 0
    assert target["observed"] == 1
    assert target["complete"] is True
    assert all(
        target[name] is None
        for name in (
            "accuracy",
            "correct",
            "accuracy_ci95",
            "macro_accuracy",
            "sr_bench_score",
        )
    )
    assert probe["execution_complete"] is True
    assert probe["planned_repeats"] == probe["completed_repeats"] == 6
    assert probe["cached_token_reporting"] == native_summary["cached_token_reporting"]
    assert probe["cached_token_field_rate"] == native_summary["cached_token_field_rate"]
    assert probe["success_rate"] == native_summary["success_rate"]
    assert probe["validation"]["status"] == "pass"
    assert len(probe["repeats"]) == len(native_requests) == len(provider.requests) == 6
    if cached is None:
        assert probe["cached_prompt_ratio"] is None
        assert probe["cached_tokens"] is None
    else:
        assert probe["cached_prompt_ratio"] == native_summary["cached_prompt_ratio"]
        assert probe["cached_tokens"] == native_summary["cached_tokens"]
        assert probe["prompt_tokens"] == native_summary["prompt_tokens"]
    for index, request in enumerate(provider.requests):
        body = request["body"]
        assert {
            key: item
            for key, item in body.items()
            if key not in {"stream", "stream_options"}
        } == native_requests[index]["body"]
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert probe["repeats"][index]["repeat_index"] == index
        assert probe["repeats"][index]["response_status"] == HTTPStatus.OK
    sessions = {request["session_id"] for request in provider.requests}
    assert len(sessions) == 1 and None not in sessions
    assert sessions != {native_requests[0]["session_id"]}
    assert hashlib.sha256(native.probe_prompt().encode()).hexdigest() == PREFIX_SHA256
    calls = service.store.calls(report["run_id"])
    for call in calls:
        assert call["raw_usage"] == provider.usage_events[0]
        assert call["usage"] == normalize_usage(provider.usage_events[0])
        assert (
            call["provider_cache_observation"]["version"] == CACHE_OBSERVATION_VERSION
        )
        assert call["session_id"] in sessions
    again = run_cli(service=service, value=value)
    assert again.returncode == 0
    assert len(provider.requests) == 6
    reopened = Store(service.store.root)
    try:
        assert make_report(reopened, report["run_id"]) == report
    finally:
        reopened.db.close()


def test_root_alias_is_retained_without_qualifying_cache_thresholds(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    provider.usage_events = [usage(source="root", cached=40)]
    legacy = native_cli(provider=provider, output=tmp_path / "native")
    assert json.loads(legacy.stdout)["summary"]["cached_token_reporting"] == "positive"
    provider.requests.clear()
    value = document(root=tmp_path, provider=provider)
    value["benchmark_options"] = {
        cache_probe.BENCHMARK_ID: {
            "thresholds": {
                "min_cached_token_reporting": "positive",
                "min_cached_token_field_rate": 1.0,
                "min_cached_prompt_ratio": 0.1,
            }
        }
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    probe = report["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_tokens"] is None
    assert probe["cached_prompt_ratio"] is None
    assert probe["validation"]["status"] == "incomplete"
    assert probe["validation"]["predicates"]["min_success_rate"]["status"] == "pass"
    assert all(
        row["provider_cache_observation"]["qualification"] == "unqualified"
        for row in probe["repeats"]
    )
    calls = service.store.calls(report["run_id"])
    assert all(call["raw_usage"]["cached_tokens"] == 40 for call in calls)
    assert all(call["usage"]["cached_input_tokens"] == 0 for call in calls)


@pytest.mark.parametrize(
    ("source", "parent", "key", "exit_code"),
    [
        ("chat", "prompt_tokens_details", "cached_tokens", 2),
        ("input", "input_tokens_details", "cached_tokens", 2),
        ("deepseek", None, "prompt_cache_hit_tokens", 0),
        ("root", None, "cached_tokens", 0),
    ],
)
def test_explicit_null_preserves_the_existing_transport_contract(
    tmp_path: Path,
    provider: Provider,
    service: Server,
    source: str,
    parent: str | None,
    key: str,
    exit_code: int,
) -> None:
    raw = usage(source=source, cached=0)
    (raw[parent] if parent else raw)[key] = None
    provider.usage_events = [raw]
    legacy = native_cli(provider=provider, output=tmp_path / "native")
    assert legacy.returncode == 0
    provider.requests.clear()
    result = run_cli(service=service, value=document(root=tmp_path, provider=provider))
    assert result.returncode == exit_code, result.stderr
    probe = json.loads(result.stdout)["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_tokens"] is None
    assert probe["cached_prompt_ratio"] is None
    assert probe["repeats"][0]["provider_cache_observation"]["field_present"] is False
    if exit_code:
        assert probe["repeats"][0]["error"] == "Invalid token usage buckets"
        assert probe["undispatched_repeats"] == 5


@pytest.mark.parametrize(
    "fault", ["http", "connection", "buffered", "truncated", "length"]
)
def test_stopped_sequences_retain_every_planned_repeat(
    tmp_path: Path, provider: Provider, service: Server, fault: str
) -> None:
    provider.fault = fault
    result = run_cli(service=service, value=document(root=tmp_path, provider=provider))
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    probe = report["summary"]["targets"][0]["cache_observation"]
    assert report["status"] == "failed"
    assert probe["execution_complete"] is False
    assert len(provider.requests) == 2
    assert len(probe["repeats"]) == 6
    assert probe["undispatched_repeats"] == 4
    assert all(row["status"] == "undispatched" for row in probe["repeats"][2:])
    assert probe["cached_prompt_ratio"] is None
    assert probe["validation"]["status"] != "pass"
    assert probe["repeats"][1]["response_status"] == (
        None
        if fault == "connection"
        else HTTPStatus.SERVICE_UNAVAILABLE if fault == "http" else HTTPStatus.OK
    )
    if fault == "length":
        assert probe["repeats"][1]["output_complete"] is False
    else:
        assert probe["repeats"][1]["status"] == "failed"


def test_cancellation_keeps_the_inflight_and_undispatched_repeats(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    provider.fault = "cancel"
    result = run_cli(
        service=service,
        value=document(root=tmp_path, provider=provider),
        detach=True,
    )
    assert result.returncode == 0, result.stderr
    run_id = json.loads(result.stdout)["id"]
    assert provider.entered.wait(timeout=5)
    cancelled = benchmark_cli(service=service, args=["cancel", run_id])
    assert cancelled.returncode == 0, cancelled.stderr
    provider.release.set()
    assert wait_run(service.store, run_id)["status"] == "cancelled"
    report_result = benchmark_cli(service=service, args=["report", run_id])
    assert report_result.returncode == 0
    probe = json.loads(report_result.stdout)["summary"]["targets"][0][
        "cache_observation"
    ]
    assert len(provider.requests) == 2
    assert probe["repeats"][1]["status"] == "cancelled"
    assert probe["undispatched_repeats"] == 4
    assert probe["cached_prompt_ratio"] is None
    assert probe["validation"]["status"] == "incomplete"


def test_plan_rejects_accidental_default_generation_settings(
    tmp_path: Path, provider: Provider
) -> None:
    value = document(root=tmp_path, provider=provider)
    value.pop("sampling")
    with pytest.raises(ValueError, match="native request settings"):
        plan(value)
    assert provider.requests == []


@pytest.mark.parametrize(
    ("raw", "qualification"),
    [
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 40},
                "prompt_cache_hit_tokens": 41,
            },
            "conflicting",
        ),
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "prompt_cache_hit_tokens": "40",
            },
            "malformed",
        ),
        (
            {"prompt_tokens": 100, "prompt_cache_hit_tokens": 40},
            "incomplete",
        ),
        (
            {"prompt_tokens": 100, "completion_tokens": 3, "cached_tokens": 0},
            "unqualified",
        ),
    ],
)
def test_successful_http_cannot_qualify_bad_cache_evidence(
    tmp_path: Path,
    provider: Provider,
    service: Server,
    raw: dict[str, Any],
    qualification: str,
) -> None:
    provider.usage_events = [raw]
    value = document(root=tmp_path, provider=provider, repeats=1)
    value["benchmark_options"] = {
        cache_probe.BENCHMARK_ID: {
            "thresholds": {"min_cached_token_reporting": "reported_zero"}
        }
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    probe = report["summary"]["targets"][0]["cache_observation"]
    assert probe["execution_complete"] is True
    assert (
        probe["repeats"][0]["provider_cache_observation"]["qualification"]
        == qualification
    )
    assert probe["validation"]["predicates"]["min_success_rate"]["status"] == "pass"
    assert probe["validation"]["status"] == "incomplete"
    assert probe["cached_prompt_ratio"] is None
    call = service.store.calls(report["run_id"])[0]
    assert call["raw_usage"] == raw
    assert call["usage"] == normalize_usage(raw)
    detail = benchmark_cli(
        service=service, args=["show", report["run_id"], "--call-id", call["id"]]
    )
    assert detail.returncode == 0, detail.stderr
    assert json.loads(detail.stdout)["raw_usage"] == raw


@pytest.mark.parametrize("missing", ["all", "prompt", "output", "cache"])
def test_missing_usage_does_not_become_zero_cache_tokens(
    tmp_path: Path, provider: Provider, service: Server, missing: str
) -> None:
    raw = usage()
    if missing == "all":
        provider.usage_events = []
    else:
        raw.pop(
            {
                "prompt": "prompt_tokens",
                "output": "completion_tokens",
                "cache": "prompt_tokens_details",
            }[missing]
        )
        provider.usage_events = [raw]
    result = run_cli(
        service=service, value=document(root=tmp_path, provider=provider, repeats=1)
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    probe = report["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_prompt_ratio"] is None
    assert probe["cached_tokens"] is None
    assert probe["validation"]["predicates"]["min_success_rate"]["status"] == "pass"
    if missing != "cache":
        assert service.store.calls(report["run_id"])[0]["usage"] is None


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.333333, "pass"), (0.333334, "fail")],
)
def test_native_ratio_rounding_controls_threshold_edges(
    tmp_path: Path,
    provider: Provider,
    service: Server,
    threshold: float,
    expected: str,
) -> None:
    raw = usage(cached=1)
    raw["prompt_tokens"] = 3
    provider.usage_events = [raw]
    native_result = native_cli(
        provider=provider,
        output=tmp_path / "native",
        repeats=1,
        extra_args=["--min-cached-prompt-ratio", str(threshold)],
    )
    assert native_result.returncode == (0 if expected == "pass" else 1)
    provider.requests.clear()
    value = document(root=tmp_path, provider=provider, repeats=1)
    value["benchmark_options"] = {
        cache_probe.BENCHMARK_ID: {"thresholds": {"min_cached_prompt_ratio": threshold}}
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_prompt_ratio"] == 0.333333
    assert probe["validation"]["status"] == expected


@pytest.mark.parametrize(
    ("threshold", "expected"), [(0.6667, "pass"), (0.6668, "fail")]
)
def test_native_field_rate_rounding_does_not_fill_a_missing_cache_bucket(
    tmp_path: Path,
    provider: Provider,
    service: Server,
    threshold: float,
    expected: str,
) -> None:
    provider.usage_by_request = [[usage()], [usage()], [usage(cached=None)]]
    native_result = native_cli(
        provider=provider,
        output=tmp_path / "native",
        repeats=3,
        extra_args=["--min-cached-token-field-rate", str(threshold)],
    )
    assert native_result.returncode == (0 if expected == "pass" else 1)
    provider.requests.clear()
    value = document(root=tmp_path, provider=provider, repeats=3)
    value["benchmark_options"] = {
        cache_probe.BENCHMARK_ID: {
            "thresholds": {"min_cached_token_field_rate": threshold}
        }
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_token_field_rate"] == 0.6667
    assert probe["cached_prompt_ratio"] is None
    assert probe["validation"]["status"] == expected


@pytest.mark.parametrize("later_omission", [False, True])
def test_stream_snapshots_and_duplicate_aliases_are_not_added(
    tmp_path: Path, provider: Provider, service: Server, later_omission: bool
) -> None:
    raw = usage(cached=40)
    raw["prompt_tokens_details"]["cache_write_tokens"] = 10
    raw["prompt_cache_hit_tokens"] = 40
    raw["prompt_cache_miss_tokens"] = 60
    provider.usage_events = [raw, usage(cached=None) if later_omission else raw]
    result = run_cli(
        service=service, value=document(root=tmp_path, provider=provider, repeats=1)
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    probe = report["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_tokens"] == 40
    assert probe["prompt_tokens"] == 100
    assert probe["cached_prompt_ratio"] == 0.4
    observation = probe["repeats"][0]["provider_cache_observation"]
    assert observation["source"] == "usage.prompt_tokens_details.cached_tokens"
    assert observation["usage_event_index"] == (0 if later_omission else 1)
    assert service.store.calls(report["run_id"])[0]["usage"] == normalize_usage(
        provider.usage_events[-1]
    )
    assert "cache_write_tokens" in observation["usage"]


def test_later_valid_event_does_not_hide_an_invalid_alias(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    provider.usage_events = [usage(source="deepseek", cached="40"), usage(cached=40)]
    value = document(root=tmp_path, provider=provider, repeats=1)
    value["benchmark_options"] = {
        cache_probe.BENCHMARK_ID: {"thresholds": {"min_cached_prompt_ratio": 0.1}}
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)["summary"]["targets"][0]["cache_observation"]
    assert probe["cached_prompt_ratio"] is None
    assert probe["validation"]["status"] == "incomplete"
    assert probe["repeats"][0]["provider_cache_observation"]["event_issues"] == {
        "malformed": 1
    }


@pytest.mark.parametrize("fault", [None, "http"])
def test_router_then_backend_preserves_arm_order_and_stopped_baseline(
    tmp_path: Path, provider: Provider, service: Server, fault: str | None
) -> None:
    with running_provider() as backend:
        value = document(root=tmp_path, provider=provider, repeats=3)
        value["targets"][0].update(
            id="router", kind="mom", config_hash="fixture-config"
        )
        value["targets"].append(
            {
                "id": "direct-backend",
                "kind": "single",
                "model": "requested-model",
                "base_url": backend.base_url,
                "session_mode": SESSION_AWARE,
            }
        )
        provider.fault = fault
        result = run_cli(service=service, value=value)
        assert result.returncode == (2 if fault else 0), result.stderr
        report = json.loads(result.stdout)
        router, direct = [
            target["cache_observation"] for target in report["summary"]["targets"]
        ]
        assert router["planned_repeats"] == direct["planned_repeats"] == 3
        if fault:
            assert backend.requests == []
            assert direct["undispatched_repeats"] == 3
            assert direct["cached_token_reporting"] is None
            assert direct["cached_prompt_ratio"] is None
        else:
            assert len(provider.requests) == len(backend.requests) == 3
            assert (
                provider.requests[0]["session_id"] != backend.requests[0]["session_id"]
            )
            assert router["cached_prompt_ratio"] == direct["cached_prompt_ratio"] == 0.4
            calls = service.store.calls(report["run_id"])
            assert [call["target_id"] for call in calls] == ["router"] * 3 + [
                "direct-backend"
            ] * 3
            assert all(
                row["config_hash"] == "fixture-config" for row in router["repeats"]
            )
        assert "do not prove model equivalence" in " ".join(report["limitations"])


def test_unobserved_model_is_not_the_requested_model(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    provider.omit_model = True
    result = run_cli(
        service=service, value=document(root=tmp_path, provider=provider, repeats=1)
    )
    assert result.returncode == 0, result.stderr
    target = json.loads(result.stdout)["summary"]["targets"][0]
    assert target["selected_models"] == {}
    assert target["cache_observation"]["target"]["requested_model"] == "requested-model"
    assert target["cache_observation"]["repeats"][0]["selected_model"] is None


def test_router_config_acknowledgement_is_not_inferred_from_configuration(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    value = document(root=tmp_path, provider=provider, repeats=1)
    value["targets"][0].update(kind="mom", config_hash="expected-config")
    provider.config_hash = None
    result = run_cli(service=service, value=value)
    assert result.returncode == 2, result.stderr
    probe = json.loads(result.stdout)["summary"]["targets"][0]["cache_observation"]
    assert probe["target"]["expected_config_hash"] == "expected-config"
    assert probe["repeats"][0]["config_hash"] is None
    assert probe["execution_complete"] is False


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5, 33, 10**12])
def test_preparation_bounds_the_native_sequence(repeats: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeats"):
        cache_probe.prepare(store=tmp_path, repeats=repeats)


def test_preparation_and_copy_only_discovery_do_not_need_bench(
    tmp_path: Path,
) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.sr_bench.cache_probe",
            "--store",
            str(tmp_path / "data"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    dataset = json.loads(prepared.stdout)
    case = json.loads(Path(dataset["path"]).read_text())
    assert case["metadata"]["cache_probe"]["repeats"] == 6
    assert case["metadata"]["cache_probe"]["stable_prefix_sha256"] == PREFIX_SHA256
    assert (
        case["metadata"]["cache_probe"]["native_prefix_sha256"]
        == native.probe_prompt_hash()
    )
    assert case["metadata"]["cache_probe"]["request_settings"] == {
        "max_tokens": 16,
        "temperature": 0.0,
        "timeout": 120.0,
    }
    copied = tmp_path / "copied"
    shutil.copytree(
        ROOT / "src/vllm-sr/cli",
        copied / "cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    discovered = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cli.sr_bench.adapters import get_adapter; "
            "print(get_adapter('cache-token-probe-v1').result_kind)",
        ],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(copied)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert discovered.returncode == 0, discovered.stderr
    assert discovered.stdout.strip() == CACHE_OBSERVATION_RESULT


def test_mixed_observations_have_no_combined_quality_or_binary_only_exports(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    value = plan(document(root=tmp_path, provider=provider, repeats=1))
    for key in ("dataset", "adapter_versions", "plan_sha256"):
        value.pop(key)
    value["cases"].append(
        {
            "id": "binary",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": "Return A"}],
            "answer": "A",
        }
    )
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    target = report["summary"]["targets"][0]
    assert target["result_kind"] == "mixed"
    assert target["scored"] == target["observed"] == 1
    assert target["complete"] is True
    assert "cache_observation" not in target
    assert all(
        target[field] is None
        for field in (
            "correct",
            "accuracy",
            "accuracy_ci95",
            "macro_accuracy",
            "sr_bench_score",
        )
    )
    by_benchmark = {row["benchmark"]: row for row in report["benchmarks"]}
    assert by_benchmark["mmlu-pro"]["accuracy"] == 1
    assert (
        by_benchmark[cache_probe.BENCHMARK_ID]["cache_observation"]["planned_repeats"]
        == 1
    )
    run_id = report["run_id"]
    for operation in (
        lambda: compare(service.store, run_id, run_id),
        lambda: replay(service.store, run_id, run_id),
        lambda: regrade(service.store, run_id),
        lambda: export_training(service.store, run_id),
    ):
        with pytest.raises(ValueError, match="binary results only"):
            operation()


def test_readiness_requires_local_preparation_and_reuses_the_frozen_source(
    tmp_path: Path,
) -> None:
    reader = DatasetReader(tmp_path)
    before = reader.selection(profile="quick")
    entry = next(
        row for row in before["benchmarks"] if row["id"] == cache_probe.BENCHMARK_ID
    )
    assert before["model_requests"] == 0
    assert entry["eligible"] is False
    assert entry["reason_code"] == "local_preparation_required"
    prepared = cache_probe.prepare(store=tmp_path)
    assert prepared == cache_probe.prepare(store=tmp_path)
    selected = reader.selection(profile="quick")
    entry = next(
        row for row in selected["benchmarks"] if row["id"] == cache_probe.BENCHMARK_ID
    )
    assert entry["eligible"] is False
    assert entry["reason_code"] == "local_preparation_required"
    detail = reader.detail(prepared["id"])
    assert detail["custom_subset"] is True
    assert detail["case_count"] == 1
    assert reader.page(prepared["id"])["cases"][0]["messages"] == cache_probe.messages(
        repeat_index=0
    )


@pytest.mark.parametrize(
    "change",
    [
        "prefix",
        "source",
        "repeat_budget",
        "concurrency",
        "stateless",
        "request_params",
        "timeout",
        "answer",
        "result_kind",
        "threshold",
    ],
)
def test_preflight_rejects_workload_and_contract_drift_before_dispatch(
    tmp_path: Path, provider: Provider, change: str
) -> None:
    value = document(root=tmp_path, provider=provider)
    prepared = value.pop("dataset")
    value["cases"] = [json.loads(Path(prepared["path"]).read_text())]
    case = value["cases"][0]
    if change == "prefix":
        case["messages"][1]["content"] += "Different prompt."
    elif change == "source":
        case["metadata"]["cache_probe"]["source_revision"] = "not-the-native-source"
    elif change == "repeat_budget":
        value["limits"]["max_calls_per_case"] = 5
    elif change == "concurrency":
        value["limits"]["concurrency"] = 2
    elif change == "stateless":
        value["targets"][0]["session_mode"] = "stateless"
    elif change == "request_params":
        value["targets"][0]["request_params"] = {"top_p": 0.8}
    elif change == "timeout":
        value["limits"]["total_timeout_s"] = 4
    elif change == "answer":
        case["answer"] = "A"
    elif change == "result_kind":
        value["result_kinds"] = {cache_probe.BENCHMARK_ID: "binary"}
    else:
        value["benchmark_options"] = {
            cache_probe.BENCHMARK_ID: {"thresholds": {"min_success_rate": float("nan")}}
        }
    with pytest.raises(ValueError):
        plan(value)
    assert provider.requests == []
