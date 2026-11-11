from __future__ import annotations

import importlib.util
import json
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
from cli.sr_bench import session_routing
from cli.sr_bench.contracts import SESSION_AWARE, plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.grading import SESSION_OBSERVATION_RESULT, validate_result
from cli.sr_bench.report import make_report
from cli.sr_bench.store import Store
from cli.sr_bench.transport import SESSION_ID_HEADER
from cli.vsr_trace import TRACE_HEADER
from test_sr_bench import wait_run

ROOT: Path = Path(__file__).resolve().parents[3]
NATIVE_SPEC = importlib.util.spec_from_file_location(
    "bench.native_live_benchmark", ROOT / "bench/agentic_routing_live_benchmark.py"
)
assert NATIVE_SPEC is not None and NATIVE_SPEC.loader is not None
native = importlib.util.module_from_spec(NATIVE_SPEC)
sys.modules[NATIVE_SPEC.name] = native
NATIVE_SPEC.loader.exec_module(native)
SCENARIOS: tuple[str, ...] = tuple(native.scenario_names())


class Provider(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict[str, Any]] = []
        self.fault_at: int | None = None
        self.include_usage: bool = True
        self.finish_reason: str = "stop"
        self.selected_models: list[str] | None = None

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
        provider.requests.append(
            {
                "body": body,
                "session_id": self.headers.get(SESSION_ID_HEADER),
            }
        )
        if index == provider.fault_at:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "text/event-stream" if body.get("stream") is True else "application/json",
        )
        selected_model = (
            self.server.selected_models[index]
            if self.server.selected_models is not None
            else "router-model"
        )
        self.send_header("x-vsr-selected-model", selected_model)
        self.send_header("x-vsr-selected-decision", "fixture-decision")
        self.send_header(TRACE_HEADER, "session_phase=provider_state")
        self.end_headers()
        if body.get("stream") is not True:
            response = {
                "id": f"response-{index}",
                "model": "provider-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "Observed response."},
                        "finish_reason": self.server.finish_reason,
                    }
                ],
            }
            if self.server.include_usage:
                response["usage"] = {"prompt_tokens": 12, "completion_tokens": 2}
            self.wfile.write(json.dumps(response).encode())
            return
        events: list[dict[str, Any]] = [
            {
                "model": "provider-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Observed response."},
                        "finish_reason": self.server.finish_reason,
                    }
                ],
            }
        ]
        if provider.include_usage:
            events.append(
                {
                    "model": "provider-model",
                    "choices": [],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                }
            )
        with suppress(BrokenPipeError, ConnectionResetError):
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


@contextmanager
def running_provider() -> Iterator[Provider]:
    server = Provider()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def provider() -> Iterator[Provider]:
    with running_provider() as server:
        yield server


def manifest(
    *, store_path: Path, provider: Provider, scenarios: tuple[str, ...] = SCENARIOS
) -> dict[str, Any]:
    dataset = session_routing.prepare(
        store=store_path,
        scenarios=scenarios,
        sessions=1,
        turns=6,
        max_tokens=16,
        temperature=0.0,
        timeout=3.0,
        turn_delay_seconds=0.0,
        idle_pause_seconds=0.0,
        profile="quick",
    )
    return {
        "version": "sr-bench-1.0",
        "profile": "quick",
        "seed": 0,
        "cost_policy": "capability_only",
        "dataset": dataset,
        "targets": [
            {
                "id": "subject",
                "kind": "single",
                "model": "requested-model",
                "base_url": provider.base_url,
                "session_mode": SESSION_AWARE,
            }
        ],
        "sampling": {"temperature": 0.0, "max_tokens": 16},
        "limits": {
            "max_calls_per_case": 6,
            "total_timeout_s": 3,
            "idle_timeout_s": 2,
            "case_timeout_s": 20,
            "max_run_seconds": 30,
        },
    }


def execute(*, store: Store, provider: Provider, root: Path) -> dict[str, Any]:
    frozen = plan(manifest(store_path=root, provider=provider))
    engine = Engine(store)
    run = engine.start(frozen)
    wait_engine_run(
        engine=engine,
        store=store,
        run_id=run["id"],
        expected_status="completed",
    )
    return make_report(store, run["id"])


def wait_engine_run(
    *, engine: Engine, store: Store, run_id: str, expected_status: str
) -> dict[str, Any]:
    try:
        run = wait_run(store, run_id)
    finally:
        worker = engine.threads[run_id]
        worker.join(timeout=35)
        assert not worker.is_alive()
    assert run["status"] == expected_status
    return run


def native_requests(
    *, provider: Provider, output: Path, scenario: str
) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/agentic_routing_live_benchmark.py"),
            "--base-url",
            provider.base_url,
            "--scenario",
            scenario,
            "--sessions",
            "1",
            "--turns",
            "6",
            "--max-tokens",
            "16",
            "--timeout",
            "3",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    requests = list(provider.requests)
    provider.requests.clear()
    return requests


def test_prepared_turns_match_all_native_schedules_and_messages(
    tmp_path: Path, provider: Provider
) -> None:
    dataset = session_routing.prepare(
        store=tmp_path,
        scenarios=SCENARIOS,
        sessions=2,
        turns=8,
        max_tokens=16,
        temperature=0.0,
        timeout=3.0,
        turn_delay_seconds=0.25,
        idle_pause_seconds=0.5,
        profile="quick",
    )
    cases = [
        json.loads(line)
        for line in Path(dataset["path"]).read_text().splitlines()
        if line.strip()
    ]
    assert len(cases) == len(SCENARIOS) * 2
    for case in cases:
        spec = case["metadata"]["session_routing"]
        session_index = spec["session_index"]
        session_id = f"live-router-{spec['scenario']}-{session_index:04d}"
        assert len(spec["turns"]) == 8
        for turn, saved in enumerate(spec["turns"]):
            phase = native.phase_for_turn(spec["scenario"], turn)
            prompt = native.prompt_for_phase(
                spec["scenario"], session_index, turn, phase
            )
            native_plan = native.TurnPlan(
                session_id=session_id,
                turn=turn,
                phase=phase,
                prompt=prompt,
                previous_response_id="",
            )
            assert saved["phase"] == phase
            assert saved["messages"] == native.build_messages(native_plan)
            assert saved["pause_seconds"] == (
                0.5 if phase == "idle_boundary" else 0.25 if turn > 0 else 0.0
            )


def test_live_adapter_uses_one_sr_bench_session_and_reports_observations(
    tmp_path: Path, provider: Provider
) -> None:
    native_by_scenario = {
        scenario: native_requests(
            provider=provider, output=tmp_path / "native" / scenario, scenario=scenario
        )
        for scenario in SCENARIOS
    }
    store = Store(tmp_path / "journal")
    try:
        report = execute(store=store, provider=provider, root=tmp_path / "prepared")
        target = report["summary"]["targets"][0]
        observation = target["session_observation"]
        assert target["result_kind"] == SESSION_OBSERVATION_RESULT
        assert target["scored"] == 0
        assert target["observed"] == len(SCENARIOS)
        assert target["accuracy"] is None
        assert target["macro_accuracy"] is None
        assert target["sr_bench_score"] is None
        assert observation["execution_complete"] is True
        assert observation["planned_sessions"] == len(SCENARIOS)
        assert observation["planned_turns"] == len(SCENARIOS) * 6
        assert observation["completed_turns"] == len(provider.requests)
        assert observation["success_rate"] == 1.0
        assert observation["selected_model_switches"] == 0
        assert (
            observation["session_recovery"]["error_recovery_status"] == "not_measured"
        )
        assert observation["session_recovery"]["idle_resumption"]["planned"] == 1
        assert len(observation["turns"]) == len(provider.requests)
        assert observation["turns"][0]["request_phase"] == "user_turn"
        assert observation["turns"][0]["previous_response_id_sent"] is False
        assert observation["turns"][0]["router_phase"] == "provider_state"
        assert observation["turns"][0]["phase_source"] == "router"
        assert (
            observation["phase_metrics"]["user_turn"]["request_phase_counts"][
                "user_turn"
            ]
            > 0
        )
        assert len({request["session_id"] for request in provider.requests}) == len(
            SCENARIOS
        )
        assert all(
            request["body"]["stream"] is True
            and request["body"]["stream_options"] == {"include_usage": True}
            and "previous_response_id" not in request["body"]
            for request in provider.requests
        )
        tool_request = next(
            request["body"]
            for request in provider.requests
            if any(
                message.get("role") == "tool" for message in request["body"]["messages"]
            )
        )
        assert tool_request["messages"][-2]["role"] == "tool"
        assert all(
            turn["benchmark_phase"] != turn["router_phase"]
            for turn in observation["turns"]
            if turn["benchmark_phase"] == "tool_loop"
        )
        assert observation["turns"][0]["usage"]["input_tokens"] == 12
        for index, (_scenario, native_rows) in enumerate(native_by_scenario.items()):
            assert len(native_rows) == 6
            assert all(
                request["body"]["max_tokens"] == 16
                and request["body"]["temperature"] == 0.0
                and "previous_response_id" not in request["body"]
                for request in native_rows
            )
            routed_messages = [
                request["body"]["messages"]
                for request in provider.requests[index * 6 : (index + 1) * 6]
            ]
            assert [
                request["body"]["messages"] for request in native_rows
            ] == routed_messages
        calls = store.calls(report["run_id"])
        assert all(
            "previous_response_id" not in call["request"]["effective_body"]
            for call in calls
        )
    finally:
        store.db.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("session_mode", "session_aware"),
        ("call_budget", "call or timeout limits"),
        ("responses_state", "Responses state"),
        ("recovery", "Responses state"),
    ],
)
def test_preflight_rejects_unsupported_controls(
    tmp_path: Path, provider: Provider, change: str, message: str
) -> None:
    value = manifest(
        store_path=tmp_path / "prepared",
        provider=provider,
        scenarios=("balanced",),
    )
    if change == "session_mode":
        value["targets"][0]["session_mode"] = "stateless"
    elif change == "call_budget":
        value["limits"]["max_calls_per_case"] = 5
    elif change == "responses_state":
        value["benchmark_options"] = {
            session_routing.BENCHMARK_ID: {"responses_state": True}
        }
    else:
        value["benchmark_options"] = {
            session_routing.BENCHMARK_ID: {"recovery": {"enabled": True}}
        }
    with pytest.raises(ValueError, match=message):
        plan(value)


def test_missing_usage_stays_unknown_and_failed_turn_keeps_planned_denominator(
    tmp_path: Path, provider: Provider
) -> None:
    provider.include_usage = False
    provider.fault_at = 1
    store = Store(tmp_path / "journal")
    try:
        value = manifest(
            store_path=tmp_path / "prepared",
            provider=provider,
            scenarios=("balanced",),
        )
        value["benchmark_options"] = {
            session_routing.BENCHMARK_ID: {"thresholds": {"min_success_rate": 0.9}}
        }
        frozen = plan(value)
        engine = Engine(store)
        run = engine.start(frozen)
        wait_engine_run(
            engine=engine,
            store=store,
            run_id=run["id"],
            expected_status="failed",
        )
        report = make_report(store, run["id"])
        observation = report["summary"]["targets"][0]["session_observation"]
        assert observation["planned_turns"] == 6
        assert observation["attempted_turns"] == 2
        assert observation["completed_turns"] == 1
        assert observation["failed_turns"] == 1
        assert observation["success_rate"] == 0.1667
        assert observation["usage_complete"] is False
        assert observation["turns"][0]["usage"] is None
        assert (
            observation["session_recovery"]["error_recovery_status"] == "not_measured"
        )
        assert observation["turns"][0]["previous_response_id_sent"] is False
        assert observation["validation"]["status"] == "fail"
        assert report["summary"]["targets"][0]["correct"] is None
        assert len(provider.requests) == 2
    finally:
        store.db.close()


def test_output_truncation_fails_closed_and_keeps_all_turns_planned(
    tmp_path: Path, provider: Provider
) -> None:
    provider.finish_reason = "length"
    store = Store(tmp_path / "journal")
    try:
        frozen = plan(
            manifest(
                store_path=tmp_path / "prepared",
                provider=provider,
                scenarios=("balanced",),
            )
        )
        engine = Engine(store)
        run = engine.start(frozen)
        wait_engine_run(
            engine=engine,
            store=store,
            run_id=run["id"],
            expected_status="failed",
        )
        observation = make_report(store, run["id"])["summary"]["targets"][0][
            "session_observation"
        ]
        assert observation["planned_turns"] == 6
        assert observation["attempted_turns"] == 1
        assert observation["completed_turns"] == 0
        assert observation["turns"][0]["output_complete"] is False
        assert observation["execution_complete"] is False
    finally:
        store.db.close()


def test_pause_cancellation_stops_before_dispatching_the_next_turn(
    tmp_path: Path,
) -> None:
    dataset = session_routing.prepare(
        store=tmp_path,
        scenarios=("balanced",),
        sessions=1,
        turns=2,
        max_tokens=16,
        temperature=0.0,
        timeout=3.0,
        turn_delay_seconds=1.0,
        idle_pause_seconds=0.0,
        profile="quick",
    )
    case = json.loads(Path(dataset["path"]).read_text().splitlines()[0])

    class Context:
        def __init__(self) -> None:
            self.calls = 0

        def cancelled(self) -> bool:
            return self.calls > 0

        def call(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            return {"output_complete": True}

    context = Context()
    with pytest.raises(session_routing.CallFailure, match="cancelled"):
        session_routing.execute(case=case, context=context)
    assert context.calls == 1


def test_session_observation_result_has_no_answer_or_quality_score() -> None:
    result = {
        "answer": None,
        "correct": None,
        "score": None,
        "details": {
            "session_observation": {
                "version": SESSION_OBSERVATION_RESULT,
                "planned_turns": 2,
                "completed_turns": 2,
            }
        },
    }
    validate_result(kind=SESSION_OBSERVATION_RESULT, result=result)
    for field in ("answer", "correct", "score"):
        invalid = {**result, field: "answer" if field == "answer" else 1}
        with pytest.raises(ValueError):
            validate_result(kind=SESSION_OBSERVATION_RESULT, result=invalid)
