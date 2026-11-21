"""Agent-session replay benchmark: fixture validation, preparation and replay."""

from __future__ import annotations

import copy
import itertools
import json
import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench import agent_session
from cli.sr_bench.contracts import SESSION_AWARE, plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.report import make_report
from cli.sr_bench.sources import prepare_dataset
from cli.sr_bench.store import TERMINAL, Store
from cli.sr_bench.transport import SESSION_ID_HEADER

Mutation = Callable[[dict[str, Any]], None]

REPO_FIXTURE = (
    Path(__file__).resolve().parents[3] / "bench/data/coding_agent_session.v1.json"
)


def _fixture() -> dict[str, Any]:
    """Smallest session with every phase: user_turn, tool_loop, follow_up."""
    return {
        "schema_version": "agent-session-fixture.v1",
        "id": "fixture-session",
        "description": "Synthetic coding-agent session.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "turns": [
            {
                "id": "request",
                "phase": "user_turn",
                "append_messages": [
                    {"role": "system", "content": "You fix failing tests."},
                    {"role": "user", "content": "Fix the failing test."},
                ],
            },
            {
                "id": "tool-result",
                "phase": "tool_loop",
                "append_messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "a.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "def f(): ...",
                    },
                ],
            },
            {
                "id": "follow-up",
                "phase": "follow_up",
                "append_messages": [
                    {"role": "assistant", "content": "The fix is ready."},
                    {"role": "user", "content": "Run the tests again."},
                ],
            },
        ],
    }


def _turn_messages(data: dict[str, Any], turn_id: str) -> list[dict[str, Any]]:
    turn = next(turn for turn in data["turns"] if turn["id"] == turn_id)
    return turn["append_messages"]


def _set(path: tuple[Any, ...], value: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _rename_tool_call(data: dict[str, Any]) -> None:
    _turn_messages(data, "tool-result")[0]["tool_calls"][0]["function"][
        "name"
    ] = "delete_repository"


def _break_arguments(data: dict[str, Any]) -> None:
    _turn_messages(data, "tool-result")[0]["tool_calls"][0]["function"][
        "arguments"
    ] = "a.py"


def _orphan_result(data: dict[str, Any]) -> None:
    _turn_messages(data, "tool-result")[1]["tool_call_id"] = "call_unknown"


def _user_before_result(data: dict[str, Any]) -> None:
    _turn_messages(data, "tool-result").insert(
        1, {"role": "user", "content": "still there?"}
    )


def _end_on_assistant(data: dict[str, Any]) -> None:
    _turn_messages(data, "follow-up").pop()


def _duplicate_tool(data: dict[str, Any]) -> None:
    data["tools"].append(copy.deepcopy(data["tools"][0]))


def _second_system(data: dict[str, Any]) -> None:
    _turn_messages(data, "follow-up").insert(
        0, {"role": "system", "content": "New rules."}
    )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (_set(("schema_version",), "v0"), "schema_version"),
        (_set(("source",), "trace"), "unknown keys"),
        (_duplicate_tool, "duplicate tool name"),
        (_rename_tool_call, "not in the tool catalog"),
        (_break_arguments, "arguments must be a JSON object"),
        (_orphan_result, "does not answer a pending call"),
        (_user_before_result, "arrives before results"),
        (_end_on_assistant, "must end with a user message or tool result"),
        (_set(("turns", 0, "phase"), "tool_loop"), "tool_loop turn only if"),
        (_second_system, "only system message"),
    ],
    ids=[
        "schema-version",
        "unknown-key",
        "duplicate-tool",
        "unknown-tool",
        "non-object-arguments",
        "orphan-tool-result",
        "user-before-tool-result",
        "ends-on-assistant",
        "mislabeled-tool-loop",
        "second-system-message",
    ],
)
def test_parse_session_rejects_malformed_sessions(mutate: Mutation, error: str) -> None:
    data = _fixture()
    mutate(data)
    with pytest.raises(agent_session.FixtureError, match=error):
        agent_session.parse_session(data)


def test_prepare_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text('{"id": "a", "id": "b"}', encoding="utf-8")
    with pytest.raises(agent_session.FixtureError, match="duplicate key 'id'"):
        agent_session.prepare(
            benchmark="agent-session-replay",
            profile="quick",
            store=tmp_path,
            source_path=path,
        )


def test_prepare_requires_a_source_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--source-path"):
        prepare_dataset(
            benchmark="agent-session-replay", profile="quick", store=tmp_path
        )


def test_prepare_freezes_one_replay_case(tmp_path: Path) -> None:
    """The dispatched local prepare publishes one case with cumulative turns."""
    source = tmp_path / "session.json"
    source.write_text(json.dumps(_fixture()), encoding="utf-8")
    manifest = prepare_dataset(
        benchmark="agent-session-replay",
        profile="quick",
        store=tmp_path,
        source_path=source,
    )
    assert manifest["case_count"] == 1
    assert manifest["benchmarks"] == ["agent-session-replay"]
    assert manifest["split"] == "dev"
    case = json.loads(Path(manifest["path"]).read_text().strip())
    assert case["id"] == "fixture-session"
    assert case["benchmark"] == "agent-session-replay"
    assert [turn["message_count"] for turn in case["metadata"]["turns"]] == [2, 4, 6]
    assert [turn["phase"] for turn in case["metadata"]["turns"]] == [
        "user_turn",
        "tool_loop",
        "follow_up",
    ]
    assert len(case["messages"]) == 6
    assert len(case["tools"]) == 2


@pytest.mark.skipif(not REPO_FIXTURE.is_file(), reason="repository fixture not present")
def test_prepare_accepts_the_repository_fixture(tmp_path: Path) -> None:
    manifest = prepare_dataset(
        benchmark="agent-session-replay",
        profile="quick",
        store=tmp_path,
        source_path=REPO_FIXTURE,
    )
    case = json.loads(Path(manifest["path"]).read_text().strip())
    assert case["id"] == "coding-agent-fix-failing-test"
    assert [turn["phase"] for turn in case["metadata"]["turns"]] == [
        "user_turn",
        "tool_loop",
        "follow_up",
    ]


def _case() -> dict[str, Any]:
    return agent_session.build_case(agent_session.parse_session(_fixture()))


def test_case_turns_reject_uncovered_transcript() -> None:
    case = _case()
    case["messages"].append({"role": "user", "content": "extra"})
    with pytest.raises(ValueError, match="cover the recorded transcript"):
        agent_session.case_turns(case)


class _Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        model = self.server.models[len(self.server.seen) % len(self.server.models)]
        self.server.seen.append((self.headers.get(SESSION_ID_HEADER), body))
        if (
            self.server.fail_after is not None
            and len(self.server.seen) > self.server.fail_after
        ):
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in (
            {
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "ok"},
                        "finish_reason": self.server.finish_reason,
                    }
                ],
            },
            {
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def endpoint() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.seen = []
    server.models = ["small", "large", "small"]
    server.finish_reason = "stop"
    server.fail_after = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _manifest(endpoint: ThreadingHTTPServer, **target: Any) -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "cost_policy": "capability_only",
        "targets": [
            {
                "id": "router",
                "kind": "single",
                "model": "auto",
                "base_url": f"http://127.0.0.1:{endpoint.server_port}/v1",
                "session_mode": SESSION_AWARE,
                **target,
            }
        ],
        "cases": [_case()],
        "limits": {"total_timeout_s": 2, "idle_timeout_s": 1, "max_run_seconds": 10},
    }


def test_plan_rejects_stateless_targets(endpoint: ThreadingHTTPServer) -> None:
    document = _manifest(endpoint)
    del document["targets"][0]["session_mode"]
    with pytest.raises(ValueError, match="session_aware"):
        plan(document)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_replay_sends_each_turn_in_one_session(
    tmp_path: Path, endpoint: ThreadingHTTPServer, finish_reason: str
) -> None:
    """One replay is one session: cumulative prefixes, tools and one identity."""
    endpoint.finish_reason = finish_reason
    engine = Engine(Store(tmp_path))
    run_id = engine.start(_manifest(endpoint))["id"]
    deadline = time.monotonic() + 5
    while (status := engine.store.get(run_id)["status"]) not in TERMINAL:
        assert time.monotonic() < deadline, "run did not terminate"
        time.sleep(0.01)
    assert status == "completed"

    assert len(endpoint.seen) == 3
    assert len({session for session, _ in endpoint.seen}) == 1
    bodies = [body for _, body in endpoint.seen]
    assert [len(body["messages"]) for body in bodies] == [2, 4, 6]
    assert all(len(body["tools"]) == 2 for body in bodies)
    for earlier, later in itertools.pairwise(bodies):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]

    calls = engine.store.calls(run_id, summary=True)
    assert [call.get("fixture_phase") for call in calls] == [
        "user_turn",
        "tool_loop",
        "follow_up",
    ]
    assert [call["finish_reason"] for call in calls] == [finish_reason] * 3
    assert all(call["output_complete"] == (finish_reason == "stop") for call in calls)
    result = engine.store.results(run_id)[0]
    assert result["correct"] is (finish_reason == "stop")
    if finish_reason == "length":
        assert result["details"]["quality_failure"] == "output_limit"
    assert [turn["selected_model"] for turn in result["details"]["turns"]] == [
        "small",
        "large",
        "small",
    ]


@pytest.mark.parametrize(
    ("max_calls", "fail_after", "expected_requests", "error"),
    [
        (1, None, 1, "Per-case call budget exhausted"),
        (2, None, 2, "Per-case call budget exhausted"),
        (4, 1, 2, "503"),
    ],
    ids=["one-call-budget", "two-call-budget", "later-transport-failure"],
)
def test_truncated_replay_preserves_limits_and_transport_failures(
    tmp_path: Path,
    endpoint: ThreadingHTTPServer,
    max_calls: int,
    fail_after: int | None,
    expected_requests: int,
    error: str,
) -> None:
    endpoint.finish_reason = "length"
    endpoint.fail_after = fail_after
    document = _manifest(endpoint)
    document["limits"]["max_calls_per_case"] = max_calls
    engine = Engine(Store(tmp_path))
    run_id = engine.start(document)["id"]
    deadline = time.monotonic() + 5
    while (status := engine.store.get(run_id)["status"]) not in TERMINAL:
        assert time.monotonic() < deadline, "run did not terminate"
        time.sleep(0.01)

    assert status == "failed"
    assert len(endpoint.seen) == expected_requests
    assert error in engine.store.results(run_id)[0]["error"]
    assert engine.store.calls(run_id)[0]["output_complete"] is False


def test_report_counts_switches_per_fixture_phase(
    tmp_path: Path, endpoint: ThreadingHTTPServer
) -> None:
    """small -> large -> small must land one switch each in tool_loop and follow_up."""
    engine = Engine(Store(tmp_path))
    run_id = engine.start(_manifest(endpoint))["id"]
    deadline = time.monotonic() + 5
    while engine.store.get(run_id)["status"] not in TERMINAL:
        assert time.monotonic() < deadline, "run did not terminate"
        time.sleep(0.01)
    target = make_report(engine.store, run_id)["summary"]["targets"][0]
    continuity = target["continuity"]
    assert continuity["session_mode"] == SESSION_AWARE
    assert continuity["model_switches"] == 2
    assert continuity["model_switches_by_phase"] == {"follow_up": 1, "tool_loop": 1}
    assert target["sr_bench_score"] is None
    assert "extension diagnostic" in target["score_scope"]
