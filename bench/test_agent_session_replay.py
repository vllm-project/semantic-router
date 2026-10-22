from __future__ import annotations

import copy
import itertools
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import agent_session_replay as replay
import pytest

Mutation = Callable[[dict[str, Any]], None]


def fixture_data() -> dict[str, Any]:
    return json.loads(replay.DEFAULT_FIXTURE.read_text(encoding="utf-8"))


def turn_messages(data: dict[str, Any], turn_id: str) -> list[dict[str, Any]]:
    turn = next(turn for turn in data["turns"] if turn["id"] == turn_id)
    return turn["append_messages"]


def tool_call(data: dict[str, Any]) -> dict[str, Any]:
    return turn_messages(data, "tool-result")[0]["tool_calls"][0]


def test_requests_share_a_stable_prefix_and_append_history() -> None:
    session = replay.load_session()
    requests = [
        replay.build_request(
            session=session, turn=turn, model="auto", max_tokens=8, temperature=0.0
        )
        for turn in session.turns
    ]

    for earlier, later in itertools.pairwise(requests):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
        assert later["tools"] == earlier["tools"]
    tool_request = requests[1]["messages"]
    assert tool_request[-1]["role"] == "tool"
    assert tool_request[-1]["tool_call_id"] == tool_request[-2]["tool_calls"][0]["id"]
    follow_up = requests[-1]["messages"]
    assert follow_up[-2]["role"] == "assistant"
    assert "tool_calls" not in follow_up[-2]
    assert follow_up[-1]["role"] == "user"


def _set(path: tuple[Any, ...], value: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _rename_tool_call(data: dict[str, Any]) -> None:
    tool_call(data)["function"]["name"] = "delete_repository"


def _break_arguments(data: dict[str, Any]) -> None:
    tool_call(data)["function"]["arguments"] = "tests/test_slug.py"


def _orphan_result(data: dict[str, Any]) -> None:
    turn_messages(data, "tool-result")[1]["tool_call_id"] = "call_unknown"


def _user_before_result(data: dict[str, Any]) -> None:
    turn_messages(data, "tool-result").insert(
        1, {"role": "user", "content": "still there?"}
    )


def _end_on_assistant(data: dict[str, Any]) -> None:
    turn_messages(data, "follow-up").pop()


def _duplicate_tool(data: dict[str, Any]) -> None:
    data["tools"].append(copy.deepcopy(data["tools"][0]))


def _second_system(data: dict[str, Any]) -> None:
    turn_messages(data, "follow-up").insert(
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
    data = fixture_data()
    mutate(data)

    with pytest.raises(replay.FixtureError, match=error):
        replay.parse_session(data)


def test_load_session_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text('{"id": "a", "id": "b"}', encoding="utf-8")

    with pytest.raises(replay.FixtureError, match="duplicate key 'id'"):
        replay.load_session(path)


class RouterStub(BaseHTTPRequestHandler):
    selected_models: ClassVar[list[str]] = ["small", "large", "large"]
    requests: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        RouterStub.requests.append((self.headers.get("x-session-id", ""), body))
        model = RouterStub.selected_models[len(RouterStub.requests) - 1]
        payload = {
            "model": model,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-vsr-selected-model", model)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *_args: Any) -> None:
        return


def test_replay_sends_each_turn_and_flags_a_tool_loop_switch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    RouterStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), RouterStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exit_code = replay.main(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--session-id",
                "fixture-test",
                "--timeout",
                "5",
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [session for session, _ in RouterStub.requests] == ["fixture-test"] * 3
    assert [len(body["messages"]) for _, body in RouterStub.requests] == [2, 4, 6]
    assert all(body["tools"] for _, body in RouterStub.requests)
    assert report["summary"]["model_switches"] == 1
    assert report["summary"]["tool_loop_switch_violations"] == 1
    assert report["turns"][1]["cached_tokens"] == 64
