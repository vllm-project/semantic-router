"""Check the replay against a fake Router, so no Router, model or network is needed."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load("replay")
SWITCH_AFTER_REPLIES = 2


def recording(
    run_id="run-a", model="cheap", task="Fix the build", steps=3, outcome=True
):
    rows = [
        {
            "reply": f"step {i}",
            "command": f"make step{i}\n",
            "observation": f"output {i}",
        }
        for i in range(steps - 1)
    ]
    rows.append({"reply": "done", "command": None, "observation": None})
    return replay.parse_recording(
        {
            "schema_version": replay.SCHEMA_VERSION,
            "run_id": run_id,
            "task_id": "build",
            "model": model,
            "outcome": outcome,
            "cost_usd": 0.01,
            "system": "You are a terminal agent.",
            "task": task,
            "steps": rows,
        }
    )


def serve(handler) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def fake_router(backend, choose):
    """A stand-in Router: picks a model with choose(request), then asks the backend."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(
                {"request": request, "session": self.headers.get("x-session-id")}
            )
            model = choose(request)
            code, body = backend.answer({**request, "model": model})
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-vsr-selected-model", model)
            self.send_header(
                "x-vsr-session-phase",
                "tool_loop" if request.get("tools") else "user_turn",
            )
            self.send_header("x-vsr-replay-id", f"replay-{len(seen)}")
            self.send_header("x-vsr-cost", "0.5")
            self.end_headers()
            self.wfile.write(data)

    server, url = serve(Handler)
    return server, url, seen


@pytest.fixture
def backend():
    b = replay.RecordedBackend(0)
    yield b
    b.close()


def test_parse_recording_rejects_runs_the_replay_cannot_send():
    good = dict(recording())
    for broken, message in [
        ({**good, "schema_version": "v0"}, "schema_version"),
        ({**good, "steps": []}, "steps"),
        ({**good, "steps": [{"command": "ls"}]}, "reply"),
        ({**good, "outcome": "yes"}, "outcome"),
    ]:
        with pytest.raises(replay.RecordingError, match=message):
            replay.parse_recording(broken)


def test_request_for_a_step_carries_every_earlier_reply_in_both_wire_forms():
    run = recording()
    text = replay.request_body(run, 2, "text", "vllm-sr/auto")
    assert [m["role"] for m in text["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "```bash\nmake step1\n```" in text["messages"][4]["content"]
    assert "tools" not in text
    tool = replay.request_body(run, 2, "tool", "vllm-sr/auto")
    assert [m["role"] for m in tool["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert (
        tool["messages"][4]["tool_calls"][0]["id"]
        == tool["messages"][5]["tool_call_id"]
        == "call_1"
    )
    assert tool["messages"][5]["content"] == "output 1"
    assert tool["tools"] == [replay.BASH_TOOL]
    assert [
        m["role"] for m in replay.request_body(run, 0, "tool", "m")["messages"]
    ] == ["system", "user"]


def test_backend_answers_with_the_recorded_step_in_the_requested_wire_form(backend):
    run = recording()
    backend.recordings = {"*": run}
    code, body = backend.answer(replay.request_body(run, 1, "tool", "cheap"))
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert (code, json.loads(call["function"]["arguments"])) == (
        replay.HTTP_OK,
        {"command": "make step1\n"},
    )
    code, body = backend.answer(replay.request_body(run, 1, "text", "cheap"))
    assert code == replay.HTTP_OK
    assert body["choices"][0]["message"]["content"].startswith("step 1")
    past_end = backend.answer(replay.request_body(run, 3, "text", "cheap"))
    assert past_end[0] == replay.HTTP_BAD_REQUEST
    backend.recordings = {"other": run}
    no_recording = backend.answer(replay.request_body(run, 0, "text", "cheap"))
    assert no_recording[0] == replay.HTTP_BAD_REQUEST


def test_run_report_counts_the_switch_inside_a_run(backend):
    # The stand-in Router moves to strong once the request holds two replies.
    def choose(request):
        replies = sum(1 for m in request["messages"] if m["role"] == "assistant")
        return "strong" if replies >= SWITCH_AFTER_REPLIES else "cheap"

    server, url, seen = fake_router(backend, choose)
    try:
        rows = replay.replay_runs([recording(steps=4)], "text", url, backend)
    finally:
        server.shutdown()
    assert {s["session"] for s in seen} == {"run-a"}
    row = rows[0]
    assert {k: row[k] for k in ("errors", "steps", "first_model", "router_cost")} == {
        "errors": 0,
        "steps": 4,
        "first_model": "cheap",
        "router_cost": 2.0,
    }
    assert row["selected_models"] == {"cheap": 2, "strong": 2}
    assert row["switch_steps"] == [2] and row["first_replay_id"] == "replay-1"
    summary = replay.summarize(rows)
    assert summary["runs_with_a_switch"] == 1 and summary["phases"] == {"user_turn": 4}


def test_learn_follows_the_chosen_recording_and_posts_its_outcome(backend):
    posted = []

    class Api(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posted.append(
                {
                    "path": self.path,
                    "key": self.headers.get("Idempotency-Key"),
                    "body": body,
                }
            )
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    api, api_url = serve(Api)
    router, router_url, seen = fake_router(backend, lambda request: "strong")
    tasks = replay.learn_tasks(
        [
            recording("a", "cheap", steps=2),
            recording("b", "strong", steps=3, outcome=False),
        ],
        ["cheap", "strong"],
    )
    try:
        poster = replay.OutcomePoster(api_url, "test")
        rows = replay.learn(tasks, "tool", router_url, backend, poster)
    finally:
        router.shutdown()
        api.shutdown()
    assert (len(seen), rows[0]["served_model"], rows[0]["steps"]) == (3, "strong", 3)
    assert posted == [
        {
            "path": "/api/v1/observability/outcomes",
            "key": "test:build",
            "body": {
                "replay_id": "replay-1",
                "target": "model",
                "target_ref": "strong",
                "verdict": "underpowered",
                "reason": "recorded_run_failed",
                "metadata": {"run_id": "b", "task_id": "build"},
            },
        }
    ]
    assert replay.learn_summary(rows)["served"] == {"strong": 1}


def test_tbench_rows_mark_strings_the_scrape_did_not_resolve():
    events = [
        {"src": "system", "msg": "Reply with JSON.", "tools": "None", "obs": "None"},
        {"src": "user", "msg": "$12", "tools": "None", "obs": "None"},
        {
            "src": "agent",
            "msg": "Look around.",
            "tools": "[{'fn': 'bash_command', 'cmd': 'ls\\n'}]",
            "obs": "$7",
        },
        {"src": "agent", "msg": "Finished.", "tools": "None", "obs": "None"},
    ]
    row = {
        "task_name": "build-thing",
        "model": "deepseek-v3.2@deepseek",
        "reward": 1,
        "cost_cents": 12.0,
        "trial_name": "build-thing__x",
        "trial_id": "id",
        "steps": json.dumps(events),
    }
    run = replay.tbench_recording(row)
    assert (run["model"], run["outcome"], run["cost_usd"]) == (
        "deepseek-v3.2",
        True,
        0.12,
    )
    assert run["task"] == f"{replay.NOT_RECORDED} Task: build-thing"
    assert run["steps"] == [
        {
            "reply": "Look around.",
            "command": "ls\n",
            "observation": replay.NOT_RECORDED,
        },
        {"reply": "Finished.", "command": None, "observation": None},
    ]
    assert replay.tbench_recording({**row, "steps": "null"}) is None
