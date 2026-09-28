"""Transport activity is durable evidence, never provisional usage or billing."""

import json
import threading
import time
from contextlib import suppress
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from cli.sr_bench.activity import CallActivity
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store


class Clock:
    elapsed = 0.0

    def __call__(self):
        return self.elapsed


def test_checkpoints_are_throttled_and_terminal_flush_preserves_last_observation():
    clock = Clock()
    saved = []
    activity = CallActivity(saved.append, clock=clock, wall_clock=lambda: 1000)
    assert activity.snapshot()["phase"] == "preparing"
    assert activity.snapshot()["last_activity_at"] is None
    clock.elapsed = 1
    activity.waiting()
    clock.elapsed = 2
    activity.received(1, clock())
    for count in range(2, 100):
        clock.elapsed += 0.001
        activity.received(count, clock())
    assert [row["phase"] for row in saved] == ["waiting", "streaming"]
    observed = activity.snapshot()["last_activity_at"]
    clock.elapsed = 3
    activity.received(99, 2.098, force=True)
    assert saved[-1]["received_bytes"] == 99
    assert saved[-1]["last_activity_at"] == observed
    assert saved[-1]["updated_at"] != observed
    activity.received(98, 2, force=True)
    activity.received(99, 3, force=True)
    assert len(saved) == 3
    clock.elapsed = 4
    activity.received(100, clock())
    assert len(saved) == 4


def _document(url="http://127.0.0.1:1/v1"):
    return {
        "version": "sr-bench-1.0",
        "output_policy": "native",
        "targets": [
            {
                "id": "subject",
                "kind": "single",
                "model": "physical",
                "base_url": url,
                "native_limits": {
                    "physical": {"context_window": 32, "max_output_tokens": 24}
                },
                "request_params": {"chat_template_kwargs": {"reasoning_effort": "max"}},
                "prices": {
                    "physical": {
                        "input": 1,
                        "cached_input": 0.1,
                        "cache_write": 1.25,
                        "output": 3,
                    }
                },
            }
        ],
        "cases": [
            {
                "id": "one",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Choose A."}],
                "answer": "A",
            }
        ],
        "limits": {
            "total_timeout_s": 10,
            "idle_timeout_s": 5,
            "case_timeout_s": 15,
            "max_run_seconds": 20,
        },
    }


@pytest.fixture
def api(tmp_path):
    store = Store(tmp_path)
    server = Server(("127.0.0.1", 0), store, "test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    store.db.close()


def _get(api, run_id, query="active=true", *, actor="alice"):
    return requests.get(
        f"http://127.0.0.1:{api.server_port}{PREFIX}/runs/{run_id}/calls?{query}",
        headers={
            "Authorization": "Bearer test-token",
            "X-SR-Bench-Actor-ID": actor,
            "X-SR-Bench-Actor-Role": "read",
        },
        timeout=2,
    )


def test_active_page_is_read_only_owned_and_preserves_historical_unknowns(api):
    store = api.store
    run, _ = store.create(plan(_document()), owner="alice")
    identity = run["id"]
    historical = store.start_call(identity, "old", "subject", "subject", {})
    store.finish_call(historical, "completed", {"cost_usd": 0.1})
    active = [
        store.start_call(identity, f"q{i}", "subject", "subject", {}) for i in range(3)
    ]
    unknown = store.start_call(identity, "uncertain", "subject", "subject", {})
    store.finish_call(unknown, "sent_unknown", {})
    before = store.db.total_changes
    first = _get(api, identity, "active=true&limit=2").json()
    assert first["total"] == 3
    assert [call["id"] for call in first["calls"]] == active[:2]
    second = _get(
        api, identity, f"active=true&limit=2&after={first['next_cursor']}"
    ).json()
    assert [call["id"] for call in second["calls"]] == active[2:]
    assert second["next_cursor"] is None
    all_calls = _get(api, identity, "active=false").json()["calls"]
    assert len(all_calls) == 5
    assert all("activity" not in call for call in all_calls)
    assert _get(api, identity, actor="bob").status_code == HTTPStatus.NOT_FOUND
    assert store.db.total_changes == before


@pytest.mark.parametrize(
    "query", ["active=1", "active=", "active=true&active=false", "extra=true"]
)
def test_invalid_active_filter_is_rejected(api, query):
    run, _ = api.store.create(plan(_document()), owner="alice")
    assert _get(api, run["id"], query).status_code == HTTPStatus.BAD_REQUEST


def test_checkpoint_does_not_refresh_run_events_or_mutate_terminal_evidence(tmp_path):
    store = Store(tmp_path)
    run, _ = store.create(plan(_document()), owner="alice")
    request = {"effective_body": {"model": "physical", "messages": []}}
    call = store.start_call(
        run["id"], "one", "subject", "subject", {"request": request}
    )
    before_run = store.get(run["id"])
    before_events = store.events(run["id"])
    recorder = CallActivity(lambda value: store.update_call_activity(call, value))
    recorder.waiting()
    recorder.received(10, time.monotonic())
    persisted = store.call(run["id"], call)
    assert persisted["request"] == request
    assert "cost_usd" not in persisted and "usage" not in persisted
    assert store.get(run["id"]) == before_run
    assert store.events(run["id"]) == before_events
    # A reopened journal observes the durable checkpoint without a live recorder.
    reopened = Store(tmp_path)
    assert reopened.call(run["id"], call)["activity"] == persisted["activity"]
    reopened.db.close()
    store.finish_call(call, "cancelled", {"cost_usd": None, "usage": None})
    terminal = store.call(run["id"], call)
    barrier = threading.Barrier(5)

    def late_callback(number):
        barrier.wait()
        store.update_call_activity(
            call, {**persisted["activity"], "received_bytes": number}
        )

    threads = [threading.Thread(target=late_callback, args=(n,)) for n in range(11, 15)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert store.call(run["id"], call) == terminal
    store.db.close()


def _wait(predicate):
    deadline = time.monotonic() + 5
    while not (value := predicate()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return value


class PausedStream(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.path.endswith("/render"):
            self.server.render_started.set()
            self.server.render_release.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "model": "physical",
                        "token_ids": list(range(12)),
                        "sampling_params": {"max_tokens": 20},
                    }
                ).encode()
            )
            return
        self.server.generation_started.set()
        self.server.stream_release.wait(5)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # A heartbeat is real received activity, but has no semantic/token evidence.
        heartbeat = b": heartbeat\n\n"
        self.wfile.write(heartbeat)
        self.wfile.flush()
        self.server.finish_release.wait(5)
        event = {
            "model": "physical",
            "choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "prompt_tokens_details": {
                    "cached_tokens": 3,
                    "cache_creation_tokens": 2,
                },
            },
        }
        ending = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        self.server.expected_stream = heartbeat + ending
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(ending)
            self.wfile.flush()


@pytest.fixture
def stream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), PausedStream)
    server.requests = []
    for name in (
        "render_started",
        "render_release",
        "generation_started",
        "stream_release",
        "finish_release",
    ):
        setattr(server, name, threading.Event())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.render_release.set()
    server.stream_release.set()
    server.finish_release.set()
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("cancel", [False, True])
def test_native_http_activity_phases_final_flush_and_cancellation(api, stream, cancel):
    frozen = plan(_document(f"http://127.0.0.1:{stream.server_port}/v1"))
    run = api.engine.start(frozen, owner="alice")
    identity = run["id"]
    assert stream.render_started.wait(3)
    preparing = _get(api, identity).json()["calls"][0]
    assert preparing["activity"]["phase"] == "preparing"
    assert preparing["activity"]["last_activity_at"] is None
    assert preparing["activity"]["received_bytes"] == 0
    stream.render_release.set()
    assert stream.generation_started.wait(3)
    waiting = _get(api, identity).json()["calls"][0]
    assert waiting["activity"]["phase"] == "waiting"
    assert waiting["activity"]["received_bytes"] == 0
    run_updated = api.store.get(identity)["updated_at"]
    events = api.store.events(identity)
    stream.stream_release.set()

    def streaming_call():
        call = _get(api, identity).json()["calls"][0]
        return call if call["activity"]["phase"] == "streaming" else None

    active = _wait(streaming_call)
    assert active["activity"]["received_bytes"] > 0
    assert datetime.fromisoformat(active["activity"]["last_activity_at"])
    assert "usage" not in active and "cost_usd" not in active and "ttft_s" not in active
    assert api.store.get(identity)["updated_at"] == run_updated
    assert api.store.events(identity) == events
    assert api.store.results(identity)[0]["status"] == "running"
    if cancel:
        api.engine.cancel(identity)
    else:
        stream.finish_release.set()
    api.engine.threads[identity].join(5)
    assert not api.engine.threads[identity].is_alive()
    assert _get(api, identity).json()["total"] == 0
    calls = api.store.calls(identity)
    assert len(calls) == 1
    call = calls[0]
    raw = next((api.store.root / "runs" / identity).rglob("*.sse")).read_bytes()
    assert call["activity"]["received_bytes"] == len(raw)
    assert len(raw) > active["activity"]["received_bytes"]
    assert len(stream.requests) == 2
    render_body, generation_body = [body for _, body in stream.requests]
    assert "max_tokens" not in render_body
    assert generation_body == {**render_body, "max_tokens": 20}
    assert call["request"]["effective_body"] == render_body
    assert call["native_output"]["effective_request_sha256"] == digest(generation_body)
    assert call["native_output"]["input_tokens"] == 12
    assert call["native_output"]["max_output_tokens"] == 20
    if cancel:
        assert call["status"] == "cancelled"
        assert call["usage"] is None and call["cost_usd"] is None
        assert api.store.get(identity)["status"] == "cancelled"
        assert raw == b": heartbeat\n\n"
    else:
        assert call["status"] == "completed"
        assert api.store.get(identity)["status"] == "completed"
        assert raw == stream.expected_stream
        assert call["final"] == "A"
        assert call["usage"] == {
            "input_tokens": 7,
            "cached_input_tokens": 3,
            "cache_write_tokens": 2,
            "output_tokens": 1,
        }
        assert call["cost_usd"] == pytest.approx(0.0000128)
        assert api.store.results(identity)[0]["correct"] is True
