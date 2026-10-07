from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
from cli.sr_bench import adapters
from cli.sr_bench.contracts import SESSION_AWARE, STATELESS, plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.report import make_report
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import TERMINAL, Store
from cli.sr_bench.transport import SESSION_ID_HEADER


class _Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.seen.append((body["model"], self.headers.get(SESSION_ID_HEADER)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in (
            {
                "model": body["model"],
                "choices": [
                    {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"}
                ],
            },
            {
                "model": body["model"],
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def agent_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    def execute(case: dict[str, Any], context: Any) -> dict[str, Any]:
        context.call(case["messages"])
        context.call(case["messages"], role="judge", target="judge")
        context.call(case["messages"], role="simulator", target="simulator")
        context.call(case["messages"])
        return {"answer": "A", "correct": True, "score": 1}

    adapters.list_adapters()
    monkeypatch.setattr(adapters, "_adapters", dict(adapters._adapters))
    adapters.register_adapter(
        adapters.BenchmarkAdapter(
            id="agent-turns",
            title="Agent turns",
            kind="agent",
            source_url="https://example.test/agent-turns",
            version="agent-turns-v1",
            execute=execute,
        )
    )


def _manifest(
    endpoint: ThreadingHTTPServer, *subjects: dict[str, Any]
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    return {
        "version": "sr-bench-1.0",
        "cost_policy": "capability_only",
        "targets": [
            {"kind": "single", "model": "subject", "base_url": url, **subject}
            for subject in subjects
        ],
        "auxiliary_targets": {
            role: {"id": role, "kind": "single", "model": role, "base_url": url}
            for role in ("judge", "simulator")
        },
        "cases": [
            {
                "id": case_id,
                "benchmark": "agent-turns",
                "messages": [{"role": "user", "content": "Return A"}],
            }
            for case_id in ("task-1", "task-2")
        ],
        "limits": {"total_timeout_s": 2, "idle_timeout_s": 1, "max_run_seconds": 10},
    }


def _completed_run(engine: Engine, document: dict[str, Any]) -> str:
    run_id = engine.start(document)["id"]
    deadline = time.monotonic() + 5
    while (status := engine.store.get(run_id)["status"]) not in TERMINAL:
        assert time.monotonic() < deadline, "run did not terminate"
        time.sleep(0.01)
    assert status == "completed"
    return run_id


def test_subject_calls_share_one_identity_per_task_and_run(
    tmp_path: Path, endpoint: ThreadingHTTPServer
) -> None:
    engine = Engine(Store(tmp_path))
    document = _manifest(endpoint, {"id": "mom", "session_mode": SESSION_AWARE})
    identities_by_run = []
    for _ in range(2):
        sent_before = len(endpoint.seen)
        run_id = _completed_run(engine, document)
        calls = engine.store.calls(run_id, summary=True)
        sent = endpoint.seen[sent_before:]
        assert [model for model, _ in sent] == [
            "subject",
            "judge",
            "simulator",
            "subject",
        ] * 2
        assert [header for _, header in sent] == [
            call.get("session_id") for call in calls
        ]
        assert all(
            call.get("session_id") is None
            for call in calls
            if call["role"] != "subject"
        )
        by_task: dict[str, set[str]] = {}
        for call in calls:
            if call["role"] == "subject":
                by_task.setdefault(call["case_id"], set()).add(call["session_id"])
        assert all(len(identities) == 1 for identities in by_task.values())
        identities = {identity for ids in by_task.values() for identity in ids}
        assert len(identities) == len(by_task) == 2
        identities_by_run.append(identities)
    assert identities_by_run[0].isdisjoint(identities_by_run[1])


def test_report_labels_each_target_continuity_by_session_mode(
    tmp_path: Path, endpoint: ThreadingHTTPServer
) -> None:
    engine = Engine(Store(tmp_path))
    run_id = _completed_run(
        engine,
        _manifest(
            endpoint, {"id": "plain"}, {"id": "aware", "session_mode": SESSION_AWARE}
        ),
    )
    calls = engine.store.calls(run_id, summary=True)
    assert not any(
        "session_id" in call for call in calls if call["target_id"] == "plain"
    )
    report = make_report(engine.store, run_id)
    expected = {"plain": STATELESS, "aware": SESSION_AWARE}
    for item in report["summary"]["targets"] + report["benchmarks"]:
        target_id = item.get("target_id", item["id"])
        assert item["continuity"]["session_mode"] == expected[target_id]
        assert item["continuity"]["multi_request_tasks"] == 2


def test_preview_report_stays_stateless_without_live_subject_calls(
    tmp_path: Path, endpoint: ThreadingHTTPServer
) -> None:
    document = _manifest(
        endpoint,
        {
            "id": "mom",
            "kind": "mom",
            "config_hash": "a" * 64,
            "preview_url": "http://127.0.0.1:1/api/v1/routing/preview",
            "session_mode": SESSION_AWARE,
        },
    )
    store = Store(tmp_path)
    run, _ = store.create(plan({**document, "mode": "preview"}))
    report = make_report(store, run["id"])
    assert report["summary"]["targets"][0]["continuity"]["session_mode"] == STATELESS


@pytest.mark.parametrize("session_mode", ["per_request", None, 7, [], {}])
def test_plan_rejects_unknown_session_mode(
    endpoint: ThreadingHTTPServer, session_mode: object
) -> None:
    with pytest.raises(ValueError, match="session_mode must be"):
        plan(_manifest(endpoint, {"id": "mom", "session_mode": session_mode}))


def test_registered_session_aware_target_freezes_through_service(
    tmp_path: Path, endpoint: ThreadingHTTPServer
) -> None:
    document = _manifest(endpoint, {"id": "mom", "session_mode": SESSION_AWARE})
    (tmp_path / "targets.json").write_text(
        json.dumps(document["targets"] + list(document["auxiliary_targets"].values()))
    )
    service = Server(("127.0.0.1", 0), Store(tmp_path), "fixture-token")
    threading.Thread(target=service.serve_forever, daemon=True).start()
    try:
        response = requests.post(
            f"http://127.0.0.1:{service.server_port}{PREFIX}/plans",
            headers={
                "Authorization": "Bearer fixture-token",
                "X-SR-Bench-Actor-ID": "alice",
                "X-SR-Bench-Actor-Role": "write",
            },
            json={"manifest": {**document, "targets": [{"id": "mom"}]}},
            timeout=2,
        )
        assert response.status_code == 200, response.text
        assert (
            response.json()["manifest"]["targets"][0]["session_mode"] == SESSION_AWARE
        )
    finally:
        service.shutdown()
        service.server_close()
