"""Integration assertions for durable work ownership and real HTTP transport."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine, basic_grade
from cli.sr_bench.report import compare, make_report
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store, TERMINAL
from cli.sr_bench.transport import CallFailure, chat, final_content, normalize_usage


class Target(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if self.server.ack:
            self.send_header("X-SR-Bench-Config-Hash", self.server.ack)
        self.end_headers()
        events = [
            {
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "The answer might be B."},
                    }
                ],
            },
            {
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": self.server.answer},
                        "finish_reason": "length" if self.server.truncated else "stop",
                    }
                ],
            },
            {
                "model": "model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {
                        "cached_tokens": 2,
                        "cache_creation_tokens": 1,
                    },
                },
            },
        ]
        try:
            for event in events:
                if self.server.delay:
                    time.sleep(self.server.delay)
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def target():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    server.requests = []
    server.answer = "A"
    server.truncated = False
    server.delay = 0
    server.ack = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def manifest(target, **updates):
    payload = {
        "version": "sr-bench-1.0",
        "name": "integration",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{target.server_port}/v1",
                "prices": {
                    "model": {
                        "input": 1,
                        "cached_input": 0.1,
                        "cache_write": 2,
                        "output": 3,
                    }
                },
            }
        ],
        "cases": [
            {
                "id": "q1",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Return A"}],
                "answer": "A",
            }
        ],
        "limits": {"total_timeout_s": 2, "idle_timeout_s": 1, "max_run_seconds": 10},
    }
    payload.update(updates)
    return payload


def wait_run(store, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in TERMINAL:
            return run
        time.sleep(0.01)
    raise AssertionError("run did not terminate")


def test_live_http_usage_final_channel_and_idempotency(tmp_path, target):
    store = Store(tmp_path)
    engine = Engine(store)
    run = engine.start(manifest(target), request_key="once")
    assert wait_run(store, run["id"])["status"] == "completed"
    again = engine.start(manifest(target), request_key="once")
    assert again["id"] == run["id"] and len(target.requests) == 1
    result = store.results(run["id"])[0]
    assert result["answer"] == "A" and result["correct"] is True
    report = make_report(store, run["id"])
    score = report["summary"]["targets"][0]
    assert score["tokens"] == {
        "input_tokens": 7,
        "cached_input_tokens": 2,
        "cache_write_tokens": 1,
        "output_tokens": 3,
    }
    assert score["cost_usd"] == pytest.approx(18.2 / 1_000_000)
    assert score["accuracy"] == 1 and score["total"] == 1
    assert list((tmp_path / "runs" / run["id"]).glob("*/*.sse"))
    assert store.calls(run["id"])[0]["reasoning"] == "The answer might be B."


def test_truncated_final_counts_incorrect_and_continues(tmp_path, target):
    target.truncated = True
    m = manifest(target)
    m["cases"] *= 1
    m["cases"].append({**m["cases"][0], "id": "q2"})
    store = Store(tmp_path)
    run = Engine(store).start(m)
    assert wait_run(store, run["id"])["status"] == "completed"
    assert len(target.requests) == 2
    result = store.results(run["id"])[0]
    assert result["status"] == "completed" and result["correct"] is False
    assert result["details"]["quality_failure"] == "output_limit"
    assert store.calls(run["id"])[0]["final"] == "A"
    assert make_report(store, run["id"])["summary"]["targets"][0]["accuracy"] == 0


def test_wall_deadline_stops_even_with_stream_progress(target):
    target.delay = 0.2
    m = plan(manifest(target))
    limits = {**m["limits"], "total_timeout_s": 0.25, "idle_timeout_s": 1}
    started = time.monotonic()
    with pytest.raises(CallFailure, match="deadline"):
        chat(
            m["targets"][0],
            m["cases"][0]["messages"],
            m["sampling"],
            limits,
            lambda: False,
        )
    assert time.monotonic() - started < 1


def test_mom_runtime_identity_is_mandatory(tmp_path, target):
    m = manifest(target)
    m["targets"][0].update(kind="mom", config_hash="fixed")
    m["cost_policy"] = "capability_only"
    store = Store(tmp_path)
    run = Engine(store).start(m)
    assert wait_run(store, run["id"])["status"] == "failed"
    assert "identity" in store.results(run["id"])[0]["error"]
    target.ack = "fixed"
    run = Engine(store).start(m)
    assert wait_run(store, run["id"])["status"] == "completed"
    assert make_report(store, run["id"])["summary"]["targets"][0]["cost_usd"] is None


def test_recovery_never_reissues_sent_call(tmp_path, target):
    store = Store(tmp_path)
    frozen = plan(manifest(target))
    run, _ = store.create(frozen)
    store.status(run["id"], "running")
    store.result(run["id"], "q1", "single", "running", {})
    store.start_call(run["id"], "q1", "single", "subject", {})
    Engine(store)
    assert store.get(run["id"])["status"] == "interrupted"
    assert store.calls(run["id"])[0]["status"] == "sent_unknown"
    assert not target.requests


def test_comparison_requires_identical_frozen_cases(tmp_path, target):
    store = Store(tmp_path)
    engine = Engine(store)
    base = engine.start(manifest(target))
    wait_run(store, base["id"])
    target.answer = "B"
    candidate = engine.start(manifest(target))
    wait_run(store, candidate["id"])
    difference = compare(store, base["id"], candidate["id"])["comparisons"][0]
    assert difference["quality_delta"] == -1 and difference["paired_cases"] == 1
    m = manifest(target)
    m["cases"][0]["answer"] = "B"
    other = engine.start(m)
    wait_run(store, other["id"])
    with pytest.raises(ValueError, match="case_sha256"):
        compare(store, base["id"], other["id"])


def test_shared_api_enforces_actor_scope(tmp_path, target):
    (tmp_path / "targets.json").write_text(json.dumps(manifest(target)["targets"]))
    service = Server(("127.0.0.1", 0), Store(tmp_path), "test-service-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}" + PREFIX
    headers = {
        "Authorization": "Bearer test-service-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "editor",
    }
    try:
        run = requests.post(
            url + "/runs",
            headers=headers,
            json={"manifest": manifest(target), "idempotency_key": "submit"},
            timeout=2,
        ).json()
        assert "id" in run
        wait_run(service.store, run["id"])
        bob = {**headers, "X-SR-Bench-Actor-ID": "bob"}
        assert (
            requests.get(url + "/runs/" + run["id"], headers=bob, timeout=2).status_code
            == 404
        )
        assert requests.get(url + "/runs", headers=bob, timeout=2).json() == {
            "runs": []
        }
        admin = {**bob, "X-SR-Bench-Actor-Role": "admin"}
        assert (
            len(requests.get(url + "/runs", headers=admin, timeout=2).json()["runs"])
            == 1
        )
        assert requests.get(url + "/runs", timeout=2).status_code == 403
    finally:
        service.shutdown()
        service.server_close()


def test_final_only_and_bucket_validation():
    assert final_content("<think>A") == ""
    assert final_content("<think>B</think>A") == "A"
    case = {"benchmark": "mmlu-pro", "answer": "A"}
    assert basic_grade(case, "A is probably the answer")["correct"] is False
    with pytest.raises(CallFailure):
        normalize_usage(
            {"prompt_tokens": 2, "completion_tokens": 1, "cache_read_input_tokens": 3}
        )


def test_cancellation_retains_evidence_and_does_not_dispatch_rest(tmp_path, target):
    target.delay = 0.2
    store = Store(tmp_path)
    engine = Engine(store)
    m = manifest(target)
    m["cases"].append({**m["cases"][0], "id": "q2"})
    run = engine.start(m)
    deadline = time.monotonic() + 2
    while not target.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    engine.cancel(run["id"])
    assert wait_run(store, run["id"])["status"] == "cancelled"
    assert len(target.requests) == 1
    assert len(store.calls(run["id"])) == 1


def test_cost_reservation_rejects_before_generation(tmp_path, target):
    store = Store(tmp_path)
    engine = Engine(store)
    m = manifest(target)
    m["limits"]["max_cost_usd"] = 0.000001
    run = engine.start(m)
    assert wait_run(store, run["id"])["status"] == "failed"
    assert not target.requests
    assert "reservation" in store.results(run["id"])[0]["error"]


def test_multimodel_four_bucket_cost_receipt():
    from cli.sr_bench.transport import mom_usage

    prices = {
        "a": {"input": 1, "cached_input": 0.1, "cache_write": 2, "output": 3},
        "b": {"input": 2, "cached_input": 0.2, "cache_write": 4, "output": 6},
    }
    receipt = {
        "version": 1,
        "complete": True,
        "calls": [
            {
                "model": m,
                "role": "subject",
                "stage": "selection",
                "status": "completed",
                "usage": {
                    "prompt_tokens": 10,
                    "cached_input_tokens": 2,
                    "cache_write_tokens": 1,
                    "completion_tokens": 3,
                },
            }
            for m in ("a", "b")
        ],
    }
    result = mom_usage({"x-vsr-model-usage": json.dumps(receipt)}, None, prices)
    assert result["cost_usd"] == pytest.approx(54.6 / 1_000_000)
    assert result["usage"]["input_tokens"] == 14
    assert result["inference_call_count"] == 2
    receipt["complete"] = False
    assert (
        mom_usage({"x-vsr-model-usage": json.dumps(receipt)}, None, prices)["cost_usd"]
        is None
    )


def test_arc_multiple_test_grids_are_atomic():
    case = {
        "benchmark": "arc-agi-2",
        "answer": [[[1, 2]], [[3]]],
        "metadata": {"output_format": "grids"},
    }
    assert basic_grade(case, "[[[1,2]],[[3]]]")["correct"] is True
    assert basic_grade(case, "[[1,2]]")["correct"] is False


def test_offline_replay_regrade_and_export_never_infer(tmp_path, target):
    from cli.sr_bench.offline import replay, regrade, export_training
    from cli.sr_bench.contracts import digest

    store = Store(tmp_path)
    engine = Engine(store)
    m = manifest(target)
    m["cases"][0]["metadata"] = {"split": "dev"}
    baseline = engine.start(m)
    wait_run(store, baseline["id"])
    pm = plan(manifest(target))
    pm.update(
        {
            "mode": "preview",
            "cases": m["cases"],
            "case_sha256": digest(m["cases"]),
            "targets": [
                {
                    "id": "balance",
                    "kind": "mom",
                    "model": "balance",
                    "base_url": m["targets"][0]["base_url"],
                }
            ],
        }
    )
    pm["plan_sha256"] = digest({k: v for k, v in pm.items() if k != "plan_sha256"})
    preview, _ = store.create(pm)
    store.result(
        preview["id"],
        "q1",
        "balance",
        "completed",
        {
            "benchmark": "mmlu-pro",
            "details": {
                "routing": {
                    "selection_status": "selected",
                    "selection_method": "static",
                    "selected_model": "model",
                    "decision_result": {"plugins": []},
                }
            },
        },
    )
    store.status(preview["id"], "completed")
    cached = replay(store, baseline["id"], preview["id"])
    report = make_report(store, cached["id"])
    assert report["summary"]["total_spend_usd"] == 0
    assert report["summary"]["targets"][0]["accuracy"] is None
    assert report["summary"]["targets"][0]["estimated_accuracy"] == 1
    assert report["summary"]["targets"][0]["sr_bench_score"] is None
    assert regrade(store, baseline["id"])["changed_count"] == 0
    exported = export_training(store, baseline["id"])
    assert (
        exported["split"] == "dev"
        and exported["cases"][0]["targets"][0]["final"] == "A"
    )
    assert len(target.requests) == 1
    with pytest.raises(ValueError, match="live"):
        regrade(store, cached["id"])


def test_training_export_refuses_unknown_split(tmp_path, target):
    from cli.sr_bench.offline import export_training

    store = Store(tmp_path)
    run = Engine(store).start(manifest(target))
    wait_run(store, run["id"])
    with pytest.raises(ValueError, match="holdout and unknown"):
        export_training(store, run["id"])
