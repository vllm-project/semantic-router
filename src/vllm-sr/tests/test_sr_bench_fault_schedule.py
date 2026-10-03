"""Integration and unit tests for request-scoped fault schedules in sr-bench."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.report import fault_summary, make_report
from cli.sr_bench.store import TERMINAL, Store


def wait_run(store, run_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not terminate within {timeout}s")


PRICES = {
    "model": {
        "input": 1,
        "cached_input": 0.1,
        "cache_write": 2,
        "output": 3,
    },
    "gpt-4o": {
        "input": 1,
        "cached_input": 0.1,
        "cache_write": 2,
        "output": 3,
    },
}


class FaultInjectingTarget(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.server.received_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )

        fault_key = self.headers.get("x-vsr-fault-key") or self.headers.get(
            "x-vsr-test-session-id"
        )
        call_idx = self.server.call_counts.get(fault_key, 0)
        self.server.call_counts[fault_key] = call_idx + 1

        schedule = self.server.fault_schedules.get(fault_key, [])
        active_fault = None
        for item in schedule:
            if item.get("call_index") == call_idx:
                active_fault = item
                break

        selected_model = self.server.selected_model or "backend-model-primary"
        decision = self.server.decision or "route-normal"

        if active_fault:
            if active_fault.get("status"):
                status_code = active_fault["status"]
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-vsr-fault-injected", "true")
                self.send_header("x-vsr-selected-model", selected_model)
                self.send_header("x-vsr-selected-decision", decision)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": {
                                "message": f"Injected fault: HTTP {status_code}",
                                "type": "fault_injection",
                                "code": status_code,
                            }
                        }
                    ).encode()
                )
                return

            if active_fault.get("delay"):
                time.sleep(active_fault["delay"])

            if active_fault.get("stream_cut_short"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("x-vsr-fault-injected", "true")
                self.send_header("x-vsr-selected-model", selected_model)
                self.send_header("x-vsr-selected-decision", decision)
                self.end_headers()
                event = {
                    "model": "model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Initial chunk before premature cut"},
                        }
                    ],
                }
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
                # Terminate abruptly without terminal finish_reason or [DONE]
                return

        # Normal successful streaming response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-vsr-selected-model", selected_model)
        self.send_header("x-vsr-selected-decision", decision)
        self.end_headers()

        events = [
            {
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "A"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "model": "model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def fault_target():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FaultInjectingTarget)
    server.received_requests = []
    server.call_counts = {}
    server.fault_schedules = {}
    server.selected_model = "backend-primary-a"
    server.decision = "decision-rule-1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_fault_schedule_contract_validations():
    """Verify that contracts.py strictly validates fault schedules."""
    base_manifest = {
        "version": "sr-bench-1.0",
        "name": "validation-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:8000/v1",
                "prices": PRICES,
            }
        ],
        "cases": [
            {
                "id": "task-1",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question 1"}],
                "answer": "A",
            }
        ],
        "limits": {"total_timeout_s": 2, "idle_timeout_s": 1, "max_run_seconds": 10},
    }

    # Valid manifest-level fault_schedules
    valid_manifest = {
        **base_manifest,
        "fault_schedules": {
            "task-1": [{"call_index": 0, "status": 503}],
        },
    }
    planned = plan(valid_manifest)
    assert planned["fault_schedules"] == {"task-1": [{"call_index": 0, "status": 503}]}

    # Valid case-level fault_schedule
    valid_case = {
        **base_manifest,
        "cases": [
            {
                "id": "task-1",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question 1"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 1, "delay": 0.5}],
            }
        ],
    }
    planned_case = plan(valid_case)
    assert planned_case["cases"][0]["fault_schedule"] == [
        {"call_index": 1, "delay": 0.5}
    ]

    # Invalid status code
    with pytest.raises(ValueError, match="valid HTTP status code"):
        plan(
            {
                **base_manifest,
                "fault_schedules": {
                    "task-1": [{"call_index": 0, "status": 999}],
                },
            }
        )

    # Invalid negative call_index
    with pytest.raises(ValueError, match="call_index must be a non-negative integer"):
        plan(
            {
                **base_manifest,
                "fault_schedules": {
                    "task-1": [{"call_index": -1, "status": 500}],
                },
            }
        )

    # Missing fault specification
    with pytest.raises(ValueError, match="must specify at least one fault"):
        plan(
            {
                **base_manifest,
                "fault_schedules": {
                    "task-1": [{"call_index": 0}],
                },
            }
        )

    # Unknown task ID in fault_schedules
    with pytest.raises(ValueError, match="not found in manifest cases"):
        plan(
            {
                **base_manifest,
                "fault_schedules": {
                    "non-existent-task": [{"call_index": 0, "status": 503}],
                },
            }
        )


def test_fault_summary_unit():
    """Unit test for report.fault_summary partitioning and reporting."""
    results = [
        {
            "case_id": "c1",
            "status": "failed",
            "correct": None,
            "error": "Target HTTP 503",
        },
        {"case_id": "c2", "status": "completed", "correct": True},
        {"case_id": "c3", "status": "completed", "correct": False},
    ]
    calls = [
        {
            "id": "call-1",
            "case_id": "c1",
            "role": "subject",
            "status": "failed",
            "call_index": 0,
            "injected_fault": {"call_index": 0, "status": 503},
            "selected_model": "router-picked-model-a",
            "decision": "fallback-decision",
            "response_status": 503,
            "error": "Target HTTP 503",
        },
        {
            "id": "call-2",
            "case_id": "c2",
            "role": "subject",
            "status": "completed",
            "selected_model": "model",
            "decision": "rule-normal",
            "response_status": 200,
        },
        {
            "id": "call-3",
            "case_id": "c3",
            "role": "subject",
            "status": "completed",
            "selected_model": "model",
            "decision": "rule-normal",
            "response_status": 200,
        },
    ]

    summary = fault_summary(results, calls)
    assert summary["faulted_tasks"]["total"] == 1
    assert summary["faulted_tasks"]["failed"] == 1
    assert summary["faulted_tasks"]["completed"] == 0
    assert summary["faulted_tasks"]["accuracy"] == 0.0

    faulted_task = summary["faulted_tasks"]["tasks"][0]
    assert faulted_task["case_id"] == "c1"
    assert faulted_task["outcome"] == "failed"
    assert len(faulted_task["faulted_calls"]) == 1

    f_call = faulted_task["faulted_calls"][0]
    assert f_call["call_index"] == 0
    assert f_call["injected_fault"] == {"call_index": 0, "status": 503}
    assert f_call["selected_model"] == "router-picked-model-a"
    assert f_call["router_response"]["status"] == 503
    assert f_call["router_response"]["decision"] == "fallback-decision"
    assert f_call["router_response"]["error"] == "Target HTTP 503"

    assert summary["unfaulted_tasks"]["total"] == 2
    assert summary["unfaulted_tasks"]["completed"] == 2
    assert summary["unfaulted_tasks"]["correct"] == 1
    assert summary["unfaulted_tasks"]["accuracy"] == 0.5
    assert set(summary["unfaulted_tasks"]["case_ids"]) == {"c2", "c3"}


def test_sr_bench_run_with_status_fault_on_one_task(tmp_path, fault_target):
    """Run an sr-bench evaluation where one task has a 503 schedule and others are clean."""
    store = Store(tmp_path)
    engine = Engine(store)

    fault_target.fault_schedules = {
        "task-faulted": [{"call_index": 0, "status": 503}],
    }

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "fault-status-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            }
        ],
        "cases": [
            {
                "id": "task-faulted",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question with fault"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "status": 503}],
            },
            {
                "id": "task-clean-1",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Clean Question 1"}],
                "answer": "A",
            },
            {
                "id": "task-clean-2",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Clean Question 2"}],
                "answer": "A",
            },
        ],
        "limits": {"total_timeout_s": 5, "idle_timeout_s": 2, "max_run_seconds": 15},
    }

    run = engine.start(manifest_doc, request_key="status-fault-run")
    wait_run(store, run["id"])

    # Verify header transmission
    reqs_by_fault_key = {
        r["headers"].get("x-vsr-fault-key"): r
        for r in fault_target.received_requests
        if r["headers"].get("x-vsr-fault-key")
    }
    assert "task-faulted" in reqs_by_fault_key
    assert "task-clean-1" in reqs_by_fault_key
    assert "task-clean-2" in reqs_by_fault_key

    # Check also x-vsr-test-session-id
    reqs_by_session_id = {
        r["headers"].get("x-vsr-test-session-id"): r
        for r in fault_target.received_requests
        if r["headers"].get("x-vsr-test-session-id")
    }
    assert "task-faulted" in reqs_by_session_id

    report = make_report(store, run["id"])
    assert "fault_summary" in report
    f_summary = report["fault_summary"]

    # Faulted task assertion
    faulted_tasks = f_summary["faulted_tasks"]
    assert faulted_tasks["total"] == 1
    assert faulted_tasks["failed"] == 1
    assert faulted_tasks["completed"] == 0

    faulted_detail = faulted_tasks["tasks"][0]
    assert faulted_detail["case_id"] == "task-faulted"
    assert faulted_detail["outcome"] == "failed"
    assert len(faulted_detail["faulted_calls"]) == 1

    faulted_call = faulted_detail["faulted_calls"][0]
    assert faulted_call["call_index"] == 0
    assert faulted_call["injected_fault"] == {"call_index": 0, "status": 503}
    assert faulted_call["selected_model"] == "backend-primary-a"
    assert faulted_call["router_response"]["status"] == 503
    assert faulted_call["router_response"]["decision"] == "decision-rule-1"

    # Unfaulted tasks assertion
    unfaulted = f_summary["unfaulted_tasks"]
    assert unfaulted["total"] == 2
    assert unfaulted["completed"] == 2
    assert unfaulted["correct"] == 2
    assert unfaulted["accuracy"] == 1.0
    assert set(unfaulted["case_ids"]) == {"task-clean-1", "task-clean-2"}


def test_sr_bench_run_with_delay_fault(tmp_path, fault_target):
    """Run sr-bench where one task has a delay fault that still completes."""
    store = Store(tmp_path)
    engine = Engine(store)

    fault_target.fault_schedules = {
        "task-delayed": [{"call_index": 0, "delay": 0.05}],
    }

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "fault-delay-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            }
        ],
        "cases": [
            {
                "id": "task-delayed",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question with delay"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "delay": 0.05}],
            },
            {
                "id": "task-normal",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Clean Question"}],
                "answer": "A",
            },
        ],
        "limits": {"total_timeout_s": 5, "idle_timeout_s": 2, "max_run_seconds": 15},
    }

    run = engine.start(manifest_doc, request_key="delay-fault-run")
    wait_run(store, run["id"])

    report = make_report(store, run["id"])
    f_summary = report["fault_summary"]

    assert f_summary["faulted_tasks"]["total"] == 1
    assert f_summary["faulted_tasks"]["completed"] == 1
    assert f_summary["faulted_tasks"]["correct"] == 1
    task = f_summary["faulted_tasks"]["tasks"][0]
    assert task["case_id"] == "task-delayed"
    assert task["outcome"] == "completed"
    assert task["correct"] is True
    assert task["faulted_calls"][0]["injected_fault"]["delay"] == 0.05
    assert task["faulted_calls"][0]["selected_model"] == "backend-primary-a"

    assert f_summary["unfaulted_tasks"]["total"] == 1
    assert f_summary["unfaulted_tasks"]["completed"] == 1
    assert f_summary["unfaulted_tasks"]["case_ids"] == ["task-normal"]


def test_sr_bench_run_with_stream_cut_short_fault(tmp_path, fault_target):
    """Run sr-bench where one task has a stream cut short fault."""
    store = Store(tmp_path)
    engine = Engine(store)

    fault_target.fault_schedules = {
        "task-cut-short": [{"call_index": 0, "stream_cut_short": True}],
    }

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "fault-stream-cut-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            }
        ],
        "cases": [
            {
                "id": "task-cut-short",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question cut short"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "stream_cut_short": True}],
            },
            {
                "id": "task-unaffected",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Unaffected question"}],
                "answer": "A",
            },
        ],
        "limits": {"total_timeout_s": 5, "idle_timeout_s": 2, "max_run_seconds": 15},
    }

    run = engine.start(manifest_doc, request_key="stream-cut-run")
    wait_run(store, run["id"])

    report = make_report(store, run["id"])
    f_summary = report["fault_summary"]

    assert f_summary["faulted_tasks"]["total"] == 1
    assert f_summary["faulted_tasks"]["failed"] == 1
    task = f_summary["faulted_tasks"]["tasks"][0]
    assert task["case_id"] == "task-cut-short"
    assert task["outcome"] == "failed"
    assert task["faulted_calls"][0]["injected_fault"]["stream_cut_short"] is True
    assert task["faulted_calls"][0]["selected_model"] == "backend-primary-a"

    assert f_summary["unfaulted_tasks"]["total"] == 1
    assert f_summary["unfaulted_tasks"]["completed"] == 1
    assert f_summary["unfaulted_tasks"]["case_ids"] == ["task-unaffected"]
