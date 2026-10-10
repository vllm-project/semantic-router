"""Integration and unit tests for request-scoped fault schedules in sr-bench."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.sr_bench.adapters import BenchmarkAdapter
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

        schedule_key = self.headers.get("x-vsr-fault-schedule-id")
        counter_key = (
            self.headers.get("x-vsr-fault-session-id")
            or self.headers.get("x-vsr-test-session-id")
            or schedule_key
        )
        call_idx = self.server.call_counts.get(counter_key, 0)
        self.server.call_counts[counter_key] = call_idx + 1

        schedule = self.server.fault_schedules.get(schedule_key, [])
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
        if active_fault:
            self.send_header("x-vsr-fault-injected", "true")
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
            "target_id": "single",
            "case_id": "c1",
            "status": "failed",
            "correct": None,
            "error": "Target HTTP 503",
        },
        {
            "target_id": "single",
            "case_id": "c2",
            "status": "completed",
            "correct": True,
        },
        {
            "target_id": "single",
            "case_id": "c3",
            "status": "completed",
            "correct": False,
        },
        {
            "target_id": "single",
            "case_id": "c4",
            "status": "completed",
            "correct": True,
        },
    ]
    calls = [
        {
            "id": "call-1",
            "target_id": "single",
            "case_id": "c1",
            "role": "subject",
            "status": "failed",
            "call_index": 0,
            "fault_injected": True,
            "injected_fault": {"call_index": 0, "status": 503},
            "selected_model": "router-picked-model-a",
            "decision": "fallback-decision",
            "response_status": 503,
            "error": "Target HTTP 503",
        },
        {
            "id": "call-2",
            "target_id": "single",
            "case_id": "c2",
            "role": "subject",
            "status": "completed",
            "fault_injected": False,
            "selected_model": "model",
            "decision": "rule-normal",
            "response_status": 200,
        },
        {
            "id": "call-3",
            "target_id": "single",
            "case_id": "c3",
            "role": "subject",
            "status": "completed",
            "fault_injected": False,
            "selected_model": "model",
            "decision": "rule-normal",
            "response_status": 200,
        },
        {
            "id": "call-4",
            "target_id": "single",
            "case_id": "c4",
            "role": "subject",
            "status": "completed",
            "call_index": 0,
            "scheduled_fault": {"call_index": 0, "status": 503},
            "fault_injected": False,
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

    assert summary["unfaulted_tasks"]["total"] == 3
    assert summary["unfaulted_tasks"]["completed"] == 3
    assert summary["unfaulted_tasks"]["correct"] == 2
    assert summary["unfaulted_tasks"]["accuracy"] == 2 / 3
    assert set(summary["unfaulted_tasks"]["case_ids"]) == {"c2", "c3", "c4"}


def test_fault_summary_multi_target():
    """Verify multi-target fault_summary aggregates by (target_id, case_id) and maintains denominator."""
    # Two targets (t1 and t2) running the same case (c1), plus c2 on both
    results = [
        {
            "target_id": "t1",
            "case_id": "c1",
            "status": "failed",
            "correct": False,
            "latency_s": 0.5,
        },
        {
            "target_id": "t2",
            "case_id": "c1",
            "status": "completed",
            "correct": True,
            "latency_s": 0.2,
        },
        {
            "target_id": "t1",
            "case_id": "c2",
            "status": "completed",
            "correct": True,
            "latency_s": 0.1,
        },
        {
            "target_id": "t2",
            "case_id": "c2",
            "status": "completed",
            "correct": True,
            "latency_s": 0.1,
        },
    ]
    calls = [
        {
            "id": "call-t1-c1",
            "target_id": "t1",
            "case_id": "c1",
            "role": "subject",
            "status": "failed",
            "call_index": 0,
            "fault_injected": True,
            "injected_fault": {"call_index": 0, "status": 503},
            "selected_model": "model-a",
            "response_status": 503,
        },
        {
            "id": "call-t2-c1",
            "target_id": "t2",
            "case_id": "c1",
            "role": "subject",
            "status": "completed",
            "call_index": 0,
            "fault_injected": False,
            "selected_model": "model-b",
            "response_status": 200,
        },
        {
            "id": "call-t1-c2",
            "target_id": "t1",
            "case_id": "c2",
            "role": "subject",
            "status": "completed",
            "call_index": 0,
            "fault_injected": False,
            "selected_model": "model-a",
            "response_status": 200,
        },
        {
            "id": "call-t2-c2",
            "target_id": "t2",
            "case_id": "c2",
            "role": "subject",
            "status": "completed",
            "call_index": 0,
            "fault_injected": False,
            "selected_model": "model-b",
            "response_status": 200,
        },
    ]

    # Total planned execution cells = 4 (2 targets * 2 cases)
    summary = fault_summary(results, calls, total=4)

    # Exactly 1 faulted execution: (t1, c1)
    assert summary["faulted_tasks"]["total"] == 1
    assert summary["faulted_tasks"]["failed"] == 1
    assert summary["faulted_tasks"]["completed"] == 0
    f_task = summary["faulted_tasks"]["tasks"][0]
    assert f_task["case_id"] == "c1"
    assert f_task["target_id"] == "t1"
    assert f_task["outcome"] == "failed"
    assert f_task["latency_s"] == 0.5

    # Exactly 3 unfaulted executions: (t2, c1), (t1, c2), (t2, c2)
    assert summary["unfaulted_tasks"]["total"] == 3
    assert summary["unfaulted_tasks"]["completed"] == 3
    assert summary["unfaulted_tasks"]["correct"] == 3
    assert summary["unfaulted_tasks"]["accuracy"] == 1.0


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
    reqs_by_schedule_id = {
        r["headers"].get("x-vsr-fault-schedule-id"): r
        for r in fault_target.received_requests
        if r["headers"].get("x-vsr-fault-schedule-id")
    }
    assert "task-faulted" in reqs_by_schedule_id
    assert "task-clean-1" in reqs_by_schedule_id
    assert "task-clean-2" in reqs_by_schedule_id

    # Check also x-vsr-fault-session-id has per-target/task scoped session
    reqs_by_session_id = {
        r["headers"].get("x-vsr-fault-session-id"): r
        for r in fault_target.received_requests
        if r["headers"].get("x-vsr-fault-session-id")
    }
    assert any(k.endswith(":task-faulted") for k in reqs_by_session_id)
    assert any(k.endswith(":task-clean-1") for k in reqs_by_session_id)
    assert any(k.endswith(":task-clean-2") for k in reqs_by_session_id)

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


def test_sr_bench_scheduled_fault_not_injected_by_provider_reports_as_unfaulted(
    tmp_path, fault_target
):
    """Scheduled fault not injected by provider must report as unfaulted."""
    store = Store(tmp_path)
    engine = Engine(store)

    # Provider is NOT configured with a fault schedule for this task
    fault_target.fault_schedules = {}

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "unexercised-fault-test",
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
                "id": "task-with-planned-fault",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question expecting fault"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "status": 503}],
            },
        ],
        "limits": {"total_timeout_s": 5, "idle_timeout_s": 2, "max_run_seconds": 15},
    }

    run = engine.start(manifest_doc, request_key="unexercised-fault-run")
    wait_run(store, run["id"])

    calls = store.calls(run["id"])
    assert len(calls) == 1
    call = calls[0]
    # Planned intent is recorded
    assert call["scheduled_fault"] == {"call_index": 0, "status": 503}
    # But because provider did not inject it, injected_fault is None
    assert call.get("injected_fault") is None
    assert call["fault_injected"] is False

    report = make_report(store, run["id"])
    f_summary = report["fault_summary"]

    # Must NOT report as a faulted task
    assert f_summary["faulted_tasks"]["total"] == 0
    assert f_summary["faulted_tasks"]["tasks"] == []

    # Must report as an unfaulted task
    assert f_summary["unfaulted_tasks"]["total"] == 1
    assert f_summary["unfaulted_tasks"]["completed"] == 1
    assert f_summary["unfaulted_tasks"]["correct"] == 1
    assert f_summary["unfaulted_tasks"]["case_ids"] == ["task-with-planned-fault"]


def test_unexpected_transport_failure_without_provider_receipt_fails_closed(tmp_path):
    """Unexpected transport failure without provider receipt must fail closed."""
    store = Store(tmp_path)
    engine = Engine(store)

    # Use a port where no server is listening to simulate transport failure
    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "fail-closed-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:9",
                "prices": PRICES,
            }
        ],
        "cases": [
            {
                "id": "task-transport-failure",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "status": 503}],
            },
            {
                "id": "task-subsequent",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question 2"}],
                "answer": "B",
            },
        ],
        "limits": {
            "concurrency": 1,
            "total_timeout_s": 2,
            "idle_timeout_s": 1,
            "max_run_seconds": 5,
        },
    }

    run = engine.start(manifest_doc, request_key="fail-closed-run")
    wait_run(store, run["id"])

    # Fail closed prevented subsequent tasks from running
    results = store.results(run["id"])
    assert len(results) == 1
    assert results[0]["case_id"] == "task-transport-failure"
    assert results[0]["status"] == "failed"

    calls = store.calls(run["id"])
    assert len(calls) == 1
    call = calls[0]
    # Planned intent is recorded
    assert call["scheduled_fault"] == {"call_index": 0, "status": 503}
    # No provider receipt exists
    assert call["fault_injected"] is False
    assert call.get("injected_fault") is None


def test_earlier_injected_call_does_not_exempt_later_unexpected_failure(
    tmp_path, fault_target, monkeypatch
):
    """An earlier injected call with receipt must not exempt a later unexpected failure from fail-closed."""
    store = Store(tmp_path)
    engine = Engine(store)

    fault_target.fault_schedules = {
        "task-failing": [{"call_index": 0, "delay": 0.01}],
    }

    def multi_call_execute(case, context):
        # Call 0: succeeds and observes injection receipt from fault_target
        context.call(case["messages"])
        # Call 1: calls broken auxiliary target which fails with connection refused
        context.call(case["messages"], target="broken")

    custom_adapter = BenchmarkAdapter(
        id="mmlu-pro",
        title="MMLU-Pro",
        kind="mcq",
        source_url="https://example.com",
        version="sr-bench-1.0",
        execute=multi_call_execute,
    )
    monkeypatch.setattr("cli.sr_bench.engine.get_adapter", lambda _: custom_adapter)

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "multi-call-fail-closed-test",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            }
        ],
        "auxiliary_targets": {
            "broken": {
                "id": "broken",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:9",
                "prices": PRICES,
            }
        },
        "cases": [
            {
                "id": "task-failing",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question 1"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "delay": 0.01}],
            },
            {
                "id": "task-subsequent",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question 2"}],
                "answer": "B",
            },
        ],
        "limits": {
            "concurrency": 1,
            "total_timeout_s": 5,
            "idle_timeout_s": 2,
            "max_run_seconds": 10,
        },
    }

    run = engine.start(manifest_doc, request_key="multi-call-run")
    wait_run(store, run["id"])

    # Task 1 failed with unexpected failure on Call 1, so fail-closed stopped dispatching Task 2
    results = store.results(run["id"])
    assert len(results) == 1
    assert results[0]["case_id"] == "task-failing"
    assert results[0]["status"] == "failed"

    calls = store.calls(run["id"])
    assert len(calls) == 2
    # Call 0 had injection receipt
    assert calls[0]["fault_injected"] is True
    # Call 1 failed without injection receipt
    assert calls[1]["fault_injected"] is False

    # Second task was suppressed due to fail-closed
    assert not any(r["case_id"] == "task-subsequent" for r in results)


def test_sr_bench_multi_target_sharing_same_case_receives_schedule(
    tmp_path, fault_target
):
    """Multiple targets sharing the same case ID both receive their scheduled fault independently."""
    store = Store(tmp_path)
    engine = Engine(store)

    fault_target.fault_schedules = {
        "task-shared": [{"call_index": 0, "status": 503}],
    }

    manifest_doc = {
        "version": "sr-bench-1.0",
        "name": "multi-target-schedule-test",
        "targets": [
            {
                "id": "target-a",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            },
            {
                "id": "target-b",
                "kind": "single",
                "model": "model",
                "base_url": f"http://127.0.0.1:{fault_target.server_port}/v1",
                "prices": PRICES,
            },
        ],
        "cases": [
            {
                "id": "task-shared",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Question"}],
                "answer": "A",
                "fault_schedule": [{"call_index": 0, "status": 503}],
            },
        ],
        "limits": {"total_timeout_s": 5, "idle_timeout_s": 2, "max_run_seconds": 15},
    }

    run = engine.start(manifest_doc, request_key="multi-target-run")
    wait_run(store, run["id"])

    report = make_report(store, run["id"])
    f_summary = report["fault_summary"]

    # Both target executions received their intended call_index 0 fault independently
    assert f_summary["faulted_tasks"]["total"] == 2
    assert f_summary["faulted_tasks"]["failed"] == 2
    targets_reported = {t["target_id"] for t in f_summary["faulted_tasks"]["tasks"]}
    assert targets_reported == {"target-a", "target-b"}
