from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench import agent_task, agent_task_kernel
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.canonical import canonical
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, SESSION_AWARE, plan
from cli.sr_bench.dataset_io import MAX_ROWS
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.grading import validate_result
from cli.sr_bench.offline import export_training, regrade, replay
from cli.sr_bench.report import compare
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from cli.sr_bench.transport import PROVIDER_STATE_PHASE, SESSION_ID_HEADER

from bench import agent_task_live_benchmark as native

ROOT: Path = Path(__file__).resolve().parents[3]


def coverage_result(
    content: str, terms: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
) -> dict[str, Any]:
    score, missing = native.score_answer(content=content, expected_terms=terms)
    return {
        "answer": content,
        "correct": None,
        "score": score,
        "details": {
            "token_coverage": {
                "expected_terms": list(terms),
                "matched_terms": [term for term in terms if term not in missing],
                "missing_terms": missing,
                "full_credit": score == 1.0,
            }
        },
    }


def test_agent_task_is_an_opt_in_token_coverage_diagnostic() -> None:
    adapter = get_adapter("agent-task-chat-v1")
    assert adapter.result_kind == "token-coverage-v1"
    assert adapter.id not in BENCHMARK_WEIGHTS


@pytest.mark.parametrize(
    ("content", "score", "full_credit"),
    [("", 0.0, False), ("A B C", 0.4286, False), ("a b c d e f g", 1.0, True)],
)
def test_result_contract_preserves_native_rounded_credit(
    content: str, score: float, full_credit: bool
) -> None:
    result = coverage_result(content=content)
    validate_result(kind="token-coverage-v1", result=result)
    assert result["score"] == score
    assert result["details"]["token_coverage"]["full_credit"] is full_credit
    assert result["correct"] is None


@pytest.mark.parametrize(
    "field",
    [
        "unrounded",
        "nan",
        "inf",
        "negative",
        "above_one",
        "boolean",
        "missing",
        "binary_correct",
        "empty_expected",
        "matched",
        "missing_terms",
        "full_credit",
        "non_boolean_success",
        "answer",
    ],
)
def test_result_contract_rejects_unknown_or_inconsistent_credit(field: str) -> None:
    result = coverage_result(content="a b c")
    coverage = result["details"]["token_coverage"]
    if field in {
        "unrounded",
        "nan",
        "inf",
        "negative",
        "above_one",
        "boolean",
        "missing",
    }:
        result["score"] = {
            "unrounded": 3 / 7,
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -0.1,
            "above_one": 1.1,
            "boolean": True,
            "missing": None,
        }[field]
    elif field == "binary_correct":
        result["correct"] = False
    elif field == "empty_expected":
        coverage["expected_terms"] = []
    elif field == "matched":
        coverage["matched_terms"] = ["d", "e", "f"]
    elif field == "missing_terms":
        coverage["missing_terms"].pop()
    elif field == "full_credit":
        coverage["full_credit"] = True
    elif field == "non_boolean_success":
        coverage["full_credit"] = 0
    else:
        result["answer"] = None
    with pytest.raises(ValueError):
        validate_result(kind="token-coverage-v1", result=result)


def test_token_credit_does_not_fit_the_weighted_rubric_contract() -> None:
    result = coverage_result(content="a b c")
    result["details"]["rubric"] = {
        "total": 3,
        "max_positive": 7,
        "positive_earned": 3,
        "negative_penalty": 0,
        "n_negative_triggered": 0,
        "per_section": {"terms": 3},
        "verdicts": [{"id": "term", "weight": 1, "matched": True}],
    }
    with pytest.raises(ValueError, match="signed-weight contract"):
        validate_result(kind="weighted-rubric-v1", result=result)


@pytest.mark.parametrize(
    "result", [None, {}, {"score": None}, {"correct": False, "score": 0}]
)
def test_missing_results_cannot_become_zero_credit(result: object) -> None:
    with pytest.raises(ValueError):
        validate_result(kind="token-coverage-v1", result=result)


class Provider(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict[str, Any]] = []
        self.native_turns = {
            canonical(native.build_messages(task=task, turn_index=index)): (
                task_index,
                index,
                turn,
            )
            for task_index, task in enumerate(native.task_specs("all"))
            for index, turn in enumerate(task.turns)
        }
        self.fault: str | None = None
        self.fault_on_request: int = 2
        self.omit_usage = False
        self.omit_model = False
        self.omit_phase = False
        self.hidden_answer: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        server = self.server
        assert isinstance(server, Provider)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(
            {
                "body": body,
                "session_id": self.headers.get(SESSION_ID_HEADER),
                "path": self.path,
            }
        )
        matched = server.native_turns.get(canonical(body["messages"]))
        task_index, turn_index, turn = matched or (0, 0, None)
        content = "A"
        if turn is not None:
            terms = turn.expected_terms
            chosen = (
                terms
                if task_index % 3 == 0
                else terms[: len(terms) // 2] if task_index % 3 == 1 else ()
            )
            content = "\n".join(chosen) if terms else "MODEL ANSWER NOT USED AS HISTORY"
        fault = (
            server.fault if len(server.requests) == server.fault_on_request else None
        )
        if fault == "http":
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        model = "model-a" if turn_index % 2 == 0 else "model-b"
        streamed = bool(body.get("stream")) and fault != "buffered"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "text/event-stream" if streamed else "application/json"
        )
        if not server.omit_model:
            self.send_header("x-vsr-selected-model", model)
        if not server.omit_phase:
            self.send_header("x-vsr-session-phase", PROVIDER_STATE_PHASE)
        self.send_header("x-vsr-selected-decision", "local-fixture")
        self.end_headers()
        if fault == "cancel":
            self.wfile.write(b": waiting for cancellation\n\n")
            self.wfile.flush()
            server.entered.set()
            assert server.release.wait(timeout=15)
            return
        finish = "length" if fault == "length" else "stop"
        message = {"content": content}
        if server.hidden_answer and turn is not None and turn.expected_terms:
            tokens = "\n".join(turn.expected_terms)
            message = (
                {"content": "", "reasoning_content": tokens}
                if server.hidden_answer == "reasoning"
                else {"content": f"<think>{tokens}</think>visible answer"}
            )
        event = {
            **({"model": model} if not server.omit_model else {}),
            "choices": [
                {
                    "index": 0,
                    "delta" if streamed else "message": message,
                    "finish_reason": finish,
                }
            ],
        }
        if not server.omit_usage:
            event["usage"] = {"prompt_tokens": 10, "completion_tokens": 3}
        with suppress(BrokenPipeError, ConnectionResetError):
            if streamed:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                if fault != "incomplete_sse":
                    self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps(event).encode())


@pytest.fixture
def provider() -> Iterator[Provider]:
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


def document(
    root: Path, provider: Provider, suite: str = "smoke", repetitions: int = 1
) -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "profile": "quick",
        "seed": 0,
        "cost_policy": "capability_only",
        "dataset": agent_task.prepare(
            store=root, suite=suite, task_repetitions=repetitions
        ),
        "targets": [
            {
                "id": "subject",
                "kind": "single",
                "model": "requested-entrypoint",
                "base_url": provider.base_url,
                "session_mode": SESSION_AWARE,
            }
        ],
        "benchmark_options": {
            agent_task.BENCHMARK_ID: {"grader_version": agent_task.GRADER_VERSION}
        },
        "sampling": {"temperature": 0, "max_tokens": 96},
        "limits": {
            "max_calls_per_case": 6,
            "total_timeout_s": 5,
            "idle_timeout_s": 2,
            "case_timeout_s": 15,
            "max_run_seconds": 60,
        },
    }


def benchmark_cli(service: Server, *args: str) -> subprocess.CompletedProcess[str]:
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
        cwd=service.store.root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def run_cli(
    service: Server, value: dict[str, Any], *, detach: bool = False
) -> subprocess.CompletedProcess[str]:
    path = service.store.root / "run.json"
    path.write_text(json.dumps(value))
    return benchmark_cli(
        service,
        "run",
        "--manifest",
        str(path),
        "--idempotency-key",
        "agent-task-once",
        *(["--detach"] if detach else []),
    )


@pytest.mark.parametrize(
    ("suite", "repetitions"), [("smoke", 2), ("long-horizon", 1), ("all", 1)]
)
def test_native_and_cli_requests_grades_counts_and_phases_match(
    tmp_path: Path, provider: Provider, service: Server, suite: str, repetitions: int
) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/agent_task_live_benchmark.py"),
            "--base-url",
            provider.base_url,
            "--model",
            "requested-entrypoint",
            "--suite",
            suite,
            "--task-repetitions",
            str(repetitions),
            "--output-dir",
            str(legacy_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert legacy.returncode == 0
    legacy_summary = json.loads((legacy_dir / "summary.json").read_text())
    legacy_rows = [
        json.loads(line)
        for line in (legacy_dir / "turns.jsonl").read_text().splitlines()
    ]
    legacy_requests = list(provider.requests)
    provider.requests.clear()
    value = document(
        root=service.store.root, provider=provider, suite=suite, repetitions=repetitions
    )
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    target = report["summary"]["targets"][0]
    quality = target["token_coverage"]
    assert quality["planned"] == quality["graded"] == legacy_summary["task_instances"]
    assert quality["mean_credit"] == legacy_summary["task_score_mean"]
    assert quality["full_credit_tasks"] == legacy_summary["task_successes"]
    assert quality["full_completion_rate"] == legacy_summary["task_success_rate"]
    assert target["request_count"] == legacy_summary["requests"]
    assert target["continuity"]["model_switches"] == legacy_summary["model_switches"]
    assert (
        quality["scripted_tool_loop_switches"]
        == legacy_summary["tool_loop_switch_violations"]
    )
    assert quality["scripted_phase_counts"] == legacy_summary["phase_counts"]
    assert quality["observed_router_phase_counts"] == {
        PROVIDER_STATE_PHASE: len(legacy_rows)
    }
    assert [
        {
            key: value
            for key, value in request["body"].items()
            if key not in {"stream", "stream_options"}
        }
        for request in provider.requests
    ] == [request["body"] for request in legacy_requests]
    assert all(request["body"]["stream"] is True for request in provider.requests)
    assert all(
        "MODEL ANSWER NOT USED AS HISTORY" not in canonical(request["body"]["messages"])
        for request in provider.requests
    )
    run_id = service.store.list()[0]["id"]
    calls = service.store.calls(run_id)
    rows = service.store.results(run_id)
    assert {
        (row["case_id"].split("/")[-1], int(row["case_id"].split("/")[-2][1:])): row[
            "score"
        ]
        for row in rows
    } == {
        (row["task"], row["task_repetition"]): row["answer_score"]
        for row in legacy_rows
        if row["scored_turn"]
    }
    sessions: dict[str, set[str]] = {}
    for call, request, turn in zip(
        calls, provider.requests, quality["turns"], strict=True
    ):
        sessions.setdefault(call["case_id"], set()).add(call["session_id"])
        assert call["session_id"] == request["session_id"] == turn["session_id"]
        assert call["id"] == turn["call_id"]
        assert turn["observed_phase"] == PROVIDER_STATE_PHASE
        assert turn["phase_source"] == "router"
        assert turn["selected_model"] in {"model-a", "model-b"}
    assert all(len(ids) == 1 for ids in sessions.values())
    assert len(set.union(*sessions.values())) == quality["planned"]
    assert all(
        target[key] is None
        for key in ("accuracy", "accuracy_ci95", "macro_accuracy", "sr_bench_score")
    )
    assert target["continuity"]["switched_accuracy"] is None
    assert (
        json.loads((service.store.root / "runs" / run_id / "report.json").read_text())[
            "summary"
        ]
        == report["summary"]
    )
    assert run_cli(service=service, value=value).returncode == 0
    assert len(provider.requests) == legacy_summary["requests"]
    print(
        json.dumps(
            {
                "suite": suite,
                "repetitions": repetitions,
                "native": {
                    key: legacy_summary[key]
                    for key in (
                        "requests",
                        "task_instances",
                        "task_score_mean",
                        "task_success_rate",
                        "model_switches",
                    )
                },
                "migrated": {
                    key: quality[key]
                    for key in (
                        "planned",
                        "graded",
                        "mean_credit",
                        "full_completion_rate",
                        "scripted_phase_counts",
                        "observed_router_phase_counts",
                    )
                },
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("fault", ["http", "length", "buffered", "incomplete_sse"])
def test_cli_failures_leave_grades_missing_and_stop_dispatch(
    provider: Provider, service: Server, fault: str
) -> None:
    provider.fault = fault
    provider.fault_on_request = 5
    result = run_cli(
        service=service, value=document(root=service.store.root, provider=provider)
    )
    assert result.returncode == 2, result.stderr
    target = json.loads(result.stdout)["summary"]["targets"][0]
    quality = target["token_coverage"]
    assert (
        quality["planned"],
        quality["graded"],
        quality["failed"],
        quality["pending"],
    ) == (4, 1, 1, 2)
    assert quality["mean_credit"] is None
    assert quality["full_completion_rate"] == 1 / 4
    assert len(provider.requests) == 5
    run_id = service.store.list()[0]["id"]
    failed = next(
        row for row in service.store.results(run_id) if row["status"] == "failed"
    )
    assert failed.get("score") is None
    assert failed.get("correct") is None
    assert sum(turn["status"] == "unattempted" for turn in quality["turns"]) == 7
    assert quality["turns"][4]["finish_reason"] == (
        "length" if fault == "length" else "stop" if fault == "incomplete_sse" else None
    )
    assert target["cost_usd"] is None
    print(
        json.dumps(
            {
                "fault": fault,
                "requests": len(provider.requests),
                "coverage": {
                    key: quality[key]
                    for key in (
                        "planned",
                        "graded",
                        "failed",
                        "pending",
                        "mean_credit",
                        "full_completion_rate",
                    )
                },
                "error": failed["error"],
            },
            sort_keys=True,
        )
    )


def test_output_limit_on_a_full_credit_final_still_leaves_the_grade_missing(
    provider: Provider, service: Server
) -> None:
    provider.fault = "length"
    provider.fault_on_request = 3
    result = run_cli(
        service=service,
        value=document(root=service.store.root, provider=provider),
    )
    assert result.returncode == 2, result.stderr
    run_id = service.store.list()[0]["id"]
    final = service.store.calls(run_id)[-1]["final"]
    score, _ = native.score_answer(
        content=final, expected_terms=native.task_specs()[0].turns[-1].expected_terms
    )
    assert score == 1.0
    assert service.store.results(run_id)[0].get("score") is None
    quality = json.loads(result.stdout)["summary"]["targets"][0]["token_coverage"]
    assert quality["graded"] == 0
    assert quality["mean_credit"] is None
    assert len(provider.requests) == 3


def test_cli_cancellation_retains_partial_turns_without_grades(
    provider: Provider, service: Server
) -> None:
    provider.fault = "cancel"
    result = run_cli(
        service=service,
        value=document(root=service.store.root, provider=provider),
        detach=True,
    )
    assert result.returncode == 0, result.stderr
    run_id = json.loads(result.stdout)["id"]
    assert provider.entered.wait(timeout=10)
    cancel = benchmark_cli(service, "cancel", run_id)
    assert cancel.returncode == 0, cancel.stderr
    service.engine.threads[run_id].join(timeout=10)
    provider.release.set()
    assert not service.engine.threads[run_id].is_alive()
    assert service.store.get(run_id)["status"] == "cancelled"
    report = benchmark_cli(service, "report", run_id)
    assert report.returncode == 0, report.stderr
    quality = json.loads(report.stdout)["summary"]["targets"][0]["token_coverage"]
    assert (
        quality["planned"],
        quality["graded"],
        quality["cancelled"],
        quality["pending"],
    ) == (4, 0, 1, 3)
    assert quality["mean_credit"] is None
    assert len(provider.requests) == 2
    assert service.store.results(run_id)[0].get("score") is None
    print(
        json.dumps(
            {
                "status": service.store.get(run_id)["status"],
                "requests": len(provider.requests),
                "coverage": {
                    key: quality[key]
                    for key in (
                        "planned",
                        "graded",
                        "cancelled",
                        "pending",
                        "mean_credit",
                    )
                },
            },
            sort_keys=True,
        )
    )


def test_missing_usage_models_and_router_phases_remain_unknown(
    provider: Provider, service: Server
) -> None:
    provider.omit_usage = provider.omit_model = provider.omit_phase = True
    result = run_cli(
        service=service, value=document(root=service.store.root, provider=provider)
    )
    assert result.returncode == 0, result.stderr
    target = json.loads(result.stdout)["summary"]["targets"][0]
    assert target["token_coverage"]["graded"] == 4
    assert target["tokens"] is target["cost_usd"] is target["total_spend_usd"] is None
    assert target["cost_complete"] is False
    assert target["selected_models"] == {}
    assert target["continuity"]["unknown_model_requests"] == 12
    assert target["token_coverage"]["observed_router_phase_counts"] == {}
    assert all(
        turn["selected_model"] is None and turn["phase_source"] == "request"
        for turn in target["token_coverage"]["turns"]
    )
    print(
        json.dumps(
            {
                "missing_usage_and_selection": {
                    key: target[key]
                    for key in (
                        "tokens",
                        "cost_usd",
                        "total_spend_usd",
                        "selected_models",
                    )
                },
                "graded": target["token_coverage"]["graded"],
                "unknown_model_requests": target["continuity"][
                    "unknown_model_requests"
                ],
            },
            sort_keys=True,
        )
    )


def test_missing_usage_fails_closed_when_priced_execution_is_required(
    provider: Provider, service: Server
) -> None:
    provider.omit_usage = True
    value = document(root=service.store.root, provider=provider)
    value["cost_policy"] = "require_priced"
    value["targets"][0]["prices"] = {
        model: {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        for model in ("requested-entrypoint", "model-a", "model-b")
    }
    result = run_cli(service=service, value=value)
    assert result.returncode == 2, result.stderr
    target = json.loads(result.stdout)["summary"]["targets"][0]
    assert target["token_coverage"]["graded"] == 0
    assert target["token_coverage"]["mean_credit"] is None
    assert target["cost_usd"] is None
    assert len(provider.requests) == 1


@pytest.mark.parametrize("channel", ["reasoning", "thinking_tags"])
def test_hidden_content_cannot_earn_token_credit(
    provider: Provider, service: Server, channel: str
) -> None:
    provider.hidden_answer = channel
    result = run_cli(
        service=service, value=document(root=service.store.root, provider=provider)
    )
    assert result.returncode == 0, result.stderr
    quality = json.loads(result.stdout)["summary"]["targets"][0]["token_coverage"]
    assert quality["graded"] == 4
    assert quality["mean_credit"] == quality["full_completion_rate"] == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "script",
        "prompt",
        "answer",
        "kind",
        "version",
        "budget",
        "session",
        "protocol",
        "repetitions",
        "request_fields",
    ],
)
def test_preflight_rejects_contract_drift_before_dispatch(
    provider: Provider, service: Server, field: str
) -> None:
    value = plan(document(root=service.store.root, provider=provider))
    value.pop("dataset")
    if field == "script":
        value["cases"][0]["metadata"]["scripted_task"]["turns"][1][
            "tool_result"
        ] = "changed"
    elif field == "prompt":
        value["cases"][0]["messages"][1]["content"] = "changed"
    elif field == "answer":
        value["cases"][0]["answer"] = ["changed"]
    elif field == "kind":
        value["result_kinds"][agent_task.BENCHMARK_ID] = "binary"
    elif field == "version":
        value["benchmark_options"][agent_task.BENCHMARK_ID][
            "grader_version"
        ] = "changed"
    elif field == "budget":
        value["limits"]["max_calls_per_case"] = 2
    elif field == "session":
        value["targets"][0]["session_mode"] = "stateless"
    elif field == "protocol":
        value["benchmark_options"][agent_task.BENCHMARK_ID]["protocol"] = "responses"
    elif field == "request_fields":
        value["cases"][0]["response_format"] = {"type": "json_object"}
    else:
        value["cases"][0]["metadata"]["task_repetition"] = True
    with pytest.raises(ValueError):
        service.engine.start(value)
    assert provider.requests == []


def test_mixed_quality_and_binary_only_paths_remain_separate(
    provider: Provider, service: Server
) -> None:
    value = plan(document(root=service.store.root, provider=provider))
    value.pop("dataset")
    value.pop("adapter_versions")
    value.pop("plan_sha256")
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
    assert "token_coverage" not in target
    assert all(
        target[key] is None for key in ("accuracy", "macro_accuracy", "sr_bench_score")
    )
    assert (
        next(row for row in report["benchmarks"] if row["benchmark"] == "mmlu-pro")[
            "accuracy"
        ]
        == 1
    )
    assert (
        next(
            row
            for row in report["benchmarks"]
            if row["benchmark"] == agent_task.BENCHMARK_ID
        )["token_coverage"]["graded"]
        == 4
    )
    run_id = service.store.list()[0]["id"]
    for operation in (
        lambda: compare(service.store, run_id, run_id),
        lambda: replay(service.store, run_id, run_id),
        lambda: regrade(service.store, run_id),
        lambda: export_training(service.store, run_id),
    ):
        with pytest.raises(ValueError, match="binary results only"):
            operation()


@pytest.mark.parametrize("suite", ["smoke", "long-horizon", "all"])
def test_preparation_freezes_native_order_repetitions_and_shared_kernel(
    tmp_path: Path, suite: str
) -> None:
    prepared = agent_task.prepare(store=tmp_path, suite=suite, task_repetitions=2)
    assert prepared == agent_task.prepare(
        store=tmp_path, suite=suite, task_repetitions=2
    )
    cases = [
        json.loads(line) for line in Path(prepared["path"]).read_text().splitlines()
    ]
    tasks = native.task_specs(suite=suite)
    assert [case["metadata"]["scripted_task"]["name"] for case in cases] == [
        task.name for task in tasks
    ] * 2
    assert [case["metadata"]["task_repetition"] for case in cases] == [0] * len(
        tasks
    ) + [1] * len(tasks)
    assert (
        cases[0]["metadata"]["task_identity"]
        == cases[len(tasks)]["metadata"]["task_identity"]
    )
    assert [
        case["metadata"]["scripted_task"] for case in cases[: len(tasks)]
    ] == json.loads(json.dumps([asdict(task) for task in tasks]))
    assert native.TaskSpec is agent_task_kernel.TaskSpec
    assert native.build_messages is agent_task_kernel.build_messages
    assert native.score_answer is agent_task_kernel.score_answer
    entry = next(
        item
        for item in DatasetReader(tmp_path).selection("smoke")["benchmarks"]
        if item["id"] == agent_task.BENCHMARK_ID
    )
    assert entry["reason_code"] == "local_preparation_required"


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5, MAX_ROWS])
def test_preparation_rejects_unbounded_or_invalid_repetitions(
    tmp_path: Path, repetitions: Any
) -> None:
    with pytest.raises(ValueError, match="repetitions"):
        agent_task.prepare(store=tmp_path, task_repetitions=repetitions)
    assert not (tmp_path / "datasets").exists()


def test_preparation_does_not_infer_a_holdout_or_unknown_suite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke/quick"):
        agent_task.prepare(store=tmp_path, profile="standard")
    with pytest.raises(ValueError, match="smoke, long-horizon or all"):
        agent_task.prepare(store=tmp_path, suite="unknown")
    assert not (tmp_path / "datasets").exists()


def test_preparation_and_discovery_work_outside_checkout_and_in_copied_cli(
    tmp_path: Path,
) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.sr_bench.agent_task",
            "--suite",
            "all",
            "--task-repetitions",
            "2",
            "--store",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout)["case_count"] == 64
    copied = tmp_path / "copied"
    shutil.copytree(
        ROOT / "src/vllm-sr/cli",
        copied / "cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cli.sr_bench.adapters import get_adapter; print(get_adapter('agent-task-chat-v1').result_kind)",
        ],
        cwd=copied,
        env={**env, "PYTHONPATH": str(copied)},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "token-coverage-v1"
