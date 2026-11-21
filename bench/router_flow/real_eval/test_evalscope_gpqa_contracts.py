# ruff: noqa: PLR2004
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench.contracts import plan
from cli.sr_bench.evalscope_gpqa_worker import execute, prepare_cases
from cli.sr_bench.router_flow_gpqa import BENCHMARK, PROTOCOL
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.models.utils.openai import openai_chat_messages
from evalscope.utils.model_utils import seed_everything

from bench.router_flow.real_eval.export_sr_bench_gpqa import build_manifest
from bench.router_flow.real_eval.run_evalscope_suite import DEFAULT_SUITE


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    rows = [
        {
            "Question": f"Synthetic question {index}?",
            "Correct Answer": f"Correct [title]  choice {index}",
            "Incorrect Answer 1": "First [citation] choice",
            "Incorrect Answer 2": "Second choice",
            "Incorrect Answer 3": "Third choice",
        }
        for index in range(24)
    ]
    path = tmp_path / "gpqa.csv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def target_manifest() -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "targets": [
            {
                "id": "fixture",
                "kind": "single",
                "base_url": "http://127.0.0.1:1/v1",
                "model": "fixture",
            }
        ],
        "cost_policy": "capability_only",
    }


def test_native_order_choices_and_prompts_are_preserved(
    source: tuple[Path, list[dict[str, str]]],
) -> None:
    path, rows = source
    cases = prepare_cases(source_path=path, limit=20)
    assert len(cases) == 20
    seed_everything(42)
    native = get_benchmark(name="gpqa_diamond", config=TaskConfig())
    for index, case in enumerate(cases):
        sample = native.record_to_sample(rows[index])
        expected = openai_chat_messages(
            native.process_sample_str_input(sample=sample, subset="default")
        )
        assert case["messages"] == expected
        assert case["answer"] == sample.target
        assert case["metadata"]["choices"] == sample.choices
        assert case["metadata"]["source_index"] == index
        assert case["benchmark"] == BENCHMARK
        assert "Correct Answer" not in json.dumps(case["messages"])
    assert prepare_cases(source_path=path, limit=20) == cases
    assert prepare_cases(source_path=path, limit=3) == cases[:3]


@pytest.mark.parametrize("limit", [0, -1, 21, True])
def test_invalid_smoke_limit_is_rejected(
    source: tuple[Path, list[dict[str, str]]], limit: int
) -> None:
    with pytest.raises(ValueError, match="limit"):
        prepare_cases(source_path=source[0], limit=limit)


def test_short_source_is_not_filled_or_silently_shrunk(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        prepare_cases(source_path=path, limit=20)


@pytest.mark.parametrize(
    "final",
    [
        "ANSWER: A",
        "ANSWER: A\nANSWER: B",
        "answer: a",
        "The possibilities include B, but I choose A",
        "",
    ],
)
def test_native_grader_receives_only_the_completed_final(
    source: tuple[Path, list[dict[str, str]]], final: str
) -> None:
    case = prepare_cases(source_path=source[0], limit=1)[0]
    calls: list[list[dict[str, Any]]] = []

    def bridge(messages: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(messages)
        return {
            "final": final,
            "reasoning": f"ANSWER: {case['answer']}",
            "model": "fixture",
            "finish_reason": "stop",
            "output_complete": True,
        }

    result = execute(request={"case": case}, bridge=bridge)
    native = get_benchmark(name="gpqa_diamond", config=TaskConfig())
    sample = Sample(
        input=case["messages"],
        choices=case["metadata"]["choices"],
        target=case["answer"],
    )
    expected = native.calculate_metrics(
        TaskState(
            model="fixture",
            sample=sample,
            output=ModelOutput.from_content(model="fixture", content=final),
            completed=True,
        )
    )
    assert calls == [case["messages"]]
    assert result["score"] == expected.score.value["acc"]
    assert result["correct"] == (expected.score.value["acc"] == 1)
    assert result["details"]["grader_version"] == PROTOCOL


def test_truncated_output_never_reaches_native_grading(
    source: tuple[Path, list[dict[str, str]]],
) -> None:
    case = prepare_cases(source_path=source[0], limit=1)[0]

    def bridge(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "final": f"ANSWER: {case['answer']}",
            "model": "fixture",
            "finish_reason": "length",
            "output_complete": False,
        }

    with pytest.raises(CallFailure, match="incomplete"):
        execute(request={"case": case}, bridge=bridge)


def test_export_freezes_native_settings_and_discloses_the_derived_protocol(
    source: tuple[Path, list[dict[str, str]]],
) -> None:
    manifest = build_manifest(
        base=target_manifest(),
        suite_path=DEFAULT_SUITE,
        source_path=source[0],
        revision="synthetic-fixture-v1",
        limit=3,
    )
    assert manifest["sampling"] == {"temperature": 0, "max_tokens": 32768}
    assert manifest["limits"]["concurrency"] == 2
    assert manifest["limits"]["total_timeout_s"] == 900
    assert manifest["limits"]["max_calls_per_case"] == 1
    assert manifest["seed"] == 42
    assert manifest["profile"] == "smoke"
    assert len(manifest["cases"]) == 3
    options = manifest["benchmark_options"][BENCHMARK]
    assert options["custom_subset"] is True
    assert options["source"]["revision"] == "synthetic-fixture-v1"
    assert options["selection"] == "native-ordered-prefix"
    assert "not native benchmark parity" in " ".join(manifest["limitations"])


@pytest.mark.parametrize(
    "override",
    [
        {"profile": "standard"},
        {"seed": 20260918},
        {"sampling": {"temperature": 1}},
        {"limits": {"max_output_tokens": 4096}},
        {"output_policy": "native"},
    ],
)
def test_export_rejects_conflicting_protocol_inputs(
    source: tuple[Path, list[dict[str, str]]], override: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        build_manifest(
            base={**target_manifest(), **override},
            suite_path=DEFAULT_SUITE,
            source_path=source[0],
            revision="synthetic-fixture-v1",
        )


class _Target(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode: str, cases: list[dict[str, Any]]) -> None:
        self.mode = mode
        self.requests: list[dict[str, Any]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.answers = {
            json.dumps(case["messages"], sort_keys=True): case["answer"]
            for case in cases
        }
        self.first = json.dumps(cases[0]["messages"], sort_keys=True)
        super().__init__(("127.0.0.1", 0), _TargetHandler)


class _TargetHandler(BaseHTTPRequestHandler):
    server: _Target

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        self.server.started.set()
        identity = json.dumps(body["messages"], sort_keys=True)
        if self.server.mode == "partial" and identity != self.server.first:
            assert self.server.release.wait(30)
            self.send_error(503)
            return
        if self.server.mode == "http_error":
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        response = {
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": f"ANSWER: {self.server.answers[identity]}",
                        "reasoning_content": "ANSWER: Z",
                    },
                    "finish_reason": (
                        "length" if self.server.mode == "truncated" else "stop"
                    ),
                }
            ],
        }
        if self.server.mode != "no_usage":
            response["usage"] = {"prompt_tokens": 10, "completion_tokens": 3}
        try:
            self.wfile.write(("data: " + json.dumps(response) + "\n\n").encode())
            self.wfile.flush()
            if self.server.mode == "cancel":
                assert self.server.release.wait(30)
            if self.server.mode != "incomplete":
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextmanager
def running(server: ThreadingHTTPServer) -> Iterator[ThreadingHTTPServer]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        if isinstance(server, _Target):
            server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        if isinstance(server, Server):
            for worker in server.engine.threads.values():
                worker.join(timeout=10)
                assert not worker.is_alive()


def cli(*args: str, url: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.main",
            "benchmark",
            "--url",
            url,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "no_usage",
        "truncated",
        "http_error",
        "incomplete",
        "partial",
        "cancel",
    ],
)
def test_real_cli_bridge_and_report(
    source: tuple[Path, list[dict[str, str]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    for name in ("SR_BENCH_URL", "SR_BENCH_TOKEN", "SR_BENCH_TOKEN_ENV"):
        monkeypatch.delenv(name, raising=False)
    assert os.environ.get(
        "SR_BENCH_HOME"
    ), "Set the explicitly installed test harness home"
    cases = prepare_cases(source_path=source[0], limit=4)
    store = Store(tmp_path / "store")
    target = _Target(mode=mode, cases=cases)
    service = Server(("127.0.0.1", 0), store)
    with running(target), running(service):
        url = f"http://127.0.0.1:{service.server_port}"
        base = target_manifest()
        base["targets"][0]["base_url"] = f"http://127.0.0.1:{target.server_port}/v1"
        base["targets"][0]["prices"] = {
            "fixture": {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        }
        base_path = tmp_path / "base.json"
        base_path.write_text(json.dumps(base))
        draft_path, frozen_path = tmp_path / "draft.json", tmp_path / "frozen.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "bench.router_flow.real_eval.export_sr_bench_gpqa",
                "--source-path",
                str(source[0]),
                "--revision",
                "synthetic-fixture-v1",
                "--manifest",
                str(base_path),
                "--limit",
                "4",
                "--output",
                str(draft_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        frozen = cli(
            "plan", "--manifest", str(draft_path), "--output", str(frozen_path), url=url
        )
        assert frozen["total"] == 4
        assert target.requests == []
        run = cli(
            "run",
            "--manifest",
            str(frozen_path),
            "--detach",
            "--idempotency-key",
            mode,
            url=url,
        )
        if mode == "partial":
            deadline = time.monotonic() + 30
            while not any(row.get("correct") for row in store.results(run["id"])):
                assert time.monotonic() < deadline, "First case did not complete"
                time.sleep(0.01)
            target.release.set()
        elif mode == "cancel":
            assert target.started.wait(20)
            cli("cancel", run["id"], url=url)
            target.release.set()
        service.engine.threads[run["id"]].join(timeout=30)
        assert not service.engine.threads[run["id"]].is_alive()
        report = cli("report", run["id"], url=url)
        metric = report["summary"]["targets"][0]
        assert metric["total"] == 4
        assert metric["accuracy"] == metric["correct"] / 4
        assert metric["sr_bench_score"] is None
        assert "extension diagnostic" in metric["score_scope"]
        assert report["provenance"]["adapter_versions"] == {BENCHMARK: PROTOCOL}
        assert "not native benchmark parity" in " ".join(report["limitations"])
        calls = store.calls(run["id"])
        assert len(calls) == len(target.requests)
        assert len({row["case_id"] for row in calls}) == len(calls)
        for request in target.requests:
            assert request["stream"] is True
            assert request["temperature"] == 0
            assert request["max_tokens"] == 32768
            assert json.dumps(request["messages"], sort_keys=True) in target.answers
        assert (
            cli(
                "run",
                "--manifest",
                str(frozen_path),
                "--detach",
                "--idempotency-key",
                mode,
                url=url,
            )["id"]
            == run["id"]
        )
        assert len(store.calls(run["id"])) == len(calls)
        if mode in {"success", "no_usage"}:
            assert report["status"] == "completed"
            assert metric["accuracy"] == 1
            assert len(calls) == 4
            assert all(
                row["details"]["grader_version"] == PROTOCOL
                for row in store.results(run["id"])
            )
        elif mode == "truncated":
            assert metric["correct"] == 0
            assert metric["output_diagnostics"]["output_limit_cases"] == 4
            assert all(
                "grader_version" not in row["details"]
                for row in store.results(run["id"])
            )
        elif mode == "partial":
            assert metric["correct"] == 1
            assert metric["accuracy"] == 0.25
            assert report["status"] != "completed"
        else:
            assert metric["complete"] is False
            assert report["status"] != "completed"
        if mode == "no_usage":
            assert metric["cost_usd"] is None
            assert metric["tokens"] is None
        if mode == "success":
            assert metric["tokens"]["input_tokens"] == 40
            assert metric["tokens"]["output_tokens"] == 12
            assert metric["cost_usd"] == pytest.approx(52 / 1_000_000)
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "fixture_mode": mode,
                    "status": report["status"],
                    "planned": metric["total"],
                    "requests": len(calls),
                    "correct": metric["correct"],
                    "accuracy": metric["accuracy"],
                    "cost_usd": metric["cost_usd"],
                },
                sort_keys=True,
            )
        )


@pytest.mark.parametrize("candidate_mode", ["live", "preview"])
def test_native_candidate_plan_preserves_disclosures_without_dispatch(
    source: tuple[Path, list[dict[str, str]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_mode: str,
) -> None:
    for name in ("SR_BENCH_URL", "SR_BENCH_TOKEN", "SR_BENCH_TOKEN_ENV"):
        monkeypatch.delenv(name, raising=False)
    cases = prepare_cases(source_path=source[0], limit=1)
    store = Store(tmp_path / "store")
    target = _Target(mode="success", cases=cases)
    service = Server(("127.0.0.1", 0), store)
    with running(target), running(service):
        url = f"http://127.0.0.1:{service.server_port}"
        base = target_manifest()
        base["targets"][0]["base_url"] = f"http://127.0.0.1:{target.server_port}/v1"
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                build_manifest(
                    base=base,
                    suite_path=DEFAULT_SUITE,
                    source_path=source[0],
                    revision="synthetic-fixture-v1",
                    limit=1,
                )
            )
        )
        report = cli("run", "--manifest", str(baseline_path), url=url)
        assert report["status"] == "completed"
        baseline = store.get(report["run_id"])
        service.engine.threads[baseline["id"]].join(timeout=10)
        assert len(target.requests) == 1
        target_path = tmp_path / "target.json"
        target_path.write_text(
            json.dumps(
                [
                    {
                        **base["targets"][0],
                        "id": "candidate",
                        "kind": "mom",
                        "config_hash": "a" * 64,
                        "preview_url": f"http://127.0.0.1:{target.server_port}/api/v1/routing/preview",
                    }
                ]
            )
        )
        cli(
            "--store",
            str(store.root),
            "target",
            "register",
            "--file",
            str(target_path),
            url=url,
        )
        original = list(store.db.iterdump())
        candidate = cli(
            "candidate-plan",
            baseline["id"],
            "--target",
            "candidate",
            "--mode",
            candidate_mode,
            url=url,
        )
        assert candidate["model_requests"] == 0
        assert candidate["manifest"]["mode"] == candidate_mode
        for key in (
            "limitations",
            "case_sha256",
            "adapter_versions",
            "limits",
            "sampling",
            "output_policy",
        ):
            assert candidate["manifest"][key] == baseline["manifest"][key]
        assert len(target.requests) == 1
        assert len(store.calls(baseline["id"])) == 1
        assert list(store.db.iterdump()) == original
        if candidate_mode == "live":
            without_disclosures = {
                key: value
                for key, value in candidate["manifest"].items()
                if key != "limitations"
            }
            with pytest.raises(ValueError, match="frozen smoke protocol"):
                plan(without_disclosures)
        print(
            json.dumps(
                {
                    "candidate_mode": candidate_mode,
                    "baseline_status": baseline["status"],
                    "disclosures_preserved": True,
                    "candidate_model_requests": candidate["model_requests"],
                    "total_fixture_requests": len(target.requests),
                },
                sort_keys=True,
            )
        )


def test_missing_native_harness_fails_before_dispatch(
    source: tuple[Path, list[dict[str, str]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_manifest(
        base=target_manifest(),
        suite_path=DEFAULT_SUITE,
        source_path=source[0],
        revision="synthetic-fixture-v1",
        limit=1,
    )
    monkeypatch.setenv("SR_BENCH_GPQA_EVALSCOPE_ROOT", str(tmp_path / "missing"))
    monkeypatch.setenv(
        "SR_BENCH_GPQA_EVALSCOPE_PYTHON", str(tmp_path / "missing-python")
    )
    with pytest.raises(ValueError, match="Install pinned"):
        plan(manifest)


def test_managed_copy_only_worker_uses_the_separate_native_environment(
    source: tuple[Path, list[dict[str, str]]], tmp_path: Path
) -> None:
    base_python = os.environ["SR_BENCH_TEST_BASE_PYTHON"]
    root = Path(__file__).resolve().parents[3]
    copied = tmp_path / "managed"
    shutil.copytree(
        root / "src/vllm-sr/cli",
        copied / "cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(root / "src/vllm-sr/pyproject.toml", copied / "pyproject.toml")
    code = """
import importlib.abc
import sys
from importlib.metadata import EntryPoints

class NoOptionalFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"evalscope", "bench"}:
            raise AssertionError("Managed service imported an optional framework")

sys.meta_path.insert(0, NoOptionalFramework())
from cli.sr_bench import adapters
adapters.entry_points = lambda **kwargs: EntryPoints()
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
server = Server(("127.0.0.1", 0), Store(sys.argv[1]))
print(server.server_port, flush=True)
server.serve_forever()
"""
    cases = prepare_cases(source_path=source[0], limit=1)
    with running(_Target(mode="success", cases=cases)) as target:
        base = target_manifest()
        base["targets"][0]["base_url"] = f"http://127.0.0.1:{target.server_port}/v1"
        manifest = build_manifest(
            base=base,
            suite_path=DEFAULT_SUITE,
            source_path=source[0],
            revision="synthetic-fixture-v1",
            limit=1,
        )
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(plan(manifest)))
        env = {**os.environ, "PYTHONPATH": str(copied)}
        for name in ("SR_BENCH_TOKEN", "SR_BENCH_TOKEN_ENV", "SR_BENCH_URL"):
            env.pop(name, None)
        with (tmp_path / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [base_python, "-u", "-c", code, str(tmp_path / "store")],
                cwd=copied,
                env=env,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )
            try:
                assert process.stdout is not None
                port = int(process.stdout.readline())
                url = f"http://127.0.0.1:{port}"
                assert BENCHMARK in {
                    row["id"] for row in cli("catalog", url=url)["benchmarks"]
                }
                report = cli("run", "--manifest", str(path), url=url)
                assert report["summary"]["targets"][0]["accuracy"] == 1
                assert report["summary"]["targets"][0]["sr_bench_score"] is None
                assert len(target.requests) == 1
                print(
                    json.dumps(
                        {
                            "managed_copy_only": True,
                            "base_evalscope_imports": 0,
                            "entry_points": 0,
                            "requests": len(target.requests),
                            "status": report["status"],
                        },
                        sort_keys=True,
                    )
                )
            finally:
                process.terminate()
                process.wait(timeout=10)
