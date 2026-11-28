from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from cli.sr_bench import mrcr
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.canonical import canonical
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS
from cli.sr_bench.grading import require_binary_results, validate_result
from cli.sr_bench.report import metric
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store

ROOT: Path = Path(__file__).resolve().parents[3]


class FixtureEncoding:
    def encode(self, text: str) -> range:
        return range(len(text))


def _record(
    *,
    source_id: str,
    prompt: list[dict[str, str]],
    answer: str,
    n_needles: int,
    prefix: str = "prefix ",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "prompt": json.dumps(prompt),
        "answer": answer,
        "random_string_to_prepend": prefix,
        "n_needles": n_needles,
        "desired_msg_index": 1,
        "total_messages": len(prompt),
        "n_chars": sum(len(message["content"]) for message in prompt),
    }


def test_mrcr_is_opt_in_and_keeps_native_fractional_credit() -> None:
    adapter = get_adapter(mrcr.BENCHMARK_ID)
    assert adapter.result_kind == mrcr.RESULT_KIND
    assert adapter.id not in BENCHMARK_WEIGHTS
    assert adapter.requires_answer

    cases = [
        ("prefix abcX", "prefix abc", "prefix ", 6 / 7),
        ("absent", "prefix answer", "prefix ", 0.0),
        ("prefix answer", "prefix answer", "prefix ", 1.0),
        ("abc", "xabc", "a", 2 / 3),
        ("abc", "abc", "", 1.0),
    ]
    for prediction, reference, prefix, expected in cases:
        assert (
            mrcr.native_grade(
                prediction=prediction,
                reference=reference,
                prefix=prefix,
            )
            == expected
        )


def test_native_suite_row_matches_frozen_mrcr_selection() -> None:
    suite_path = ROOT / "bench/router_flow/real_eval/evalscope_suite.yaml"
    suite = yaml.safe_load(suite_path.read_text())
    row = next(item for item in suite["benchmarks"] if item["id"] == "openai_mrcr_32k")
    assert (row["dataset"], row["metric"]) == (
        mrcr.NATIVE_ADAPTER_NAME,
        mrcr.MRCR_METRIC,
    )
    assert row["smoke_limit"] == mrcr.SMOKE_RAW_LIMIT
    assert row["formal_limit"] == 0
    assert row["smoke_dataset_args"]["extra_params"] == {
        "min_context_size": mrcr.SMOKE_MIN_CONTEXT_TOKENS,
        "max_context_size": mrcr.SMOKE_MAX_CONTEXT_TOKENS,
        "needle_count": [mrcr.SMOKE_NEEDLE_COUNT],
    }
    assert row["dataset_args"]["extra_params"] == {
        "max_context_size": mrcr.FORMAL_MAX_CONTEXT_TOKENS,
        "needle_count": [mrcr.FORMAL_NEEDLE_COUNT],
    }


def test_mrcr_result_is_not_binary_or_a_published_weighted_score() -> None:
    result = mrcr.grade(
        case={
            "answer": "prefix abc",
            "metadata": {
                "mrcr": {
                    "random_string_to_prepend": "prefix ",
                    "source_index": 0,
                    "bin_index": 0,
                    "raw_input_tok_cnt": 32,
                    "answer_tok_cnt": 10,
                    "total_tok_cnt": 42,
                }
            },
        },
        prediction="\r\n prefix abcX",
    )
    validate_result(kind=mrcr.RESULT_KIND, result=result)
    assert result["score"] == 6 / 7
    assert result["answer"] == "prefix abcX"
    assert result["correct"] is None
    assert result["details"]["mrcr"]["native_metric"] == mrcr.MRCR_METRIC
    with pytest.raises(ValueError, match="binary results only"):
        require_binary_results(
            manifest={
                "cases": [{"benchmark": mrcr.BENCHMARK_ID}],
                "result_kinds": {mrcr.BENCHMARK_ID: mrcr.RESULT_KIND},
            },
            operation="replay",
        )

    invalid = {**result, "score": True}
    with pytest.raises(ValueError):
        validate_result(kind=mrcr.RESULT_KIND, result=invalid)
    invalid = {**result, "correct": False}
    with pytest.raises(ValueError):
        validate_result(kind=mrcr.RESULT_KIND, result=invalid)


@pytest.mark.parametrize("leading_whitespace", ["\t", "\v", "\f", "\u2003"])
def test_mrcr_default_filter_strips_native_whitespace(
    leading_whitespace: str,
) -> None:
    result = mrcr.grade(
        case={
            "answer": "prefix answer",
            "metadata": {
                "mrcr": {
                    "random_string_to_prepend": None,
                    "source_index": 0,
                    "bin_index": 0,
                    "raw_input_tok_cnt": 32,
                    "total_tok_cnt": 42,
                }
            },
        },
        prediction=f"{leading_whitespace}prefix answer",
    )
    assert result["score"] == 1.0
    assert result["answer"] == "prefix answer"


def test_mixed_mrcr_and_binary_cohort_has_no_combined_accuracy() -> None:
    mrcr_result = mrcr.grade(
        case={
            "answer": "prefix abc",
            "metadata": {
                "mrcr": {
                    "random_string_to_prepend": "prefix ",
                    "source_index": 0,
                    "bin_index": 0,
                    "raw_input_tok_cnt": 32,
                    "total_tok_cnt": 42,
                }
            },
        },
        prediction="prefix abcX",
    )
    results = [
        {
            "case_id": "mrcr-case",
            "benchmark": mrcr.BENCHMARK_ID,
            "status": "completed",
            "latency_s": 1.0,
            **mrcr_result,
        },
        {
            "case_id": "binary-case",
            "benchmark": "mmlu-pro",
            "status": "completed",
            "correct": True,
            "score": 1,
            "answer": "A",
            "latency_s": 1.0,
        },
    ]
    report = metric(
        target_id="subject",
        results=results,
        calls=[],
        total=2,
        planned_case_ids={"mrcr-case", "binary-case"},
        result_kinds={
            mrcr.BENCHMARK_ID: mrcr.RESULT_KIND,
            "mmlu-pro": "binary",
        },
    )
    assert report["result_kind"] == "mixed"
    assert report["correct"] is None
    assert report["accuracy"] is None
    assert report["accuracy_ci95"] is None
    assert report["accuracy_denominator"] == (
        "binary accuracy is unavailable for non-binary or mixed cohorts"
    )


@pytest.mark.parametrize(
    ("profile", "limit", "records", "expected_indices"),
    [
        (
            "smoke",
            None,
            [
                _record(
                    source_id="at-min",
                    prompt=[{"role": "user", "content": "x" * 65536}],
                    answer="a",
                    n_needles=2,
                ),
                _record(
                    source_id="inside",
                    prompt=[
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "x" * 65536 + "y"},
                    ],
                    answer="answer",
                    n_needles=2,
                ),
                _record(
                    source_id="wrong-needle-count",
                    prompt=[{"role": "user", "content": "x" * 70000}],
                    answer="a",
                    n_needles=4,
                ),
                _record(
                    source_id="over-max",
                    prompt=[{"role": "user", "content": "x" * 262145}],
                    answer="a",
                    n_needles=2,
                ),
                _record(
                    source_id="also-inside",
                    prompt=[{"role": "user", "content": "x" * 100000}],
                    answer="answer",
                    n_needles=2,
                ),
                _record(
                    source_id="outside-raw-limit",
                    prompt=[{"role": "user", "content": "x" * 70000}],
                    answer="a",
                    n_needles=2,
                ),
            ],
            [1, 4],
        ),
        (
            "quick",
            None,
            [
                _record(
                    source_id="at-formal-max",
                    prompt=[{"role": "user", "content": "x" * 131072}],
                    answer="a",
                    n_needles=8,
                ),
                _record(
                    source_id="over-formal-max",
                    prompt=[{"role": "user", "content": "x" * 131073}],
                    answer="a",
                    n_needles=8,
                ),
                _record(
                    source_id="wrong-formal-needle-count",
                    prompt=[{"role": "user", "content": "x" * 100}],
                    answer="a",
                    n_needles=2,
                ),
            ],
            [0],
        ),
    ],
)
def test_preparation_preserves_native_raw_limit_and_selection(
    profile: str,
    limit: int | None,
    records: list[dict[str, Any]],
    expected_indices: list[int],
) -> None:
    cases, selection = mrcr.prepare_records(
        records=records,
        profile=profile,
        limit=limit,
        encoding=FixtureEncoding(),
        source_sha256="a" * 64,
        revision="fixture-revision",
    )
    assert [
        case["metadata"]["mrcr"]["source_index"] for case in cases
    ] == expected_indices
    assert all(
        case["messages"] == json.loads(records[index]["prompt"])
        for case, index in zip(cases, expected_indices, strict=True)
    )
    assert selection["raw_rows_scanned"] == (
        min(len(records), 5) if profile == "smoke" else len(records)
    )
    assert selection["source_sha256"] == "a" * 64


def test_preparation_limit_applies_before_native_filters() -> None:
    records = [
        _record(
            source_id=f"row-{index}",
            prompt=[{"role": "user", "content": "x" * 70000}],
            answer="answer",
            n_needles=4 if index == 0 else 8,
        )
        for index in range(6)
    ]
    cases, selection = mrcr.prepare_records(
        records=records,
        profile="smoke",
        limit=1,
        encoding=FixtureEncoding(),
        source_sha256="b" * 64,
        revision="fixture-revision",
    )
    assert cases == []
    assert selection["raw_rows_scanned"] == 1
    with pytest.raises(ValueError, match="smoke limit"):
        mrcr.prepare_records(
            records=records,
            profile="smoke",
            limit=6,
            encoding=FixtureEncoding(),
            source_sha256="b" * 64,
            revision="fixture-revision",
        )


@pytest.mark.parametrize(
    ("token_count", "expected_index"),
    [(4096, 0), (8192, 0), (8193, 1), (16384, 2), (1048576, 7), (1048577, 0)],
)
def test_native_token_bin_edges(token_count: int, expected_index: int) -> None:
    assert mrcr._bin_index(token_count) == expected_index


class Provider(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.responses: dict[str, str] = {}
        self.requests: list[dict[str, Any]] = []
        self.block_on: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.output_limit_on: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        provider = self.server
        assert isinstance(provider, Provider)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = canonical(body["messages"])
        provider.requests.append({"messages": body["messages"], "path": self.path})
        final = provider.responses[prompt]
        if final == provider.block_on:
            provider.entered.set()
            provider.release.wait(timeout=10)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        finish_reason = "length" if final == provider.output_limit_on else "stop"
        event = {
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": final},
                    "finish_reason": finish_reason,
                }
            ],
        }
        try:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


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


def _records() -> list[dict[str, Any]]:
    return [
        _record(
            source_id=identity,
            prompt=[{"role": "user", "content": identity}],
            answer=answer,
            n_needles=8,
        )
        for identity, answer in (
            ("full", "prefix answer"),
            ("zero", "prefix answer"),
            ("fractional", "prefix abc"),
        )
    ]


def _dataset(
    *,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    source = root / "mrcr-fixture.json"
    source.write_text(json.dumps(records))
    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(get_encoding=lambda _name: FixtureEncoding()),
    )
    return mrcr.prepare(
        benchmark=mrcr.BENCHMARK_ID,
        profile="quick",
        store=root,
        source_path=source,
        revision="synthetic-source-revision",
        seed=7,
    )


def _manifest(dataset: dict[str, Any], provider: Provider) -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "name": "MRCR local diagnostic",
        "profile": "quick",
        "seed": 7,
        "cost_policy": "capability_only",
        "dataset": dataset,
        "targets": [
            {
                "id": "subject",
                "kind": "single",
                "model": "fixture-model",
                "base_url": provider.base_url,
            }
        ],
        "benchmark_options": {mrcr.BENCHMARK_ID: {"protocol": mrcr.PROTOCOL_VERSION}},
        "sampling": {"temperature": 0, "max_tokens": 32},
        "limits": {
            "max_calls_per_case": 1,
            "total_timeout_s": 5,
            "idle_timeout_s": 2,
            "case_timeout_s": 15,
            "max_run_seconds": 30,
        },
    }


def _benchmark_cli(service: Server, *args: str) -> subprocess.CompletedProcess[str]:
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
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _prepare_cli_plan(
    *,
    root: Path,
    service: Server,
    provider: Provider,
    dataset: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(dataset, provider)))
    frozen_path = root / "frozen.json"
    result = _benchmark_cli(
        service,
        "plan",
        "--manifest",
        str(manifest_path),
        "--output",
        str(frozen_path),
    )
    assert result.returncode == 0, result.stderr
    return frozen_path, json.loads(frozen_path.read_text())


def test_cli_plan_run_report_preserves_native_mrcr_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    service: Server,
) -> None:
    records = _records()
    provider.responses = {
        canonical(json.loads(record["prompt"])): answer
        for record, answer in zip(
            records,
            ("prefix answer", "wrong", "prefix abcX"),
            strict=True,
        )
    }
    dataset = _dataset(root=tmp_path, monkeypatch=monkeypatch, records=records)
    frozen_path, frozen = _prepare_cli_plan(
        root=tmp_path,
        service=service,
        provider=provider,
        dataset=dataset,
    )
    assert frozen["result_kinds"] == {mrcr.BENCHMARK_ID: mrcr.RESULT_KIND}
    assert mrcr.BENCHMARK_ID not in BENCHMARK_WEIGHTS

    run_result = _benchmark_cli(
        service,
        "run",
        "--manifest",
        str(frozen_path),
        "--idempotency-key",
        "mrcr-local-fixture",
    )
    assert run_result.returncode == 0, run_result.stderr
    run_id = service.store.list()[0]["id"]
    report_result = _benchmark_cli(service, "report", run_id)
    assert report_result.returncode == 0, report_result.stderr
    report = json.loads(report_result.stdout)
    benchmark = report["benchmarks"][0]
    native = benchmark["mrcr"]
    assert native["overall"]["complete"] is True
    assert native["overall"]["planned"] == 3
    assert native["overall"]["graded"] == 3
    assert native["overall"]["score"] == pytest.approx((1 + 0 + 6 / 7) / 3)
    assert [row["score"] for row in service.store.results(run_id)] == [
        1.0,
        0.0,
        6 / 7,
    ]
    assert report["summary"]["targets"][0]["accuracy"] is None
    assert report["summary"]["targets"][0]["sr_bench_score"] is None
    assert len(provider.requests) == 3


def test_output_limit_is_a_missing_mrcr_grade_not_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    service: Server,
) -> None:
    record = _records()[0]
    provider.responses = {canonical(json.loads(record["prompt"])): "prefix answer"}
    provider.output_limit_on = "prefix answer"
    dataset = _dataset(root=tmp_path, monkeypatch=monkeypatch, records=[record])
    frozen_path, _ = _prepare_cli_plan(
        root=tmp_path,
        service=service,
        provider=provider,
        dataset=dataset,
    )
    run = _benchmark_cli(
        service,
        "run",
        "--manifest",
        str(frozen_path),
        "--idempotency-key",
        "mrcr-output-limit",
    )
    assert run.returncode == 2, run.stderr
    run_id = service.store.list()[0]["id"]
    result = service.store.results(run_id)[0]
    assert result["status"] == "failed"
    assert result.get("score") is None
    report_result = _benchmark_cli(service, "report", run_id)
    assert report_result.returncode == 0, report_result.stderr
    report = json.loads(report_result.stdout)
    quality = report["benchmarks"][0]["mrcr"]["overall"]
    assert (quality["planned"], quality["graded"], quality["failed"]) == (1, 0, 1)
    assert quality["complete"] is False
    assert quality["score"] is None


def test_cancelled_mrcr_grade_stays_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    service: Server,
) -> None:
    record = _records()[0]
    provider.responses = {canonical(json.loads(record["prompt"])): "prefix answer"}
    provider.block_on = "prefix answer"
    dataset = _dataset(root=tmp_path, monkeypatch=monkeypatch, records=[record])
    frozen_path, _ = _prepare_cli_plan(
        root=tmp_path,
        service=service,
        provider=provider,
        dataset=dataset,
    )
    run = _benchmark_cli(
        service,
        "run",
        "--manifest",
        str(frozen_path),
        "--idempotency-key",
        "mrcr-cancelled",
        "--detach",
    )
    assert run.returncode == 0, run.stderr
    run_id = json.loads(run.stdout)["id"]
    assert provider.entered.wait(timeout=5)
    cancelled = _benchmark_cli(service, "cancel", run_id)
    assert cancelled.returncode == 0, cancelled.stderr
    provider.release.set()
    deadline = time.monotonic() + 5
    while service.store.get(run_id)["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    result = service.store.results(run_id)[0]
    assert result["status"] == "cancelled"
    assert result.get("score") is None
    report_result = _benchmark_cli(service, "report", run_id)
    assert report_result.returncode == 0, report_result.stderr
    report = json.loads(report_result.stdout)
    quality = report["benchmarks"][0]["mrcr"]["overall"]
    assert (quality["planned"], quality["graded"], quality["cancelled"]) == (1, 0, 1)
    assert quality["complete"] is False
    assert quality["score"] is None
