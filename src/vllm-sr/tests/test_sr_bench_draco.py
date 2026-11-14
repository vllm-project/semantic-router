from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
from cli.sr_bench import draco
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, plan
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.engine import Engine
from cli.sr_bench.grading import WEIGHTED_RUBRIC_RESULT, validate_result
from cli.sr_bench.offline import export_training, regrade, replay
from cli.sr_bench.recovery import recover, recovery_plan
from cli.sr_bench.report import compare, make_report
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset, combine_datasets
from cli.sr_bench.store import Store
from test_sr_bench import wait_run

from bench.grounded_fusion.datasets import load_draco
from bench.grounded_fusion.evaluate import build_parser, run_arm

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "bench/grounded_fusion/testdata/draco_fixture.json"


class Provider(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict[str, Any]] = []
        self.matched = {"call-emergency", "define-compounding"}
        self.subject_answer = "A"
        self.fault: str | None = None
        self.fault_after_subjects = 0

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
        server.requests.append(body)
        model = body["model"]
        judge = model == "judge"
        fault = (
            server.fault
            if sum(request["model"] == "subject" for request in server.requests)
            >= server.fault_after_subjects
            else None
        )
        if fault == "http" and judge:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        content = server.subject_answer
        finish = (
            "length"
            if fault == ("judge_limit" if judge else "subject_limit")
            else "stop"
        )
        if judge:
            ids = re.findall(r'id="([^"]+)"', body["messages"][-1]["content"])
            verdicts = [
                {
                    "id": identity,
                    "matched": identity in server.matched,
                    "evidence": "fixture",
                }
                for identity in ids
            ]
            if fault == "missing":
                verdicts.pop()
            elif fault == "duplicate":
                verdicts.append(verdicts[0])
            elif fault == "unknown":
                verdicts[0]["id"] = "unknown"
            elif fault == "string_boolean":
                verdicts[0]["matched"] = "false"
            content = "malformed" if fault == "malformed" else json.dumps(verdicts)
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2, "cache_creation_tokens": 1},
        }
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "text/event-stream" if body.get("stream") else "application/json",
        )
        self.end_headers()
        if body.get("stream"):
            events = [
                {
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": finish,
                        }
                    ],
                },
                {"model": model, "choices": [], "usage": usage},
            ]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.wfile.write(
                json.dumps(
                    {
                        "model": model,
                        "choices": [
                            {"message": {"content": content}, "finish_reason": finish}
                        ],
                        "usage": usage,
                    }
                ).encode()
            )


@pytest.fixture
def provider() -> Iterator[Provider]:
    server = Provider()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def document(root: Path, provider: Provider) -> dict[str, Any]:
    prepared = draco.prepare(
        source_path=FIXTURE,
        revision="fixture-v1",
        source_partition="fixture-test",
        store=root,
    )
    targets = {
        name: {
            "id": name,
            "kind": "single",
            "model": name,
            "base_url": provider.base_url,
            "prices": {
                name: {"input": 1, "cached_input": 0.1, "cache_write": 2, "output": 3}
            },
        }
        for name in ("subject", "judge")
    }
    return {
        "version": "sr-bench-1.0",
        "profile": "quick",
        "seed": 42,
        "dataset": prepared,
        "targets": [targets["subject"]],
        "auxiliary_targets": {"judge": targets["judge"]},
        "benchmark_options": {
            draco.BENCHMARK_ID: {
                "judge": "judge",
                "grader_version": draco.GRADER_VERSION,
                "batch_size": 2,
            }
        },
        "sampling": {"temperature": 0, "max_tokens": 1024},
        "limits": {
            "max_calls_per_case": 3,
            "total_timeout_s": 5,
            "idle_timeout_s": 2,
            "max_run_seconds": 30,
        },
    }


def run_document(
    root: Path, value: dict[str, Any]
) -> tuple[Store, Engine, dict[str, Any]]:
    store = Store(root)
    engine = Engine(store)
    run = engine.start(value, request_key="draco-once")
    engine.threads[run["id"]].join(timeout=10)
    assert not engine.threads[run["id"]].is_alive()
    assert wait_run(store, run["id"])["status"] in {"completed", "failed"}
    return store, engine, run


def test_draco_is_a_builtin_weighted_diagnostic() -> None:
    adapter = get_adapter("draco-routed-answer-v1")
    assert adapter.result_kind == "weighted-rubric-v1"
    assert adapter.id not in BENCHMARK_WEIGHTS


def test_same_target_legacy_and_journaled_grades_match(
    tmp_path: Path, provider: Provider
) -> None:
    legacy_dir = tmp_path / "legacy"
    args = build_parser().parse_args(
        [
            "--endpoint",
            provider.base_url.removesuffix("/v1"),
            "--draco-path",
            str(FIXTURE),
            "--arm",
            "off",
            "--fusion-model",
            "subject",
            "--grader-base-url",
            provider.base_url,
            "--grader-model",
            "judge",
            "--grader-batch-size",
            "2",
            "--output-dir",
            str(legacy_dir),
        ]
    )
    legacy_summary = run_arm(args)
    legacy = {
        row["id"]: row
        for row in map(
            json.loads, (legacy_dir / "samples_off.jsonl").read_text().splitlines()
        )
    }
    legacy_requests = list(provider.requests)
    provider.requests.clear()
    value = document(root=tmp_path / "prepared", provider=provider)
    store, engine, run = run_document(root=tmp_path / "store", value=value)
    assert store.get(run["id"])["status"] == "completed"
    assert [r["messages"] for r in provider.requests] == [
        r["messages"] for r in legacy_requests
    ]
    assert all(request["stream"] is True for request in provider.requests)
    rows = store.results(run["id"])
    for row in rows:
        original = legacy[row["case_id"].split("/", 1)[1]]["final"]
        assert row["correct"] is None
        assert row["score"] == original["normalized"]
        rubric = row["details"]["rubric"]
        for key in ("total", "negative_penalty", "n_negative_triggered", "per_section"):
            assert rubric[key] == original[key]
        assert len(rubric["verdicts"]) in {2, 3}
    report = make_report(store, run["id"])
    target = report["summary"]["targets"][0]
    for key in (
        "accuracy",
        "accuracy_ci95",
        "correct",
        "macro_accuracy",
        "sr_bench_score",
    ):
        assert target[key] is None
    quality = target["weighted_rubric"]
    assert quality["planned"] == quality["graded"] == 2
    assert quality["mean_normalized"] == legacy_summary["level2"]["mean_normalized"]
    assert quality["mean_total"] == legacy_summary["level2"]["mean_total"]
    assert quality["per_domain"]["Medicine"]["mean_normalized"] == 10 / 15
    calls = store.calls(run["id"])
    assert [call["role"] for call in calls].count("subject") == 2
    assert [call["role"] for call in calls].count("judge") == 3
    assert target["request_count"] == 2
    assert target["evaluation_call_count"] == 3
    assert target["cost_usd"] > 0
    assert target["evaluation_cost_usd"] > target["cost_usd"]
    assert target["total_spend_usd"] == pytest.approx(
        target["cost_usd"] + target["evaluation_cost_usd"]
    )
    assert all("effective_body" in call["request"] for call in calls)
    assert all(call["status"] == "completed" for call in calls)
    assert engine.start(value, request_key="draco-once")["id"] == run["id"]
    assert len(provider.requests) == 5
    persisted = json.loads(
        (store.root / "runs" / run["id"] / "report.json").read_text()
    )
    assert persisted["summary"]["targets"][0]["weighted_rubric"] == quality


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unknown",
        "string_boolean",
        "malformed",
        "http",
        "judge_limit",
        "subject_limit",
    ],
)
def test_failed_grading_is_not_zero_or_a_success_only_mean(
    tmp_path: Path, provider: Provider, fault: str
) -> None:
    provider.fault = fault
    store, _, run = run_document(
        root=tmp_path / "store",
        value=document(root=tmp_path / "prepared", provider=provider),
    )
    assert store.get(run["id"])["status"] == "failed"
    rows = store.results(run["id"])
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0].get("score") is None
    quality = make_report(store, run["id"])["summary"]["targets"][0]["weighted_rubric"]
    assert (
        quality["planned"],
        quality["graded"],
        quality["failed"],
        quality["pending"],
    ) == (2, 0, 1, 1)
    assert quality["mean_normalized"] is None
    assert quality["mean_negative_penalty"] is None
    assert len(provider.requests) == (1 if fault == "subject_limit" else 2)


def test_valid_penalty_zero_is_a_completed_grade(
    tmp_path: Path, provider: Provider
) -> None:
    provider.matched = {"wait-home", "cite-reddit"}
    store, _, run = run_document(
        root=tmp_path / "store",
        value=document(root=tmp_path / "prepared", provider=provider),
    )
    quality = make_report(store, run["id"])["summary"]["targets"][0]["weighted_rubric"]
    assert quality["complete"] is True
    assert quality["mean_normalized"] == 0
    assert quality["mean_negative_penalty"] == -55
    assert all(
        row["score"] == 0 and row["status"] == "completed"
        for row in store.results(run["id"])
    )


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_invalid_rubric_fails_before_any_dispatch(
    tmp_path: Path, provider: Provider, weight: float | bool
) -> None:
    value = plan(document(root=tmp_path / "prepared", provider=provider))
    value.pop("dataset")
    value.pop("plan_sha256")
    for section in value["cases"][0]["answer"]["sections"]:
        for criterion in section["criteria"]:
            criterion["weight"] = weight
    with pytest.raises(ValueError):
        Engine(Store(tmp_path / "store")).start(value)
    assert provider.requests == []


@pytest.mark.parametrize(
    "field", ["version", "kind", "budget", "prompt", "judge", "metadata"]
)
def test_frozen_draco_preflight_rejects_changed_contract(
    tmp_path: Path, provider: Provider, field: str
) -> None:
    value = plan(document(root=tmp_path / "prepared", provider=provider))
    value.pop("dataset")
    if field == "version":
        value["benchmark_options"][draco.BENCHMARK_ID]["grader_version"] = "other"
    elif field == "kind":
        value["result_kinds"][draco.BENCHMARK_ID] = "binary"
    elif field == "budget":
        value["limits"]["max_calls_per_case"] = 2
    elif field == "prompt":
        value["cases"][0]["messages"][0]["content"] = "different"
    elif field == "judge":
        value["benchmark_options"][draco.BENCHMARK_ID]["judge"] = "missing"
    else:
        value["cases"][0]["metadata"] = []
    with pytest.raises(ValueError):
        Engine(Store(tmp_path / "store")).start(value)
    assert provider.requests == []


def test_mixed_run_has_separate_quality_and_no_combined_score(
    tmp_path: Path, provider: Provider
) -> None:
    value = plan(document(root=tmp_path / "prepared", provider=provider))
    value.pop("dataset")
    value.pop("plan_sha256")
    value["cases"].append(
        {
            "id": "binary",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": "Return A"}],
            "answer": "A",
            "metadata": {"split": "dev"},
        }
    )
    value.pop("adapter_versions")
    store, _, run = run_document(root=tmp_path / "store", value=value)
    report = make_report(store, run["id"])
    target = report["summary"]["targets"][0]
    assert target["result_kind"] == "mixed"
    assert target["complete"] is True
    assert target["scored"] == target["total"] == 3
    assert (
        target["accuracy"]
        is target["macro_accuracy"]
        is target["sr_bench_score"]
        is None
    )
    binary = next(row for row in report["benchmarks"] if row["benchmark"] == "mmlu-pro")
    assert binary["accuracy"] == 1
    assert binary["accuracy_ci95"] is not None
    weighted = next(
        row for row in report["benchmarks"] if row["benchmark"] == draco.BENCHMARK_ID
    )
    assert weighted["accuracy_ci95"] is None
    assert weighted["weighted_rubric"]["graded"] == 2
    for operation in (
        lambda: export_training(store, run["id"]),
        lambda: regrade(store, run["id"]),
        lambda: compare(store, run["id"], run["id"]),
        lambda: replay(store, run["id"], run["id"]),
    ):
        with pytest.raises(ValueError, match="binary results only"):
            operation()


def test_local_preparation_preserves_selection_identity_and_hides_rubrics(
    tmp_path: Path,
) -> None:
    kwargs = {
        "source_path": FIXTURE,
        "revision": "fixture-v1",
        "source_partition": "fixture-test",
        "store": tmp_path,
        "domains": ["medicine"],
        "seed": 7,
        "max_samples": 1,
    }
    prepared = draco.prepare(**kwargs)
    assert draco.prepare(**kwargs) == prepared
    rows = [
        json.loads(line) for line in Path(prepared["path"]).read_text().splitlines()
    ]
    original = load_draco(
        path=str(FIXTURE), domains=["medicine"], seed=7, max_samples=1
    )
    assert [row["metadata"]["task_identity"]["task_id"] for row in rows] == [
        row.id for row in original
    ]
    assert rows[0]["metadata"]["task_identity"]["partition"] == "fixture-test"
    assert prepared["selection"] == draco.SELECTION_VERSION
    assert prepared["custom_subset"] is True
    assert (
        draco.prepare(**{**kwargs, "source_partition": "other"})["id"] != prepared["id"]
    )
    reader = DatasetReader(tmp_path)
    entry = next(
        row
        for row in reader.selection("smoke")["benchmarks"]
        if row["id"] == draco.BENCHMARK_ID
    )
    assert entry["reason_code"] == "local_preparation_required"
    public = reader.page(prepared["id"])
    assert "call-emergency" not in json.dumps(public)
    with pytest.raises(ValueError, match="smoke/quick"):
        draco.prepare(**{**kwargs, "profile": "standard"})


def test_shared_kernel_is_importable_in_installed_and_copied_cli(
    tmp_path: Path,
) -> None:
    installed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cli.sr_bench.adapters import get_adapter; print(get_adapter('draco-routed-answer-v1').result_kind)",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert installed.stdout.strip() == WEIGHTED_RUBRIC_RESULT
    copied = tmp_path / "copied"
    shutil.copytree(
        ROOT / "src/vllm-sr/cli",
        copied / "cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    copied_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cli.sr_bench.adapters import get_adapter; print(get_adapter('draco-routed-answer-v1').result_kind)",
        ],
        cwd=copied,
        env={**os.environ, "PYTHONPATH": str(copied)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert copied_result.stdout.strip() == WEIGHTED_RUBRIC_RESULT


def test_one_completed_grade_does_not_hide_a_later_failure(
    tmp_path: Path, provider: Provider
) -> None:
    provider.fault = "missing"
    provider.fault_after_subjects = 2
    store, _, run = run_document(
        root=tmp_path / "store",
        value=document(root=tmp_path / "prepared", provider=provider),
    )
    quality = make_report(store, run["id"])["summary"]["targets"][0]["weighted_rubric"]
    assert (quality["graded"], quality["failed"], quality["pending"]) == (1, 1, 0)
    assert quality["mean_normalized"] is None
    assert quality["per_domain"]["Finance"]["mean_normalized"] == 1
    assert quality["per_domain"]["Medicine"]["mean_normalized"] is None


def test_cli_and_shared_api_run_the_prepared_diagnostic(
    tmp_path: Path, provider: Provider
) -> None:
    value = document(root=tmp_path, provider=provider)
    source = tmp_path / "run.json"
    source.write_text(json.dumps(value))
    service = Server(("127.0.0.1", 0), Store(tmp_path))
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli.main",
                "benchmark",
                "--url",
                url,
                "--store",
                str(tmp_path),
                "--no-autostart",
                "run",
                "--manifest",
                str(source),
                "--idempotency-key",
                "cli-draco",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        assert "completed" in result.stdout
        response = requests.get(url=url + PREFIX + "/runs", timeout=5)
        response.raise_for_status()
        run = response.json()["runs"][0]
        response = requests.get(
            url=url + PREFIX + "/runs/" + run["id"] + "/report", timeout=5
        )
        response.raise_for_status()
        report = response.json()
        assert report["summary"]["targets"][0]["weighted_rubric"]["graded"] == 2
        assert report["provenance"]["result_kinds"] == {
            draco.BENCHMARK_ID: WEIGHTED_RUBRIC_RESULT
        }
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)


def test_preparation_cli_runs_outside_the_checkout(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.sr_bench.draco",
            "--source-path",
            str(FIXTURE),
            "--revision",
            "fixture-v1",
            "--source-partition",
            "fixture-test",
            "--store",
            str(tmp_path),
            "--domains",
            "Medicine",
            "--max-samples",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    prepared = json.loads(result.stdout)
    assert prepared["case_count"] == 1
    assert prepared["selection"] == draco.SELECTION_VERSION


@pytest.mark.parametrize("score", [0.5, None, True, float("nan"), float("inf")])
def test_weighted_result_contract_rejects_inconsistent_grades(score: object) -> None:
    result = {
        "correct": None,
        "score": score,
        "details": {
            "rubric": {
                "total": 1,
                "max_positive": 1,
                "positive_earned": 1,
                "negative_penalty": 0,
                "n_negative_triggered": 0,
                "per_section": {"reward": 1},
                "verdicts": [{"id": "reward", "weight": 1, "matched": True}],
            }
        },
    }
    with pytest.raises(ValueError):
        validate_result(kind=WEIGHTED_RUBRIC_RESULT, result=result)


def test_binary_result_contract_still_rejects_fractional_scores() -> None:
    with pytest.raises(ValueError, match="binary"):
        validate_result(kind="binary", result={"correct": True, "score": 0.5})


def test_composition_preserves_source_identity_without_claiming_a_new_shuffle(
    tmp_path: Path, provider: Provider
) -> None:
    prepared = document(root=tmp_path, provider=provider)["dataset"]
    binary = _write_dataset(
        root=tmp_path,
        cases=[
            {
                "id": "binary",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Return A"}],
                "answer": "A",
                "metadata": {"split": "dev"},
            }
        ],
        profile="quick",
        seed=42,
        sources={"mmlu-pro": {"revision": "fixture-v1"}},
        custom=True,
    )
    combined = combine_datasets(manifests=[prepared, binary], store=tmp_path)
    reader = DatasetReader(tmp_path)
    composed = reader.compose(
        dataset_ids=[prepared["id"], binary["id"]],
        benchmarks=[draco.BENCHMARK_ID, "mmlu-pro"],
    )
    for result in (combined, composed):
        assert result["selection"] == "frozen-source-composition-v1"
        assert result["case_count"] == 3
        assert (
            result["sources"][draco.BENCHMARK_ID]
            == prepared["sources"][draco.BENCHMARK_ID]
        )


def test_changed_prepared_rubric_cannot_dispatch(
    tmp_path: Path, provider: Provider
) -> None:
    value = document(root=tmp_path, provider=provider)
    path = Path(value["dataset"]["path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match=r"identity|digest|SHA256"):
        Engine(Store(tmp_path / "store")).start(value)
    assert provider.requests == []


def test_binary_plan_identity_does_not_gain_a_result_kind_field(
    tmp_path: Path, provider: Provider
) -> None:
    value = document(root=tmp_path, provider=provider)
    value.pop("dataset")
    value["benchmark_options"] = {}
    value["cases"] = [
        {
            "id": "binary",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": "Return A"}],
            "answer": "A",
        }
    ]
    original = plan(value)
    value["result_kinds"] = {}
    explicit = plan(value)
    assert "result_kinds" not in explicit
    assert explicit["plan_sha256"] == original["plan_sha256"]


def test_retained_item_export_uses_the_shared_kernel(tmp_path: Path) -> None:
    destination = tmp_path / "items.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.grounded_fusion.items",
            "--draco-path",
            str(FIXTURE),
            "--out",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert {row["id"] for row in rows} == {"fixture-med-1", "fixture-fin-1"}


def test_recovery_keeps_weighted_identity_and_does_not_complete_the_parent(
    tmp_path: Path, provider: Provider
) -> None:
    provider.fault = "missing"
    store, engine, parent = run_document(
        root=tmp_path / "store",
        value=document(root=tmp_path / "prepared", provider=provider),
    )
    proposed = recovery_plan(store=store, parent_id=parent["id"])
    assert len(proposed["eligible_cells"]) == 1
    provider.fault = None
    child = recover(
        engine=engine,
        parent_id=parent["id"],
        body={
            "idempotency_key": "draco-recovery",
            "mode": "undispatched",
            "cells": proposed["eligible_cells"],
            "plan_sha256": proposed["plan_sha256"],
        },
    )
    engine.threads[child["id"]].join(timeout=10)
    assert wait_run(store, child["id"])["status"] == "completed"
    assert child["manifest"]["result_kinds"] == parent["manifest"]["result_kinds"]
    recovered = make_report(store, child["id"])["summary"]["targets"][0]
    assert recovered["weighted_rubric"]["planned"] == 1
    assert recovered["weighted_rubric"]["mean_normalized"] == 10 / 15
    original = make_report(store, parent["id"])["summary"]["targets"][0]
    assert original["weighted_rubric"]["planned"] == 2
    assert original["weighted_rubric"]["mean_normalized"] is None
