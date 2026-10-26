from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.legacy_reasoning import CONFIGURATIONS, inspect_runtime
from cli.sr_bench.legacy_reasoning_export import export_manifest as export_plain
from cli.sr_bench.legacy_reasoning_modes import (
    FIELD,
    export_manifest,
    selection_identity,
)
from cli.sr_bench.report import make_report
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from sr_bench_legacy_reasoning_native_checks import (
    native_python,
    source_bundle,
    wait_for_run,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SR_BENCH_NATIVE_REASONING_TESTS") != "1",
    reason="Run make test-sr-bench-legacy-reasoning for the pinned native runtime",
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    inspect_runtime()
    directory = tmp_path_factory.mktemp("mode-sources")
    path = source_bundle(directory=directory)
    data = json.loads(path.read_text())
    rows = [
        {
            "question_id": index,
            "question": f"MMLU question {index}?",
            "options": ["first", "second", "N/A"],
            "answer": "A",
            "category": "science",
            "cot_content": "REFERENCE-EXPLANATION" if index % 2 == 0 else None,
        }
        for index in range(12)
    ]
    (directory / "mmlu.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    data["sources"].append(
        {
            "dataset": "TIGER-Lab/MMLU-Pro",
            "configuration": None,
            "split": "test",
            "revision": "fixture-v1",
            "path": "mmlu.jsonl",
        }
    )
    path.write_text(json.dumps(data))
    return path


class ConditionTarget(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        self.server.request_headers.append(
            {key.lower(): value for key, value in self.headers.items()}
        )
        if self.server.fail:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "text/event-stream" if body.get("stream") else "application/json",
        )
        if self.headers.get("X-SR-Bench-Expected-Config-Hash"):
            self.send_header(
                "X-SR-Bench-Config-Hash",
                self.headers["X-SR-Bench-Expected-Config-Hash"],
            )
        if self.server.child_usage:
            self.send_header("x-vsr-model-usage", json.dumps(self.server.child_usage))
        self.end_headers()
        finish = "length" if self.server.truncated else "stop"
        usage = self.server.usage
        if not body.get("stream"):
            response = {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {
                            "role": "assistant",
                            "content": self.server.answer,
                            "reasoning_content": "Answer: B",
                        },
                    }
                ],
                "usage": usage,
            }
            self.wfile.write(json.dumps(response).encode())
            return
        events = [
            {
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Answer: B"},
                    }
                ],
            },
            {
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": self.server.answer},
                        "finish_reason": finish,
                    }
                ],
            },
        ]
        if usage is not None:
            events.append({"model": body["model"], "choices": [], "usage": usage})
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        if not self.server.incomplete_stream:
            self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def target() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ConditionTarget)
    server.requests = []
    server.request_headers = []
    server.answer = "Answer: A"
    server.fail = False
    server.truncated = False
    server.incomplete_stream = False
    server.child_usage = None
    server.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 2, "cache_creation_tokens": 1},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def template(
    target: Any, modes: tuple[str, ...] = ("NR", "NR_REASONING", "XC")
) -> dict[str, Any]:
    profiles = []
    for index, mode in enumerate(modes):
        profile = {
            "id": f"condition-{index}",
            "kind": "mom" if mode == "Router_NR" else "single",
            "model": "Qwen3-fixture",
            "base_url": f"http://127.0.0.1:{target.server_port}/v1",
        }
        if mode == "Router_NR":
            profile["config_hash"] = "fixture-config"
        if mode == "AR":
            profile["request_params"] = {"reasoning_effort": "medium"}
        profiles.append(profile)
    return {
        "version": "sr-bench-1.0",
        "profile": "smoke",
        "cost_policy": "capability_only",
        "targets": profiles,
        "limits": {
            "concurrency": 1,
            "total_timeout_s": 15,
            "idle_timeout_s": 5,
            "max_run_seconds": 180,
            "case_timeout_s": 30,
        },
        FIELD: {
            "version": 1,
            "conditions": [
                {"target_id": profile["id"], "mode": mode}
                for profile, mode in zip(profiles, modes, strict=True)
            ],
        },
    }


@pytest.mark.parametrize("dataset", [*CONFIGURATIONS, "mmlu-pro"])
def test_one_native_selection_preserves_plain_cases_and_formats_xc(
    dataset: str, bundle: Path, target: Any
) -> None:
    compiled = export_manifest(
        manifest=template(target),
        dataset=dataset,
        source_bundle=bundle,
        samples_per_category=1,
    )
    contract = compiled[FIELD]
    if dataset != "mmlu-pro":
        plain = export_plain(
            manifest={},
            dataset=dataset,
            source_bundle=bundle,
            samples_per_category=1,
        )
        assert [
            case
            for case in compiled["cases"]
            if not case["benchmark"].endswith("-explicit-cot")
        ] == plain["cases"]
    xc = contract["conditions"][-1]
    has_cot = any(member["has_cot"] for member in contract["members"][:10])
    assert xc["status"] == ("planned" if has_cot else "skipped")
    assert not target.requests
    if has_cot:
        cases = [
            case
            for case in compiled["cases"]
            if case["benchmark"].endswith("-explicit-cot")
        ]
        observed = native_python(
            source="""
import json,sys
from reasoning.dataset_factory import DatasetFactory
from reasoning.dataset_interface import Question
request=json.load(sys.stdin)
dataset=DatasetFactory.create_dataset(request["dataset"])
print(json.dumps([dataset.format_prompt(Question(**row), "explicit_cot") for row in request["questions"]]))
""",
            payload={
                "dataset": dataset,
                "questions": [case["answer"]["question"] for case in cases],
            },
        )
        assert [case["messages"][0]["content"] for case in cases] == observed
        assert {case["answer"]["plain_case"]["id"] for case in cases} == {
            member["plain_id"] for member in contract["members"]
        }
    assert all(
        "REFERENCE-" not in json.dumps(case["messages"])
        for case in compiled["cases"]
        if not case["benchmark"].endswith("-explicit-cot")
    )


def test_native_aliases_and_control_names(bundle: Path, target: Any) -> None:
    for alias, name in (("mmlu", "mmlu-pro"), ("gpqa", "gpqa-main")):
        arguments = {
            "manifest": template(target),
            "source_bundle": bundle,
            "samples_per_category": 1,
        }
        assert export_manifest(dataset=alias, **arguments) == export_manifest(
            dataset=name, **arguments
        )
    for model, parameter, off, on in (
        ("DeepSeek-V3.1", "thinking", False, True),
        ("Qwen3-fixture", "enable_thinking", False, True),
        ("gpt-oss-fixture", "reasoning_effort", "low", "high"),
    ):
        source = template(target)
        for profile in source["targets"]:
            profile["model"] = model
        compiled = export_manifest(
            manifest=source,
            dataset="mmlu",
            source_bundle=bundle,
            samples_per_category=1,
        )
        assert (
            compiled["targets"][0]["request_params"]["chat_template_kwargs"][parameter]
            == off
        )
        assert (
            compiled["targets"][1]["request_params"]["chat_template_kwargs"][parameter]
            == on
        )
        assert compiled[FIELD]["conditions"][2]["controls"] == {}


def test_native_first_ten_rule_does_not_filter_individual_questions(
    bundle: Path, target: Any, tmp_path: Path
) -> None:
    data = json.loads(bundle.read_text())
    entry = next(
        source
        for source in data["sources"]
        if source["dataset"] == "TIGER-Lab/MMLU-Pro"
    )
    rows = [
        json.loads(line)
        for line in (bundle.parent / entry["path"]).read_text().splitlines()
    ]
    for index, row in enumerate(rows):
        row["cot_content"] = "REFERENCE-ONLY-ELEVENTH" if index == 10 else "   "
    source = tmp_path / "mmlu.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    local = tmp_path / "bundle.json"
    local.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "fixture",
                "sources": [{**entry, "path": source.name}],
            }
        )
    )
    skipped = export_manifest(
        manifest=template(target), dataset="mmlu", source_bundle=local
    )
    assert skipped[FIELD]["conditions"][-1]["reason"] == "native_first_ten_have_no_cot"
    assert len(skipped["execution_cells"]) == 24
    with pytest.raises(ValueError, match="No executable"):
        export_manifest(
            manifest=template(target, ("XC",)), dataset="mmlu", source_bundle=local
        )
    rows[0]["cot_content"] = "REFERENCE-FIRST"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    enabled = export_manifest(
        manifest=template(target), dataset="mmlu", source_bundle=local
    )
    assert len(enabled["execution_cells"]) == 36
    assert enabled[FIELD]["selection_sha256"] != skipped[FIELD]["selection_sha256"]


@pytest.mark.parametrize("model", ["unknown-fixture", "gpt-4"])
def test_unrepresentable_native_controls_fail_without_http(
    model: str, bundle: Path, target: Any
) -> None:
    source = template(target)
    for profile in source["targets"]:
        profile["model"] = model
    with pytest.raises(ValueError, match=r"(Unknown model|Unsupported reasoning)"):
        export_manifest(
            manifest=source,
            dataset="mmlu",
            source_bundle=bundle,
            samples_per_category=1,
        )
    assert not target.requests


def _run(document: dict[str, Any], store: Store) -> dict[str, Any]:
    engine = Engine(store)
    try:
        run = engine.start(document, request_key="once")
        wait_for_run(store=store, run_id=run["id"])
        assert engine.start(document, request_key="once")["id"] == run["id"]
        return store.get(run["id"])
    finally:
        engine.close()


def test_modes_use_real_native_and_streaming_paths(
    bundle: Path, target: Any, tmp_path: Path
) -> None:
    compiled = export_manifest(
        manifest=template(target, ("standard", "reasoning", "AR", "XC", "Router_NR")),
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=1,
    )
    store = Store(tmp_path)
    run = _run(compiled, store)
    assert run["status"] == "completed", store.results(run["id"])
    assert len(target.requests) == 5
    assert all(row["correct"] for row in store.results(run["id"]))
    calls = store.calls(run["id"])
    assert [call["request"]["effective_body"] for call in calls] == target.requests
    assert target.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert target.requests[1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert target.requests[2]["reasoning_effort"] == "medium"
    assert all(
        not ({"reasoning_effort", "chat_template_kwargs"} & request.keys())
        for request in target.requests[3:]
    )
    assert (
        target.requests[0]["messages"]
        == target.requests[1]["messages"]
        == target.requests[2]["messages"]
    )
    report = make_report(store, run["id"])[FIELD]
    for condition in report["conditions"]:
        metric = condition["summary"]
        assert metric["accuracy"] == 1
        assert metric["completion_prompt_ratio"] == pytest.approx(0.3)
        assert metric["reported_reasoning_tokens"] == 1
        assert metric["time_per_completion_token_ms"] > 0
        assert len(condition["categories"]) == 1
    assert len(report["plain_quality_deltas"]) == 2
    assert all(row["accuracy_delta"] == 0 for row in report["plain_quality_deltas"])
    assert make_report(Store(tmp_path), run["id"])[FIELD] == report
    before = len(target.requests)
    native = native_python(
        source="""
import json,sys
from openai import OpenAI
from reasoning.dataset_factory import DatasetFactory
from reasoning.dataset_interface import Question
from reasoning.router_reason_bench_multi_dataset import process_question_single
request=json.load(sys.stdin)
question=Question(**request["question"])
client=OpenAI(base_url=request["url"],api_key="fixture",max_retries=0)
rows=[process_question_single(client=client,model="Qwen3-fixture",question=question,
dataset=DatasetFactory.create_dataset("mmlu"),prompt_mode=mode,max_tokens=4096,
temperature=0,ar_extra_body={"chat_template_kwargs":{"enable_thinking":True}})
for mode in ("NR","AR","XC")]
print(json.dumps([row["is_correct"] for row in rows]))
""",
        payload={
            "url": f"http://127.0.0.1:{target.server_port}/v1",
            "question": next(
                case
                for case in compiled["cases"]
                if case["benchmark"].endswith("-explicit-cot")
            )["answer"]["question"],
        },
    )
    assert native == [True, True, True]
    assert len(target.requests) == before + 3
    assert "chat_template_kwargs" not in target.requests[-1]
    assert target.requests[-1]["messages"] == next(
        call["request"]["effective_body"]["messages"]
        for call in calls
        if call["target_id"] == "condition-3"
    )


@pytest.mark.parametrize(
    ("fault", "expected_accuracy"),
    [
        ("reasoning-only", 0),
        ("ambiguous", 0),
        ("limit", 0),
        ("missing-usage", 1),
        ("zero-output", 1),
        ("http", 0),
        ("stream", 0),
    ],
)
def test_incomplete_output_and_usage_remain_explicit(
    fault: str, expected_accuracy: int, bundle: Path, target: Any, tmp_path: Path
) -> None:
    if fault == "reasoning-only":
        target.answer = ""
    elif fault == "ambiguous":
        target.answer = "Answer: A or B"
    elif fault == "limit":
        target.truncated = True
    elif fault == "missing-usage":
        target.usage = None
    elif fault == "zero-output":
        target.usage["completion_tokens"] = 0
    elif fault == "http":
        target.fail = True
    elif fault == "stream":
        target.incomplete_stream = True
    compiled = export_manifest(
        manifest=template(target, ("NR", "NR_REASONING")),
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=2,
    )
    store = Store(tmp_path)
    run = _run(compiled, store)
    report = make_report(store, run["id"])[FIELD]
    assert all(row["summary"]["total"] == 2 for row in report["conditions"])
    assert report["conditions"][0]["summary"]["accuracy"] == expected_accuracy
    assert len(target.requests) == (1 if fault in {"http", "stream"} else 4)
    if fault in {"limit", "missing-usage", "zero-output", "http", "stream"}:
        assert (
            report["conditions"][0]["summary"]["time_per_completion_token_ms"] is None
        )
    assert all(
        call["reasoning"] == "Answer: B"
        for call in store.calls(run["id"])
        if call.get("reasoning")
    )


@pytest.mark.parametrize(
    "change", ["profile", "cell", "prompt", "member", "skip", "source", "selection"]
)
def test_frozen_condition_changes_fail_before_dispatch(
    change: str, bundle: Path, target: Any
) -> None:
    compiled = export_manifest(
        manifest=template(target),
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=1,
    )
    altered = copy.deepcopy(compiled)
    if change == "profile":
        altered["targets"][0]["request_params"] = {"reasoning_effort": "high"}
    elif change == "cell":
        altered["execution_cells"][0]["target_id"] = "condition-1"
    elif change == "prompt":
        altered["cases"][0]["messages"][0]["content"] = "Changed"
    elif change == "member":
        altered[FIELD]["members"][0]["native_sha256"] = "f" * 64
    elif change == "skip":
        altered[FIELD]["conditions"][-1]["reason"] = "different"
    elif change == "source":
        altered[FIELD]["source"]["files"][0]["sha256"] = "f" * 64
        altered[FIELD]["selection_sha256"] = selection_identity(altered[FIELD])
    else:
        altered[FIELD]["selection"]["categories"] = ["different"]
        altered[FIELD]["selection_sha256"] = selection_identity(altered[FIELD])
    with pytest.raises(ValueError):
        plan(altered)
    assert not target.requests


def test_native_grader_failure_keeps_unattempted_conditions_in_denominators(
    bundle: Path, target: Any, tmp_path: Path
) -> None:
    target.answer = "Answer: AB"
    compiled = export_manifest(
        manifest=template(target),
        dataset="commonsenseqa",
        source_bundle=bundle,
        samples_per_category=2,
    )
    store = Store(tmp_path)
    run = _run(compiled, store)
    assert run["status"] == "failed"
    assert len(target.requests) == 1
    conditions = make_report(store, run["id"])[FIELD]["conditions"]
    assert [row["summary"]["total"] for row in conditions] == [2, 2, 0]
    assert [row["summary"]["accuracy"] for row in conditions] == [0, 0, None]
    assert conditions[-1]["skip_reason"] == "native_first_ten_have_no_cot"
    assert all(row["summary"]["completion_prompt_ratio"] is None for row in conditions)
    logs = list((tmp_path / "runs" / run["id"]).glob("*/harness.log"))
    assert len(logs) == 1 and "TypeError" in logs[0].read_text()


def test_skipped_targets_still_require_valid_frozen_profiles(
    bundle: Path, target: Any
) -> None:
    compiled = export_manifest(
        manifest=template(target),
        dataset="arc",
        source_bundle=bundle,
        samples_per_category=1,
    )
    assert compiled[FIELD]["conditions"][-1]["status"] == "skipped"
    compiled[FIELD]["conditions"][-1]["target"]["api_key"] = "forbidden-fixture-key"
    with pytest.raises(ValueError, match="raw credentials"):
        plan(compiled)
    assert not target.requests


def test_validation_uses_actual_frozen_template_kwargs(
    bundle: Path, target: Any
) -> None:
    source = template(target)
    source["sampling"] = {"chat_template_kwargs": {"date_string": "fixture-date"}}
    compiled = export_manifest(
        manifest=source,
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=1,
    )
    altered = copy.deepcopy(compiled)
    for profile in (
        altered["targets"][0],
        altered[FIELD]["conditions"][0]["target"],
    ):
        profile["request_params"]["chat_template_kwargs"].pop("date_string", None)
    with pytest.raises(ValueError, match="non-reasoning"):
        plan(altered)
    assert not target.requests


@pytest.mark.parametrize("session_mode", [None, "session_aware"])
def test_condition_and_category_reports_preserve_the_wire_session_mode(
    session_mode: str | None, bundle: Path, target: Any, tmp_path: Path
) -> None:
    source = template(target, ("NR",))
    if session_mode is not None:
        source["targets"][0]["session_mode"] = session_mode
    compiled = export_manifest(
        manifest=source,
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=2,
    )
    store = Store(tmp_path)
    run = _run(compiled, store)
    assert run["status"] == "completed"
    calls = store.calls(run["id"])
    assert len(calls) == len(target.request_headers) == 2
    for call, headers in zip(calls, target.request_headers, strict=True):
        if session_mode == "session_aware":
            assert call["session_id"]
            assert headers["x-session-id"] == call["session_id"]
        else:
            assert "session_id" not in call
            assert "x-session-id" not in headers
    report = make_report(store, run["id"])
    condition = report[FIELD]["conditions"][0]
    for metric in (
        report["summary"]["targets"][0],
        condition["summary"],
        condition["categories"][0],
    ):
        assert metric["continuity"]["session_mode"] == (session_mode or "stateless")


def test_endpoint_token_ratio_is_not_router_child_usage(
    bundle: Path, target: Any, tmp_path: Path
) -> None:
    target.child_usage = {
        "version": 1,
        "complete": True,
        "calls": [
            {"model": model, "usage": {"prompt_tokens": 20, "completion_tokens": 3}}
            for model in ("first-model", "second-model")
        ],
    }
    compiled = export_manifest(
        manifest=template(target, ("Router_NR",)),
        dataset="mmlu",
        source_bundle=bundle,
        samples_per_category=1,
    )
    store = Store(tmp_path)
    run = _run(compiled, store)
    report = make_report(store, run["id"])
    assert report["summary"]["targets"][0]["tokens"]["input_tokens"] == 40
    row = report[FIELD]["conditions"][0]["summary"]
    assert row["avg_prompt_tokens"] == 10
    assert row["avg_completion_tokens"] == 3
    assert row["completion_prompt_ratio"] == pytest.approx(0.3)
    assert len(target.requests) == 1


def test_real_cli_exports_runs_reports_and_reopens_store(
    bundle: Path, target: Any, tmp_path: Path
) -> None:
    source = tmp_path / "template.json"
    source.write_text(json.dumps(template(target, ("NR", "NR_REASONING", "XC"))))
    frozen = tmp_path / "conditions.json"
    export_command = [
        sys.executable,
        "-m",
        "cli.sr_bench.legacy_reasoning_modes",
        "--dataset",
        "mmlu",
        "--source-bundle",
        str(bundle),
        "--manifest",
        str(source),
        "--output",
        str(frozen),
        "--samples-per-category",
        "1",
    ]
    exported = subprocess.run(
        export_command, capture_output=True, text=True, check=True
    )
    assert json.loads(exported.stdout)["model_requests"] == 0
    assert frozen.stat().st_mode & 0o777 == 0o600
    before = frozen.read_bytes()
    repeated = subprocess.run(
        export_command, capture_output=True, text=True, check=False
    )
    assert repeated.returncode != 0 and frozen.read_bytes() == before
    store = Store(tmp_path / "store")
    server = Server(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [
        sys.executable,
        "-m",
        "cli.main",
        "benchmark",
        "--url",
        f"http://127.0.0.1:{server.server_port}",
        "--store",
        str(store.root),
        "--no-autostart",
    ]
    try:
        experiment = json.loads(
            subprocess.run(
                [*command, "experiment", "create", "Reasoning conditions"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        document = json.loads(frozen.read_text())
        document["experiment"] = {"id": experiment["id"], "role": "baseline"}
        frozen.write_text(json.dumps(document))
        reviewed = tmp_path / "reviewed.json"
        planned = subprocess.run(
            [
                *command,
                "plan",
                "--manifest",
                str(frozen),
                "--output",
                str(reviewed),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(planned.stdout)["total"] == 3
        run_command = [
            *command,
            "run",
            "--manifest",
            str(reviewed),
            "--idempotency-key",
            "mode-once",
        ]
        result = json.loads(
            subprocess.run(
                run_command, capture_output=True, text=True, check=True
            ).stdout
        )
        run_id = result["run_id"]
        assert len(target.requests) == 3
        subprocess.run(run_command, capture_output=True, text=True, check=True)
        report = json.loads(
            subprocess.run(
                [*command, "report", run_id],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        assert report[FIELD] == result[FIELD]
        summary = json.loads(
            subprocess.run(
                [*command, "runs"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        assert FIELD in json.dumps(summary)
        assert len(target.requests) == 3
        assert store.get(run_id)["manifest"]["experiment"]["id"] == experiment["id"]
    finally:
        server.shutdown()
        server.server_close()
        server.engine.close()
        thread.join(timeout=2)
    assert make_report(Store(store.root), run_id)[FIELD] == report[FIELD]
    print(
        json.dumps(
            {
                "cli_http_store": {
                    "conditions": 3,
                    "requests": len(target.requests),
                    "persisted_selection": report[FIELD]["selection_sha256"],
                    "completion_prompt_ratios": [
                        row["summary"]["completion_prompt_ratio"]
                        for row in report[FIELD]["conditions"]
                    ],
                }
            }
        )
    )
