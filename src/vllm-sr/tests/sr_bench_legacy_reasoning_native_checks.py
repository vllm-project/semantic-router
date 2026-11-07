from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
import test_sr_bench
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.legacy_reasoning import (
    CONFIGURATIONS,
    ENVIRONMENT,
    PROTOCOL,
    REQUIREMENTS,
    inspect_runtime,
)
from cli.sr_bench.legacy_reasoning_export import export_manifest
from cli.sr_bench.report import make_report
from cli.sr_bench.setup import harness_paths
from cli.sr_bench.store import TERMINAL, Store
from test_sr_bench import manifest

target = test_sr_bench.target

pytestmark = pytest.mark.skipif(
    os.environ.get("SR_BENCH_NATIVE_REASONING_TESTS") != "1",
    reason="Run make test-sr-bench-legacy-reasoning for the pinned optional native environment",
)


def source_bundle(directory: Path) -> Path:
    sources: list[dict[str, Any]] = []

    def add(
        dataset: str,
        rows: list[dict[str, Any]],
        split: str = "test",
        configuration: str | None = None,
    ) -> None:
        path = directory / f"source-{len(sources)}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sources.append(
            {
                "dataset": dataset,
                "configuration": configuration,
                "split": split,
                "path": path.name,
                "revision": "synthetic-fixture-v1",
            }
        )

    for variant in ("ARC-Easy", "ARC-Challenge"):
        add(
            dataset="allenai/ai2_arc",
            configuration=variant,
            rows=[
                {
                    "id": f"{variant}-{index}",
                    "question": f"ARC science {index}?",
                    "choices": {
                        "label": ["B", "A", "C", "D"],
                        "text": ["stone", "water", "air", "fire"],
                    },
                    "answerKey": "B",
                }
                for index in range(4)
            ]
            + [
                {
                    "id": "native-skips-numeric-label",
                    "question": "Not retained?",
                    "choices": {"label": ["1", "2"], "text": ["one", "two"]},
                    "answerKey": "1",
                }
            ],
        )
    for subset in ("gpqa_main", "gpqa_extended", "gpqa_diamond"):
        add(
            dataset="Idavidrein/gpqa",
            configuration=subset,
            split="train",
            rows=[
                {
                    "Record ID": f"{subset}-{index}",
                    "Question": f"Graduate science {index}?",
                    "Correct Answer": f"correct-{index}",
                    "Incorrect Answer 1": "distractor-one",
                    "Incorrect Answer 2": "distractor-two",
                    "Incorrect Answer 3": "distractor-three",
                    "Subject": "Physics" if index % 2 else "Chemistry",
                    "Explanation": "PRIVATE-GPQA-REFERENCE",
                }
                for index in range(4)
            ],
        )
    add(
        dataset="gsm8k",
        configuration="main",
        rows=[
            {
                "question": f"How many items in sum {index}?",
                "answer": "PRIVATE-GSM-SOLUTION\n#### 1,234",
            }
            for index in range(4)
        ],
    )
    add(
        dataset="aqua_rat",
        rows=[
            {
                "question": f"Which algebraic value {index}?",
                "options": ["A) 10", "B) 20", "C) 30", "D) 40", "E) 50"],
                "correct": "C",
                "rationale": "PRIVATE-AQUA-RATIONALE",
            }
            for index in range(4)
        ],
    )
    add(
        dataset="ucinlp/drop",
        split="validation",
        rows=[
            {
                "passage": "The badge was red.",
                "question": f"What color was badge {index}?",
                "answers_spans": {
                    "spans": ["red", "crimson"],
                    "types": ["span", "span"],
                },
            }
            for index in range(4)
        ],
    )
    add(
        dataset="ChilleD/StrategyQA",
        rows=[
            {
                "question": f"Is statement {index} true?",
                "answer": index % 2 == 0,
                "decomposition": ["PRIVATE-STRATEGY-DECOMPOSITION"],
            }
            for index in range(4)
        ],
    )
    add(
        dataset="sciq",
        rows=[
            {
                "question": f"Which physical process {index}?",
                "correct_answer": "evaporation",
                "distractor1": "freezing",
                "distractor2": "sublimation",
                "distractor3": "condensation",
                "support": "Background supplied by the native source.",
            }
            for index in range(4)
        ],
    )
    add(
        dataset="openbookqa",
        rows=[
            {
                "question_stem": f"Which scientific fact {index}?",
                "choices": {
                    "text": ["gas", "liquid", "solid", "plasma"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "D",
            }
            for index in range(4)
        ],
    )
    add(
        dataset="truthful_qa",
        configuration="multiple_choice",
        split="validation",
        rows=[
            {
                "question": f"Which claim is true {index}?",
                "mc1_targets": {
                    "choices": ["false", "first true", "second true"],
                    "labels": [0, 1, 1],
                },
            }
            for index in range(4)
        ],
    )
    for split in ("train", "validation"):
        add(
            dataset="commonsense_qa",
            split=split,
            rows=[
                {
                    "id": f"commonsense-{split}-{index}",
                    "question": f"Common sense {split} {index}?",
                    "choices": {
                        "label": ["C", "A", "B"],
                        "text": ["first", "second", "third"],
                    },
                    "answerKey": "C",
                }
                for index in range(4)
            ],
        )
        add(
            dataset="hellaswag",
            split=split,
            rows=[
                {
                    "ind": index + (10 if split == "validation" else 0),
                    "ctx": f"A person walks {split} {index}.",
                    "endings": ["sits down", "flies away", "vanishes", "turns blue"],
                    "label": "0",
                    "activity_label": "tv_watching" if index % 2 else "diy",
                }
                for index in range(4)
            ],
        )
    path = directory / "bundle.json"
    path.write_text(json.dumps({"version": 1, "kind": "fixture", "sources": sources}))
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    inspect_runtime()
    return source_bundle(directory=tmp_path_factory.mktemp("native-sources"))


@pytest.fixture(scope="module")
def catalog_cases(bundle: Path) -> dict[str, dict[str, Any]]:
    return {
        name: export_manifest(
            manifest={},
            dataset=name,
            source_bundle=bundle,
            seed=42,
            samples_per_category=2,
        )
        for name in CONFIGURATIONS
    }


def native_python(source: str, payload: Any) -> Any:
    root, interpreter = harness_paths(ENVIRONMENT)
    completed = subprocess.run(
        [str(interpreter), "-c", source],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=root,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        timeout=120,
        check=True,
    )
    return json.loads(completed.stdout)


def test_native_factory_selection_and_prompt_parity(
    bundle: Path, catalog_cases: dict[str, dict[str, Any]]
) -> None:
    observed = native_python(
        source="""
import importlib, json, random, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from datasets import Dataset
from reasoning.dataset_factory import DatasetFactory

request = json.load(sys.stdin)
bundle = Path(request["bundle"])
sources = json.loads(bundle.read_text())["sources"]
def local_load(path, name=None, *, split):
    entry = next(item for item in sources if
                 (item["dataset"], item["configuration"], item["split"]) == (path, name, split))
    return Dataset.from_list([json.loads(line) for line in (bundle.parent / entry["path"]).read_text().splitlines()])

result = {}
for name in request["names"]:
    dataset = DatasetFactory.create_dataset(name=name)
    module = importlib.import_module(type(dataset).__module__)
    module.load_dataset = local_load
    random.seed(42)
    np.random.seed(42)
    questions, info = dataset.load_dataset(seed=42, samples_per_category=2)
    result[name] = [{
        "answer": asdict(question),
        "messages": [{"role": "user", "content": dataset.format_prompt(question, "plain")}],
    } for question in questions]
print(json.dumps({"names": DatasetFactory.get_available_datasets(), "cases": result}))
""",
        payload={"bundle": str(bundle), "names": list(CONFIGURATIONS)},
    )
    assert len(observed["names"]) == 18
    for name, document in catalog_cases.items():
        cases = document["cases"]
        assert [
            {"answer": case["answer"], "messages": case["messages"]} for case in cases
        ] == observed["cases"][name]
        config = document["benchmark_options"]["legacy-" + name]
        assert config["source"]["kind"] == "fixture"
        assert config["selection"]["method"] == "native-loader"
        assert (
            config["requirements_sha256"]
            == hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
        )
        assert all("split" not in case["metadata"] for case in cases)
        assert all(
            "PRIVATE-"
            not in json.dumps(
                {key: value for key, value in case.items() if key != "answer"}
            )
            for case in cases
        )
        for source in config["source"]["files"]:
            entry = next(
                item
                for item in json.loads(bundle.read_text())["sources"]
                if all(
                    item.get(key) == source[key]
                    for key in ("dataset", "configuration", "split")
                )
            )
            assert (
                source["sha256"]
                == hashlib.sha256(
                    (bundle.parent / entry["path"]).read_bytes()
                ).hexdigest()
            )
    assert all(
        "native-skips-numeric-label" not in case["metadata"]["native_question_id"]
        for case in catalog_cases["arc"]["cases"]
    )
    assert {
        source["split"]
        for source in catalog_cases["hellaswag"]["benchmark_options"][
            "legacy-hellaswag"
        ]["source"]["files"]
    } == {"train", "validation"}


def test_alias_repeat_and_source_changes_have_explicit_identity(
    bundle: Path, catalog_cases: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    alias = export_manifest(
        manifest={},
        dataset="gpqa",
        source_bundle=bundle,
        seed=42,
        samples_per_category=2,
    )
    assert alias == catalog_cases["gpqa-main"]
    changed_seed = export_manifest(
        manifest={},
        dataset="gpqa-main",
        source_bundle=bundle,
        seed=43,
        samples_per_category=2,
    )
    assert {case["id"] for case in alias["cases"]}.isdisjoint(
        case["id"] for case in changed_seed["cases"]
    )
    copied = source_bundle(directory=tmp_path)
    source = tmp_path / "source-0.jsonl"
    source.write_text(source.read_text() + "\n")
    changed_bytes = export_manifest(
        manifest={},
        dataset="arc",
        source_bundle=copied,
        seed=42,
        samples_per_category=2,
    )
    assert [case["answer"] for case in changed_bytes["cases"]] == [
        case["answer"] for case in catalog_cases["arc"]["cases"]
    ]
    assert {case["id"] for case in changed_bytes["cases"]}.isdisjoint(
        case["id"] for case in catalog_cases["arc"]["cases"]
    )
    selected = export_manifest(
        manifest={},
        dataset="gpqa",
        source_bundle=bundle,
        categories=["Physics"],
        seed=42,
    )
    assert {case["metadata"]["stratum"] for case in selected["cases"]} == {"Physics"}
    assert selected["benchmark_options"]["legacy-gpqa-main"]["selection"][
        "categories"
    ] == ["Physics"]


def test_native_correctness_helper_matches_the_legacy_runner() -> None:
    result = native_python(
        source="""
import json
from reasoning.dataset_interface import Question
from reasoning.reasoning_mode_eval import _is_correct_answer
from reasoning import router_reason_bench_multi_dataset as runner
from reasoning.dataset_factory import DatasetFactory

compared, failures = 0, []
shapes = [
    (["first", "second"], 0), (["first", "second"], "A"),
    (["Yes", "No"], "Yes"), (["No", "Yes"], "No"),
    ([], "1234"), ([], "red"), ([], "Centenary Medal"),
]
outputs = ["A", "Answer: A", "Answer: a", "Answer: AB", "A or B", "Answer: A or B",
           "Yes", "No", "No, but yes", "true", "incorrect", "",
           "ANSWER: 1,234", "1234 dollars", "ANSWER: crimson", "Answer: red",
           "Answer: the Centenary Medal", "I cannot answer."]
class Plain:
    def format_prompt(self, question, style):
        return question.question

for options, reference in shapes:
    question = Question(question_id="fixture", category="fixture", question="fixture",
                        options=options, correct_answer=reference)
    for final in outputs:
        runner.call_model_detailed = lambda *args, **kwargs: {
            "text": final, "success": True, "prompt_tokens": 1,
            "completion_tokens": 1, "total_tokens": 2,
            **runner.response_accounting(headers={}, usage=None, finish_reason="stop")}
        predicted = runner.extract_answer(response=final, question=question)
        try:
            expected = runner.process_question_single(
                client=None, model="fixture", question=question, dataset=Plain(),
                prompt_mode="NR", max_tokens=100, temperature=0)["is_correct"]
        except (TypeError, ValueError) as error:
            expected = type(error).__name__
        try:
            actual = _is_correct_answer(question=question, predicted_answer=predicted)
        except (TypeError, ValueError) as error:
            actual = type(error).__name__
        assert actual == expected, (options, reference, final, expected, actual)
        if isinstance(actual, str):
            failures.append({"output": final, "error": actual})
        compared += 1
print(json.dumps({"comparisons": compared, "native_errors": failures}))
""",
        payload={},
    )
    assert result["comparisons"] == 126
    assert result["native_errors"] == [{"output": "Answer: AB", "error": "TypeError"}]


def correct_final(case: dict[str, Any]) -> str:
    reference = case["answer"]["correct_answer"]
    return "Answer: " + (chr(65 + reference) if type(reference) is int else reference)


def wait_for_run(store: Store, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in TERMINAL:
            return run
        time.sleep(0.05)
    raise AssertionError("Native reasoning run did not terminate")


def test_catalog_uses_real_bridge_http_store_and_separate_reports(
    catalog_cases: dict[str, dict[str, Any]], target, tmp_path: Path
) -> None:
    store = Store(tmp_path)
    receipts: list[dict[str, Any]] = []
    for name, exported in catalog_cases.items():
        case = exported["cases"][0]
        target.answer = correct_final(case)
        before = len(target.requests)
        document = manifest(
            target,
            cases=[case],
            benchmark_options=exported["benchmark_options"],
            seed=exported["seed"],
            limits={"total_timeout_s": 10, "idle_timeout_s": 5, "max_run_seconds": 60},
        )
        run = Engine(store).start(document)
        assert wait_for_run(store=store, run_id=run["id"])["status"] == "completed"
        result = store.results(run["id"])[0]
        assert result["status"] == "completed"
        assert result["correct"] is True
        assert result["call_count"] == 1
        assert result["details"]["grader_version"] == PROTOCOL
        assert len(target.requests) == before + 1
        request = target.requests[-1]
        assert request["messages"] == case["messages"]
        assert request["stream"] is True
        assert "PRIVATE-" not in json.dumps(request)
        call = store.calls(run["id"])[0]
        assert call["final"] == target.answer
        assert call["reasoning"] == "The answer might be B."
        report = make_report(store, run["id"])
        assert len(report["benchmarks"]) == 1
        metric = report["benchmarks"][0]
        assert metric["benchmark"] == "legacy-" + name
        assert metric["total"] == 1
        assert metric["accuracy"] == 1
        assert metric["tokens"] == {
            "input_tokens": 7,
            "cached_input_tokens": 2,
            "cache_write_tokens": 1,
            "output_tokens": 3,
        }
        assert metric["cost_usd"] == pytest.approx(18.2 / 1_000_000)
        receipts.append(
            {
                "benchmark": metric["benchmark"],
                "planned": metric["total"],
                "correct": result["correct"],
                "calls": result["call_count"],
                "tokens": metric["tokens"],
            }
        )
    print(json.dumps({"native_catalog_http": receipts}))


@pytest.mark.parametrize(
    ("final", "truncated", "expected_correct"),
    [
        ("Answer: A", False, True),
        ("Answer: a", False, False),
        ("Answer: A or B", False, False),
        ("", False, False),
        ("Answer: A", True, False),
    ],
)
def test_final_only_ambiguity_empty_and_output_limit(
    final: str,
    truncated: bool,
    expected_correct: bool,
    catalog_cases: dict[str, dict[str, Any]],
    target,
    tmp_path: Path,
) -> None:
    exported = catalog_cases["commonsenseqa"]
    target.answer, target.truncated = final, truncated
    document = manifest(
        target,
        cases=exported["cases"][:2],
        benchmark_options=exported["benchmark_options"],
        limits={"total_timeout_s": 10, "idle_timeout_s": 5, "max_run_seconds": 60},
    )
    store = Store(tmp_path)
    run = Engine(store).start(document)
    wait_for_run(store=store, run_id=run["id"])
    results = store.results(run["id"])
    assert len(results) == 2
    assert all(result.get("correct", False) is expected_correct for result in results)
    metric = make_report(store, run["id"])["benchmarks"][0]
    assert metric["total"] == 2
    assert metric["accuracy"] == float(expected_correct)
    if truncated:
        assert all(
            result["details"]["quality_failure"] == "output_limit" for result in results
        )
        assert len(target.requests) == 2
    print(
        json.dumps(
            {
                "final": final,
                "truncated": truncated,
                "planned": metric["total"],
                "accuracy": metric["accuracy"],
                "calls": len(target.requests),
                "statuses": [result["status"] for result in results],
            }
        )
    )


def test_native_grader_error_retains_call_and_all_planned_denominators(
    catalog_cases: dict[str, dict[str, Any]], target, tmp_path: Path
) -> None:
    cases = [exported["cases"][0] for exported in catalog_cases.values()]
    options = {
        key: value
        for exported in catalog_cases.values()
        for key, value in exported["benchmark_options"].items()
    }
    target.answer = "Answer: AB"
    store = Store(tmp_path)
    run = Engine(store).start(
        manifest(
            target,
            cases=cases,
            benchmark_options=options,
            limits={"concurrency": 1, "total_timeout_s": 30, "max_run_seconds": 120},
        )
    )
    assert wait_for_run(store=store, run_id=run["id"])["status"] == "failed"
    report = make_report(store, run["id"])
    assert report["summary"]["targets"][0]["total"] == 15
    assert len(report["benchmarks"]) == 15
    assert all(
        metric["total"] == 1 and metric["accuracy"] == 0
        for metric in report["benchmarks"]
    )
    assert len(target.requests) == 1
    calls = store.calls(run["id"])
    assert calls[0]["final"] == "Answer: AB"
    logs = list((tmp_path / "runs" / run["id"]).glob("*/harness.log"))
    assert len(logs) == 1 and "TypeError" in logs[0].read_text()
    print(
        json.dumps(
            {
                "native_grader_failure": {
                    "planned": 15,
                    "reported_configurations": len(report["benchmarks"]),
                    "subject_calls": len(calls),
                    "native_error": "TypeError",
                }
            }
        )
    )


def test_transport_failure_stops_without_retries(
    catalog_cases: dict[str, dict[str, Any]], target, tmp_path: Path
) -> None:
    class Unavailable(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            target.requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)

    target.RequestHandlerClass = Unavailable
    exported = catalog_cases["gsm8k"]
    store = Store(tmp_path)
    run = Engine(store).start(
        manifest(
            target,
            cases=exported["cases"],
            benchmark_options=exported["benchmark_options"],
            limits={"total_timeout_s": 10, "idle_timeout_s": 5, "max_run_seconds": 60},
        )
    )
    assert wait_for_run(store=store, run_id=run["id"])["status"] == "failed"
    assert len(target.requests) == 1
    metric = make_report(store, run["id"])["benchmarks"][0]
    assert metric["total"] == len(exported["cases"]) == 2
    assert metric["accuracy"] == 0
    assert store.calls(run["id"])[0]["status"] == "failed"
    print(
        json.dumps(
            {"transport_failure": {"http_status": 503, "requests": 1, "planned": 2}}
        )
    )


def test_export_cli_writes_private_frozen_cases_and_refuses_overwrite(
    bundle: Path, target, tmp_path: Path
) -> None:
    template = manifest(target)
    del template["cases"]
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template))
    output = tmp_path / "run.json"
    command = [
        sys.executable,
        "-m",
        "cli.sr_bench.legacy_reasoning_export",
        "--dataset",
        "gsm8k",
        "--source-bundle",
        str(bundle),
        "--manifest",
        str(template_path),
        "--output",
        str(output),
        "--seed",
        "42",
        "--samples-per-category",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_requests"] == 0
    assert output.stat().st_mode & 0o777 == 0o600
    frozen = plan(json.loads(output.read_text()))
    assert frozen["adapter_versions"] == {"legacy-gsm8k": PROTOCOL}
    assert len(frozen["cases"]) == 1
    before = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
    assert not target.requests


def test_missing_optional_runtime_and_private_data_fail_before_dispatch(
    catalog_cases: dict[str, dict[str, Any]], target, tmp_path: Path, monkeypatch
) -> None:
    exported = catalog_cases["gpqa-main"]
    document = manifest(
        target, cases=exported["cases"], benchmark_options=exported["benchmark_options"]
    )
    monkeypatch.setenv(
        "SR_BENCH_LEGACY_REASONING_PYTHON", str(tmp_path / "missing-python")
    )
    with pytest.raises(ValueError, match="Install pinned"):
        Engine(Store(tmp_path / "missing-runtime")).start(document)
    assert not target.requests
    monkeypatch.delenv("SR_BENCH_LEGACY_REASONING_PYTHON")
    invalid = copy.deepcopy(document)
    del invalid["cases"][0]["answer"]["correct_answer"]
    with pytest.raises(ValueError, match="native Question fields"):
        plan(invalid)
    invalid = copy.deepcopy(document)
    invalid["cases"][0]["metadata"]["source_record"] = invalid["cases"][0]["answer"]
    with pytest.raises(ValueError, match="must not carry"):
        plan(invalid)
    assert not target.requests


@pytest.mark.parametrize("fault", ["missing", "empty", "duplicate"])
def test_local_acquisition_failures_are_not_native_empty_success(
    fault: str, tmp_path: Path
) -> None:
    bundle = source_bundle(directory=tmp_path)
    data = json.loads(bundle.read_text())
    if fault == "missing":
        data["sources"] = [
            entry for entry in data["sources"] if entry["configuration"] != "gpqa_main"
        ]
    elif fault == "empty":
        source = next(
            entry for entry in data["sources"] if entry["configuration"] == "gpqa_main"
        )
        (tmp_path / source["path"]).write_text("")
    else:
        data["sources"].append(data["sources"][0])
    bundle.write_text(json.dumps(data))
    with pytest.raises(
        ValueError,
        match=r"Missing local native source|no records|Duplicate native source",
    ):
        export_manifest(manifest={}, dataset="gpqa-main", source_bundle=bundle)
