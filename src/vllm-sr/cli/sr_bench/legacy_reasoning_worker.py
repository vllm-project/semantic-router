from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import reasoning
from datasets import Dataset
from packaging.requirements import Requirement
from reasoning.dataset_factory import DatasetFactory
from reasoning.dataset_interface import Question
from reasoning.reasoning_mode_eval import _is_correct_answer
from reasoning.router_reason_bench_multi_dataset import extract_answer

from .canonical import digest
from .dataset_io import MAX_ROWS, bounded_lines, read_small
from .harness_worker import call
from .legacy_reasoning import (
    ALIASES,
    CONFIGURATIONS,
    MAX_EXPORT_BYTES,
    PROTOCOL,
    REQUIREMENTS,
    SOURCE_REVISION,
    case_identity,
    configuration,
    validate_case,
)


def inspect_runtime(source_root: Path) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The legacy reasoning environment requires Python 3.12")
    if (
        Path(reasoning.__file__).resolve()
        != source_root / "bench/reasoning/__init__.py"
    ):
        raise ValueError("Native reasoning imports must come from the pinned checkout")
    for line in REQUIREMENTS.read_text().splitlines():
        requirement = Requirement(line)
        if (requirement.marker is None or requirement.marker.evaluate()) and version(
            requirement.name
        ) not in requirement.specifier:
            raise ValueError(
                "Legacy reasoning dependency differs from its lock: " + requirement.name
            )
    expected = set(CONFIGURATIONS) | set(ALIASES) | {"mmlu", "mmlu-pro"}
    if set(DatasetFactory.get_available_datasets()) != expected:
        raise ValueError("Native reasoning factory differs from the frozen catalog")
    return {
        "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    }


def export_cases(request: dict[str, Any]) -> dict[str, Any]:
    name = configuration(request["dataset"])
    bundle_path = Path(request["source_bundle"])
    bundle = json.loads(read_small(bundle_path, MAX_EXPORT_BYTES))
    if (
        not isinstance(bundle, dict)
        or bundle.get("version") != 1
        or bundle.get("kind") not in {"fixture", "local-import"}
        or not isinstance(bundle.get("sources"), list)
        or not bundle["sources"]
    ):
        raise ValueError("A local source bundle needs version, kind and sources")
    entries: dict[tuple[str, str | None, str], dict[str, Any]] = {}
    for entry in bundle["sources"]:
        if (
            not isinstance(entry, dict)
            or not all(
                isinstance(entry.get(key), str) and entry[key]
                for key in ("dataset", "split", "path", "revision")
            )
            or (
                entry.get("configuration") is not None
                and not isinstance(entry["configuration"], str)
            )
        ):
            raise ValueError(
                "Every local source needs a dataset, split, path and declared revision"
            )
        key = (entry["dataset"], entry.get("configuration"), entry["split"])
        if key in entries:
            raise ValueError("Duplicate native source partition")
        entries[key] = entry

    used: list[dict[str, Any]] = []
    errors: list[Exception] = []
    total_bytes = 0

    def local_load(path: str, name: str | None = None, *, split: str) -> Dataset:
        nonlocal total_bytes
        try:
            entry = entries.get((path, name, split))
            if entry is None:
                raise ValueError(f"Missing local native source: {path}/{name}/{split}")
            source_path = bundle_path.parent / entry["path"]
            checksum = hashlib.sha256()
            records: list[dict[str, Any]] = []
            for line in bounded_lines(
                source_path, maximum=MAX_EXPORT_BYTES - total_bytes
            ):
                checksum.update(line)
                total_bytes += len(line)
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Native source JSONL rows must be objects")
                    records.append(row)
                    if len(records) > MAX_ROWS:
                        raise ValueError(
                            "Native source exceeds the supported row limit"
                        )
            if not records:
                raise ValueError("Native source contains no records")
            used.append(
                {
                    "dataset": path,
                    "configuration": name,
                    "split": split,
                    "revision": entry["revision"],
                    "sha256": checksum.hexdigest(),
                    "records": len(records),
                }
            )
            return Dataset.from_list(records)
        except (OSError, ValueError) as exc:
            # Several native loaders catch acquisition failures and return empty data.
            errors.append(exc)
            raise

    dataset = DatasetFactory.create_dataset(name=name)
    native_module = importlib.import_module(type(dataset).__module__)
    native_load = native_module.load_dataset
    native_module.load_dataset = local_load
    selection = {
        "method": "native-loader",
        "dataset": name,
        "seed": request["seed"],
        "categories": request["categories"],
        "samples_per_category": request["samples_per_category"],
    }
    random.seed(request["seed"])
    np.random.seed(request["seed"])
    try:
        questions, _ = dataset.load_dataset(
            categories=request["categories"],
            samples_per_category=request["samples_per_category"],
            seed=request["seed"],
        )
    finally:
        native_module.load_dataset = native_load
        if errors:
            raise ValueError(str(errors[0])) from errors[0]
    if not questions or len(questions) > MAX_ROWS:
        raise ValueError("Native selection must contain between 1 and 50000 questions")
    source = {
        "kind": bundle["kind"],
        "revision_verification": "declared-revision-with-content-digest",
        "files": used,
    }
    source_sha256 = digest(source)
    benchmark = "legacy-" + name
    config = {
        "protocol": PROTOCOL,
        "source_revision": SOURCE_REVISION,
        "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
        "source": source,
        "selection": selection,
    }
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for question in questions:
        native_question = asdict(question)
        messages = [
            {"role": "user", "content": dataset.format_prompt(question=question)}
        ]
        identity = case_identity(
            benchmark=benchmark,
            source_sha256=source_sha256,
            selection=selection,
            native_id=question.question_id,
        )
        if identity in ids:
            raise ValueError("Native selection contains duplicate question IDs")
        ids.add(identity)
        case = {
            "id": identity,
            "benchmark": benchmark,
            "messages": messages,
            "answer": native_question,
            "metadata": {
                "protocol": PROTOCOL,
                "native_dataset": name,
                "native_question_id": question.question_id,
                "native_question_sha256": digest(native_question),
                "source_sha256": source_sha256,
                "prompt_sha256": digest(messages),
                "stratum": question.category,
            },
        }
        validate_case(case=case, config=config)
        cases.append(case)
    return {"cases": cases, "benchmark_options": {benchmark: config}}


def execute(request: dict[str, Any]) -> dict[str, Any]:
    inspect_runtime(source_root=Path(request["source_root"]))
    case, config = request["case"], request["config"]
    validate_case(case=case, config=config)
    dataset = DatasetFactory.create_dataset(name=case["metadata"]["native_dataset"])
    question = Question(**case["answer"])
    messages = [{"role": "user", "content": dataset.format_prompt(question=question)}]
    if messages != case["messages"]:
        raise ValueError("Frozen prompt differs from the native plain formatter")
    response = call(messages=messages)
    answer = extract_answer(response=response["final"], question=question)
    correct = _is_correct_answer(question=question, predicted_answer=answer)
    return {
        "answer": answer,
        "correct": correct,
        "score": float(correct),
        "details": {
            "grader_version": PROTOCOL,
            "native_dataset": case["metadata"]["native_dataset"],
            "native_question_id": question.question_id,
            "answer_status": "parsed" if answer else "unparsed",
        },
    }


def main() -> None:
    operation, request_path, output_path = sys.argv[1:]
    request = json.loads(read_small(Path(request_path), MAX_EXPORT_BYTES))
    runtime = inspect_runtime(source_root=Path(request["source_root"]))
    if operation == "inspect":
        result = runtime
    elif operation == "export":
        result = export_cases(request)
    else:
        raise ValueError("Unsupported legacy reasoning operation")
    content = (json.dumps(result, ensure_ascii=False) + "\n").encode()
    if len(content) > MAX_EXPORT_BYTES:
        raise ValueError("Legacy reasoning export exceeds the supported byte limit")
    Path(output_path).write_bytes(content)
    Path(output_path).chmod(0o600)


if __name__ == "__main__":
    main()
