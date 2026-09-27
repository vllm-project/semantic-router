"""Gold-free JevArena collector that executes a sealed PEFT package loader.

The package supplies the model, tokenizer, CAL and scored inference modules.
This module never imports ``training.model`` from the current source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ADAPTER_VERSION = "decision2-peft-package-native-v1"
PACKAGE_VERSION = "decision2-peft-adapter-package/1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|package-sha256:[0-9a-f]{64})\Z")
KINDS = {"choice", "noul", "score"}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Prompt JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Prompt JSON contains a nonfinite value: {value}")


def load_gold_free(path: Path) -> list[dict[str, Any]]:
    """Reject labels, duplicate IDs/keys and malformed prompt containers."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Gold-free prompt input must be a regular file")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Prompt line {number} is blank")
            row = json.loads(
                line,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
            )
            if not isinstance(row, dict) or set(row) != {"id", "state", "questions"}:
                raise ValueError(
                    f"Prompt line {number} must have only id/state/questions"
                )
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
                raise ValueError(f"Prompt line {number} has a missing or duplicate ID")
            questions = row["questions"]
            if (
                not isinstance(questions, dict)
                or not questions
                or any(not isinstance(key, str) or not key for key in questions)
            ):
                raise ValueError(f"Prompt line {number} has invalid question IDs")
            if any(
                isinstance(question, dict)
                and set(question)
                & {
                    "answer",
                    "correct",
                    "correct_option",
                    "expected",
                    "gold",
                    "label",
                    "target",
                }
                for question in questions.values()
            ):
                raise ValueError(
                    f"Prompt line {number} contains a question answer field"
                )
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Gold-free prompt input is empty")
    return rows


def input_digest(item: dict[str, Any]) -> str:
    # Preserve the original question and option order used by frozen panels.
    payload = {"state": item["state"], "questions": item["questions"]}
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _package_manifest(
    package: Path, expected_sha256: str, model_id: str, model_revision: str
) -> tuple[dict[str, Any], str]:
    if package.is_symlink() or not package.is_dir():
        raise ValueError("Package must be a regular directory")
    if SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Expected package manifest SHA-256 is invalid")
    if REVISION.fullmatch(model_revision) is None:
        raise ValueError("Model revision must be an immutable commit or package digest")
    manifest_path = package / "MODEL_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Package manifest is missing or linked")
    actual = _sha_file(manifest_path)
    if actual != expected_sha256:
        raise ValueError("Package manifest differs from frozen SHA-256")
    if (
        model_revision.startswith("package-sha256:")
        and model_revision.split(":", 1)[1] != actual
    ):
        raise ValueError("Development model revision differs from package bytes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("bundle_version") != PACKAGE_VERSION
        or manifest.get("model_id") != model_id
        or any(
            not isinstance(manifest.get(key), str)
            or SHA256.fullmatch(manifest[key]) is None
            for key in ("model_sha256", "calibration_sha256")
        )
        or type(manifest.get("max_length")) is not int
        or manifest["max_length"] < 1
        or not isinstance(manifest.get("base"), dict)
        or not isinstance(manifest["base"].get("repo_id"), str)
        or not isinstance(manifest["base"].get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", manifest["base"]["revision"]) is None
        or not isinstance(manifest["base"].get("files_sha256"), dict)
        or not manifest["base"]["files_sha256"]
        or not isinstance(manifest.get("loader_files_sha256"), dict)
        or not manifest["loader_files_sha256"]
        or not isinstance(manifest.get("model_files_sha256"), dict)
        or not manifest["model_files_sha256"]
        or not isinstance(manifest.get("dependencies"), dict)
        or any(
            not isinstance(manifest["dependencies"].get(name), str)
            for name in ("torch", "peft")
        )
        or not isinstance(manifest.get("temperature_by_type"), dict)
        or set(manifest["temperature_by_type"]) != KINDS
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in manifest["temperature_by_type"].values()
        )
    ):
        raise ValueError("Package has an incomplete native model contract")
    return manifest, actual


def _import_sealed_package(package: Path) -> Any:
    """Import only the loader copied into this exact package directory."""
    if any(
        name == "decision2" or name.startswith("decision2.") for name in sys.modules
    ):
        raise RuntimeError("Another decision2 runtime is already imported")
    init = package / "decision2/__init__.py"
    if init.is_symlink() or not init.is_file():
        raise ValueError("Packaged native loader is missing or linked")
    spec = importlib.util.spec_from_file_location(
        "decision2", init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the sealed package loader")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["decision2"] = module
        spec.loader.exec_module(module)
        sealed_infer = importlib.import_module("decision2.infer")
    except BaseException:
        for name in list(sys.modules):
            if name == "decision2" or name.startswith("decision2."):
                del sys.modules[name]
        raise
    if (
        Path(module.__file__).resolve() != init.resolve()
        or Path(sealed_infer.__file__).resolve() != (init.parent / "infer.py").resolve()
    ):
        raise RuntimeError(
            "Native inference module came from outside the sealed package"
        )
    return module, sealed_infer


def _probabilities(
    question: dict[str, Any], answer: dict[str, Any]
) -> list[float] | None:
    criteria = question.get("criteria")
    if question.get("type") == "choice" and isinstance(criteria, dict):
        keys = list(criteria)
    elif question.get("type") == "score" and isinstance(criteria, list):
        keys = [str(index) for index in range(len(criteria))]
    else:
        return None
    values = answer.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(keys) or len(keys) < 2:
        return None
    ordered = [values[key] for key in keys]
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in ordered
    ) or not math.isclose(sum(ordered), 1.0, rel_tol=0, abs_tol=1e-5):
        return None
    return [float(value) for value in ordered]


def _valid_answer(question: Any, answer: Any) -> bool:
    if not isinstance(question, dict) or not isinstance(answer, dict):
        return False
    kind = question.get("type")
    if kind not in KINDS or answer.get("type") != kind or "error" in answer:
        return False
    if kind == "noul":
        probability = answer.get("noul")
        return (
            type(probability) in (int, float)
            and math.isfinite(probability)
            and 0 <= probability <= 1
        )
    values = _probabilities(question, answer)
    if values is None:
        return False
    if kind == "choice":
        keys = list(question["criteria"])
        maximum = max(values)
        winners = [
            key for key, value in zip(keys, values) if abs(value - maximum) <= 1e-8
        ]
        return len(winners) == 1 and answer.get("choice") == winners[0]
    score = answer.get("score")
    expected = sum(index * value for index, value in enumerate(values))
    return (
        type(score) in (int, float)
        and math.isfinite(score)
        and math.isclose(score, expected, rel_tol=0, abs_tol=1e-5)
    )


def collect(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    model_id: str,
    model_revision: str,
    manifest: dict[str, Any],
    package_sha256: str,
    adapter_sha256: str,
    question_to_row: Callable[[dict[str, Any], str, Any], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect one whole item at a time; malformed questions remain failures."""
    predictions: list[dict[str, Any]] = []
    counts = {
        "items": 0,
        "questions": 0,
        "valid_questions": 0,
        "invalid_questions": 0,
        "over_budget_questions": 0,
        "truncated_questions": 0,
    }
    for item in rows:
        started = clock()
        eligible = {}
        answers: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for qid, question in item["questions"].items():
            counts["questions"] += 1
            try:
                question_to_row(item, qid, question)
            except ValueError:
                kind = question.get("type") if isinstance(question, dict) else None
                answers[qid] = {"type": kind, "error": "invalid_question"}
                errors[qid] = "invalid_question"
                counts["invalid_questions"] += 1
            else:
                eligible[qid] = question
        usage = {"input_tokens": 0, "output_tokens": 0}
        if eligible:
            response = model.system_one(state=item["state"], questions=eligible)
            if not isinstance(response, dict) or response.get("model") != model_id:
                raise ValueError("Packaged API returned another model identity")
            native = response.get("answers")
            if not isinstance(native, dict) or set(native) - set(eligible):
                raise ValueError("Packaged API returned malformed or extra answers")
            usage = response.get("usage")
            if (
                not isinstance(usage, dict)
                or set(usage) != {"input_tokens", "output_tokens"}
                or any(type(value) is not int or value < 0 for value in usage.values())
            ):
                raise ValueError("Packaged API returned malformed token usage")
            for qid, question in eligible.items():
                answer = native.get(qid)
                if _valid_answer(question, answer):
                    answers[qid] = answer
                    counts["valid_questions"] += 1
                    continue
                if answer is None:
                    reason = "missing_answer"
                elif (
                    isinstance(answer, dict)
                    and isinstance(answer.get("error"), str)
                    and answer["error"]
                    in {
                        "max_length_exceeded",
                        "invalid_question",
                        "invalid_model_output",
                    }
                ):
                    reason = answer["error"]
                else:
                    reason = "invalid_model_output"
                kind = question["type"]
                answers[qid] = {"type": kind, "error": reason}
                errors[qid] = reason
                counts["invalid_questions"] += 1
                if reason == "max_length_exceeded":
                    counts["over_budget_questions"] += 1
        digest = input_digest(item)
        predictions.append(
            {
                "id": item["id"],
                "answers": {qid: answers[qid] for qid in item["questions"]},
                "latency_ms": (clock() - started) * 1000,
                "usage": usage,
                "adapter_status": (
                    "ok"
                    if not errors
                    else (
                        "invalid"
                        if len(errors) == len(item["questions"])
                        else "partial"
                    )
                ),
                "adapter_errors": errors,
                "truncated_questions": 0,
                "input_sha256": digest,
                "source_input_sha256": digest,
                "model_id": model_id,
                "model_revision": model_revision,
                "model_sha256": manifest["model_sha256"],
                "adapter_sha256": adapter_sha256,
                "calibration_sha256": manifest["calibration_sha256"],
                "package_manifest_sha256": package_sha256,
            }
        )
        counts["items"] += 1
    return predictions, counts


def prediction_manifest(
    *,
    package: dict[str, Any],
    package_sha256: str,
    model_id: str,
    model_revision: str,
    input_sha256: str,
    input_items: int,
    adapter_sha256: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    base = package["base"]
    model_files = {
        **{
            f"checkpoint/{name}": digest
            for name, digest in package["model_files_sha256"].items()
        },
        **{f"source/{name}": digest for name, digest in base["files_sha256"].items()},
    }
    return {
        "adapter_version": ADAPTER_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_sha256": package["model_sha256"],
        "model_files_sha256": model_files,
        "checkpoint_format": "peft-lora/1",
        "peft_version": package["dependencies"]["peft"],
        "adapter_sha256": adapter_sha256,
        "adapter_files_sha256": package["loader_files_sha256"],
        "package_manifest_sha256": package_sha256,
        "base_repo_id": base["repo_id"],
        "base_revision": base["revision"],
        "input_sha256": input_sha256,
        "input_items": input_items,
        "max_length": package["max_length"],
        "temperature": 1.0,
        "execution": "sealed package; one item at a time; all valid questions batched; BF16 backbone, FP32 head",
        "truncation_policy": "none; over-budget questions produce invalid answers",
        "counts": counts,
        "torch_version": package["dependencies"]["torch"],
        "calibration_sha256": package["calibration_sha256"],
        "calibration": {
            "file_sha256": package["calibration_sha256"],
            "temperature_by_type": package["temperature_by_type"],
            "binding": "package-native",
        },
    }


def write_predictions(
    path: Path, predictions: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Publish a complete gold-free output and receipt without overwrites."""
    parent = path.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
    ):
        raise ValueError(
            "Prediction output directory must already exist with mode 0700"
        )
    companion = path.with_name(path.name + ".manifest.json")
    pending = path.with_name(path.name + ".pending")
    companion_pending = companion.with_name(companion.name + ".pending")
    if any(
        target.exists() or target.is_symlink()
        for target in (path, companion, pending, companion_pending)
    ):
        raise FileExistsError(
            "Prediction, manifest or interrupted output already exists"
        )

    def private_writer(target: Path) -> Any:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise

    with private_writer(pending) as stream:
        for record in predictions:
            stream.write(
                json.dumps(
                    record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    result = {**manifest, "predictions_sha256": _sha_file(pending)}
    with private_writer(companion_pending) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or companion.exists():
        raise FileExistsError("Prediction or manifest appeared during writing")
    os.replace(companion_pending, companion)
    os.replace(pending, path)
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Exact immutable external base snapshot",
    )
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="Gold-free id/state/questions JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not re.fullmatch(r"cuda:[0-9]+", args.device):
        parser.error("Package-native JevArena runs require an explicit BF16 GPU device")
    if (
        args.package.is_symlink()
        or args.source.is_symlink()
        or args.output.is_symlink()
    ):
        raise ValueError("Package, source and output must not be symlinks")
    package = args.package.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(package) or output.is_relative_to(source):
        raise ValueError("Prediction output cannot modify the package or external base")
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError("Prediction or manifest already exists")
    contract, package_sha256 = _package_manifest(
        package, args.expected_package_sha256, args.model_id, args.model_revision
    )
    prompt_sha256 = _sha_file(args.input)
    rows = load_gold_free(args.input)
    if _sha_file(args.input) != prompt_sha256:
        raise ValueError("Gold-free prompt bytes changed during preflight")
    api, sealed_infer = _import_sealed_package(package)
    if any(input_digest(row) != sealed_infer.prompt_input_sha256(row) for row in rows):
        raise ValueError(
            "Package-native prompt digest differs from frozen panel contract"
        )
    model = api.Decision2.from_pretrained(
        package, source_path=source, device=args.device
    )
    adapter_sha256 = hashlib.sha256(
        _canonical(contract["loader_files_sha256"]).encode("utf-8")
    ).hexdigest()
    predictions, counts = collect(
        rows,
        model=model,
        model_id=args.model_id,
        model_revision=args.model_revision,
        manifest=contract,
        package_sha256=package_sha256,
        adapter_sha256=adapter_sha256,
        question_to_row=sealed_infer.question_to_row,
    )
    if _sha_file(args.input) != prompt_sha256:
        raise ValueError("Gold-free prompt bytes changed during inference")
    if (
        _sha_file(package / "MODEL_MANIFEST.json") != package_sha256
        or api.verify_bundle(package, source) != contract
    ):
        raise ValueError("Package or external base changed during inference")
    receipt = prediction_manifest(
        package=contract,
        package_sha256=package_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        input_sha256=prompt_sha256,
        input_items=len(rows),
        adapter_sha256=adapter_sha256,
        counts=counts,
    )
    receipt = write_predictions(output, predictions, receipt)
    print(
        json.dumps(
            {
                "adapter_version": ADAPTER_VERSION,
                "package_manifest_sha256": package_sha256,
                "predictions_sha256": receipt["predictions_sha256"],
                "counts": counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
