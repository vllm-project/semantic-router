"""Gold-blind Decision 2.0 adapter for frozen typed-decision benchmark prompts.

The importable conversion/normalization core needs only the Python standard
library. GPU dependencies are loaded by main() after prompt preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .calibration import (
    load_calibration,
    validate_temperatures,
    verified_materialization_origin,
)
from .data import MAX_OPTIONS, canonical, file_sha256
from .lora import LORA_FORMAT, verify_adapter_config
from .source import verify_source

ADAPTER_VERSION = "decision2-typed-benchmark-adapter-v1"
CALIBRATED_ADAPTER_VERSION = "decision2-typed-benchmark-adapter-v2-calibrated"
MODEL_ROOT_FILES = {
    "decision_config.json",
    "decision_head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
}


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != {"id", "state", "questions"}:
                raise ValueError(
                    f"{path}:{line_number}: prompts must contain exactly id/state/questions, never gold"
                )
            if not isinstance(row["id"], str) or not row["id"]:
                raise ValueError(f"{path}:{line_number}: invalid id")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id")
            if not isinstance(row["questions"], dict) or not row["questions"]:
                raise ValueError(
                    f"{path}:{line_number}: questions must be a nonempty object"
                )
            if any(not isinstance(key, str) or not key for key in row["questions"]):
                raise ValueError(f"{path}:{line_number}: invalid question id")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prompts")
    return rows


def question_to_row(
    item: dict[str, Any], question_id: str, question: Any
) -> dict[str, Any]:
    if not isinstance(question, dict) or question.get("type") not in {
        "choice",
        "noul",
        "score",
    }:
        raise ValueError("unsupported or malformed question type")
    kind = question["type"]
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("missing question instructions")
    criteria = question.get("criteria")
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must be an ordered list of 2..10 levels")
        if any(not isinstance(description, str) for description in criteria):
            raise ValueError("score criteria descriptions must be strings")
        options = [
            {"key": str(index), "description": description}
            for index, description in enumerate(criteria)
        ]
    else:
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(
                "choice/noul criteria must be an object with 2..255 options"
            )
        if any(
            not isinstance(key, str) or not key or not isinstance(description, str)
            for key, description in criteria.items()
        ):
            raise ValueError(
                "choice/noul criteria need nonempty string keys and string descriptions"
            )
        if kind == "noul" and set(criteria) != {"false", "true"}:
            raise ValueError("noul requires false and true criteria")
        options = [
            {"key": key, "description": description}
            for key, description in criteria.items()
        ]
    return {
        "id": f"{item['id']}/{question_id}",
        "state": item["state"],
        "instructions": instructions,
        "options": options,
        "task_type": kind,
        "label": 0,  # Required by the shared encoder; never read by inference.
        "family": "benchmark-unknown",  # Family/gold are absent from prompt input.
    }


def normalized_answer(
    kind: str, keys: list[str], logits: list[float], temperature: float
) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(keys) != len(logits) or not 2 <= len(keys) <= MAX_OPTIONS:
        raise ValueError("model returned the wrong number of candidate logits")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in logits
    ):
        raise ValueError("model returned a nonfinite valid-candidate logit")
    scaled = [float(value) / temperature for value in logits]
    top = max(scaled)
    exponentials = [math.exp(value - top) for value in scaled]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    probability_map = dict(zip(keys, probabilities))
    maximum = max(probabilities)
    winners = [
        index
        for index, value in enumerate(probabilities)
        if abs(value - maximum) <= 1e-8
    ]
    winner = winners[0] if len(winners) == 1 else None
    if kind == "noul":
        return {"type": "noul", "noul": probability_map["true"]}
    if kind == "score":
        return {
            "type": "score",
            "score": sum(int(key) * probability_map[key] for key in keys),
            "probabilities": probability_map,
        }
    return {
        "type": "choice",
        "choice": keys[winner] if winner is not None else None,
        "probabilities": probability_map,
    }


def prompt_input_sha256(item: dict[str, Any]) -> str:
    # The model receives only state and each question, never ID or private gold.
    # Preserve insertion order, matching the frozen benchmark and CSS panel
    # payload digests. Training-row hashes use a different, sorted contract.
    payload = {"state": item["state"], "questions": item["questions"]}
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def checkpoint_fingerprint(
    path: Path, source_path: Path | None = None
) -> dict[str, Any]:
    """Hash the full inference identity, including a LoRA checkpoint's source."""
    if (
        not path.is_dir()
        or not (path / "decision_config.json").is_file()
        or not (path / "decision_head.safetensors").is_file()
    ):
        raise ValueError(
            "Checkpoint needs decision_config.json and decision_head.safetensors"
        )
    metadata = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint decision_config.json must be an object")
    # Publication receipts, calibration JSON, and model-card assets may live
    # beside the weights. They are recorded separately and must not alter the
    # identity of the actual model/tokenizer inputs used by this adapter.
    files = [
        file
        for file in path.iterdir()
        if file.is_file() and file.name in MODEL_ROOT_FILES
    ]
    for folder in ("backbone", "adapter"):
        subtree = path / folder
        if subtree.is_dir():
            files.extend(
                file
                for file in subtree.rglob("*")
                if file.is_file()
                and file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
            )
    hashes = {str(file.relative_to(path)): file_sha256(file) for file in sorted(files)}
    if metadata.get("checkpoint_format") == LORA_FORMAT:
        if source_path is None:
            raise ValueError("LoRA checkpoint fingerprint requires --source-path")
        contract = metadata.get("lora")
        if not isinstance(contract, dict):
            raise ValueError("LoRA checkpoint is missing its source contract")
        verify_adapter_config(path / "adapter", contract)
        source = verify_source(source_path, contract.get("source_fingerprint"))
        hashes = {
            **{f"checkpoint/{key}": value for key, value in hashes.items()},
            **{f"source/{key}": value for key, value in source["files_sha256"].items()},
        }
    elif not any(
        file.parent.name == "backbone" and file.suffix in {".safetensors", ".bin"}
        for file in files
    ):
        raise ValueError("Checkpoint is missing backbone weight files")
    return {
        "model_sha256": hashlib.sha256(canonical(hashes).encode("utf-8")).hexdigest(),
        "files_sha256": hashes,
    }


def run_prompts(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    temperature: float | dict[str, float],
    encode_fn: Callable[[dict[str, Any], Any, int], dict[str, Any]],
    predict_fn: Callable[[list[dict[str, Any]]], list[list[float]]],
    model_sha256: str,
    adapter_sha256: str,
    calibration_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run one request at a time so latency means item latency, not batch share."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if isinstance(temperature, dict):
        temperature = validate_temperatures(temperature)
    elif (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
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
        started = time.perf_counter()
        answers: dict[str, Any] = {}
        errors: dict[str, str] = {}
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        usage_tokens = 0
        for question_id, question in item["questions"].items():
            counts["questions"] += 1
            try:
                row = question_to_row(item, question_id, question)
                encoded = encode_fn(row, tokenizer, max_length)
                jobs.append((question_id, row, encoded))
                usage_tokens += len(encoded["ids"])
            except ValueError as exc:
                reason = (
                    "max_length_exceeded"
                    if "exceeds max_length" in str(exc)
                    else "invalid_question"
                )
                errors[question_id] = reason
                answers[question_id] = {
                    "type": (
                        question.get("type") if isinstance(question, dict) else None
                    ),
                    "error": reason,
                }
                counts["invalid_questions"] += 1
                if reason == "max_length_exceeded":
                    counts["over_budget_questions"] += 1
        if jobs:
            scored = predict_fn([encoded for _, _, encoded in jobs])
            if len(scored) != len(jobs):
                raise RuntimeError("predict_fn returned a different number of answers")
            for (question_id, row, _), logits in zip(jobs, scored):
                try:
                    answers[question_id] = normalized_answer(
                        row["task_type"],
                        [option["key"] for option in row["options"]],
                        logits,
                        (
                            temperature[row["task_type"]]
                            if isinstance(temperature, dict)
                            else temperature
                        ),
                    )
                    counts["valid_questions"] += 1
                except ValueError:
                    errors[question_id] = "invalid_model_output"
                    answers[question_id] = {
                        "type": row["task_type"],
                        "error": "invalid_model_output",
                    }
                    counts["invalid_questions"] += 1
        counts["items"] += 1
        record = {
            "id": item["id"],
            "answers": answers,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "usage": {"input_tokens": usage_tokens, "output_tokens": 0},
            "adapter_status": (
                "ok"
                if not errors
                else "invalid" if len(errors) == len(item["questions"]) else "partial"
            ),
            "adapter_errors": errors,
            "truncated_questions": 0,
            "input_sha256": prompt_input_sha256(item),
            "source_input_sha256": prompt_input_sha256(item),
            "model_sha256": model_sha256,
            "adapter_sha256": adapter_sha256,
        }
        if calibration_sha256 is not None:
            record["calibration_sha256"] = calibration_sha256
        predictions.append(record)
    return predictions, counts


def write_output(
    path: Path, predictions: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    metadata_path = path.with_name(path.name + ".manifest.json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError("Prediction or manifest already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"Interrupted pending output exists: {pending}")
    with pending.open("x", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(
                json.dumps(
                    record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    manifest["predictions_sha256"] = file_sha256(pending)
    metadata_pending = metadata_path.with_name(metadata_path.name + ".pending")
    with metadata_pending.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or metadata_path.exists():
        raise FileExistsError("Prediction or manifest appeared while writing")
    # The scoreable prediction path only appears after its receipt is present.
    os.replace(metadata_pending, metadata_path)
    os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Required immutable source for a PEFT LoRA checkpoint",
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Gold-free id/state/questions JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional per-type CAL temperature report bound to this model hash",
    )
    parser.add_argument(
        "--materialization-receipt",
        type=Path,
        help="Receipt for applying a selected LoRA calibration to its merged full checkpoint",
    )
    args = parser.parse_args()
    if (
        args.max_length < 1
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
    ):
        parser.error("max-length and temperature must be positive and finite")
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".manifest.json").exists()
    ):
        raise FileExistsError("Refusing to overwrite predictions or manifest")
    rows = load_prompts(args.input)
    prompt_sha = file_sha256(args.input)
    model_identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    if args.calibration is not None and args.temperature != 1.0:
        parser.error(
            "--calibration and a nondefault scalar --temperature cannot be combined"
        )
    materialization = None
    if args.calibration is not None:
        raw_calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        calibration_model_sha = (
            raw_calibration.get("model_sha256")
            if isinstance(raw_calibration, dict)
            else None
        )
        if calibration_model_sha != model_identity["model_sha256"]:
            materialization = verified_materialization_origin(
                args.checkpoint,
                model_identity["model_sha256"],
                args.materialization_receipt,
            )
        calibration, calibration_report = load_calibration(
            args.calibration,
            model_identity["model_sha256"],
            materialized_source_sha256=(materialization or {}).get(
                "source_model_sha256"
            ),
        )
    else:
        if args.materialization_receipt is not None:
            parser.error("--materialization-receipt requires --calibration")
        calibration, calibration_report = None, None
    adapter_sources = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("infer.py", "decision_model.py", "data.py", "lora.py", "source.py")
    }
    if calibration is not None:
        adapter_sources["calibration.py"] = file_sha256(
            Path(__file__).with_name("calibration.py")
        )
    adapter_sha = hashlib.sha256(canonical(adapter_sources).encode("utf-8")).hexdigest()

    import torch

    from .decision_model import DecisionModel, collate, encode

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA/ROCm BF16 GPU is required")
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.checkpoint, source_path=args.source_path
    )
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")

    def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            key: (
                value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            )
            for key, value in collate(encoded, pad_id).items()
        }
        torch.cuda.synchronize(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(**batch)
        torch.cuda.synchronize(device)
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    predictions, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        temperature=calibration if calibration is not None else args.temperature,
        encode_fn=encode,
        predict_fn=predict,
        model_sha256=model_identity["model_sha256"],
        adapter_sha256=adapter_sha,
        calibration_sha256=(
            file_sha256(args.calibration) if args.calibration is not None else None
        ),
    )
    manifest = {
        "adapter_version": (
            CALIBRATED_ADAPTER_VERSION if calibration is not None else ADAPTER_VERSION
        ),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_sha256": model_identity["model_sha256"],
        "model_files_sha256": model_identity["files_sha256"],
        "checkpoint_format": json.loads(
            (args.checkpoint / "decision_config.json").read_text(encoding="utf-8")
        ).get("checkpoint_format", "full"),
        "peft_version": (
            version("peft") if (args.checkpoint / "adapter").exists() else None
        ),
        "adapter_sha256": adapter_sha,
        "adapter_files_sha256": adapter_sources,
        "input_sha256": prompt_sha,
        "input_items": len(rows),
        "max_length": args.max_length,
        "temperature": args.temperature,
        "execution": "one benchmark item at a time; all its questions batched together; BF16 backbone, FP32 head",
        "truncation_policy": "none; over-budget questions produce an invalid answer",
        "counts": counts,
        "torch_version": torch.__version__,
    }
    if calibration is not None:
        manifest["calibration"] = {
            "file_sha256": file_sha256(args.calibration),
            "cal_sha256": calibration_report["cal_sha256"],
            "temperature_by_type": calibration,
            "binding": (
                "materialized_from_lora" if materialization is not None else "direct"
            ),
        }
        if materialization is not None:
            manifest["calibration"]["materialization_receipt_sha256"] = materialization[
                "receipt_sha256"
            ]
    write_output(args.output, predictions, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": counts,
                "model_sha256": model_identity["model_sha256"],
                "predictions_sha256": manifest["predictions_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
