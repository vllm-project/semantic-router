"""Bounded, native-schema GLiNER2.5 continuation on isolated Decision rows.

The training input is generated only from the approved flattened TRAIN role.
SELECT answers are projected separately with the existing native adapter.
Nothing in this module reads release panels or changes the scoring contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference import gliner25

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

CONTRACT = "gliner25-native-continuation-pilot-v1"
TRAIN_QUOTAS = {"choice": 512, "noul": 360, "score": 152}
SEED = 20260927


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def question_and_target(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build an inference-schema question without copying a target into it."""
    kind = row["task_type"]
    options = row["options"]
    target = options[row["label"]]["key"]
    question: dict[str, Any] = {
        "type": kind,
        "instructions": _text(row["instructions"]),
    }
    if kind == "choice":
        question["criteria"] = {x["key"]: _text(x["description"]) for x in options}
    elif kind == "noul":
        if target not in {"true", "false"}:
            raise ValueError("Noul target is not Boolean")
        target = "yes" if target == "true" else "no"
    elif kind == "score":
        ordered = sorted(options, key=lambda option: int(option["key"]))
        if [option["key"] for option in ordered] != [
            str(i) for i in range(len(ordered))
        ]:
            raise ValueError("Score labels must enumerate levels")
        question["criteria"] = [_text(x["description"]) for x in ordered]
    else:
        raise ValueError(f"Unsupported task type: {kind}")
    return question, target


def native_record(native: Any, row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Compile the same schema that ``gliner25.score_question`` scores."""
    from gliner2.classification.schema import ClassificationSchema

    question, target = question_and_target(row)
    text, aliases = gliner25.prepare_question(row["state"], question)
    schema = ClassificationSchema().single(
        "decision",
        gliner25.schema_labels(question, aliases),
        instruction=gliner25._schema_safe(question["instructions"]),
    )
    compiled = native.compile_schema(schema)
    label = next((alias for alias, key in aliases if key == target), None)
    if label is None:
        raise ValueError("Target label missing from native schema")
    output = compiled.build()
    if len(output["classifications"]) != 1:
        raise ValueError("Expected one native classification task")
    output["classifications"][0]["true_label"] = [label]
    token_count = len(native.model.processor.transform_record(text, output).input_ids)
    return {"input": text, "output": output}, token_count


def _rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()


def select_training_rows(
    eligible: list[dict[str, Any]], quotas: dict[str, int] = TRAIN_QUOTAS
) -> list[dict[str, Any]]:
    """Freeze task quotas with an ID-hash draw, independent of SELECT labels."""
    chosen = []
    for kind, quota in quotas.items():
        pool = sorted((row for row in eligible if row["task_type"] == kind), key=_rank)
        if len(pool) < quota:
            raise ValueError(f"Only {len(pool)} admissible {kind} rows; need {quota}")
        chosen.extend(pool[:quota])
    return sorted(chosen, key=lambda row: row["id"])


def build(
    source: Path, train: Path, select: Path, output: Path, *, max_positions: int = 512
) -> dict[str, Any]:
    """Validate lineage, admit whole short rows, then atomically write private data."""
    if output.exists():
        raise FileExistsError(output)
    gliner25.verify_release(source, gliner25.REVISION)
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    check_partition_isolation({"train": train_rows, "select": select_rows})
    native = gliner25.load_native(source, "cpu")
    eligible = []
    records = {}
    excluded = Counter()
    for row in train_rows:
        record, length = native_record(native, row)
        if length > max_positions:
            excluded[row["task_type"]] += 1
        else:
            eligible.append(row)
            records[row["id"]] = record
    chosen = select_training_rows(eligible)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(
                json.dumps(records[row["id"]], ensure_ascii=False, sort_keys=True)
            )
            stream.write("\n")
    return {
        "contract": CONTRACT,
        "source_revision": gliner25.REVISION,
        "source_weights_sha256": gliner25.MODEL_FILES["model.safetensors"],
        "library_commit": gliner25.LIBRARY_COMMIT,
        "train_sha256": file_sha256(train),
        "select_sha256": file_sha256(select),
        "native_train_sha256": file_sha256(output),
        "native_train_rows": len(chosen),
        "chosen_by_type": dict(sorted(Counter(x["task_type"] for x in chosen).items())),
        "eligible_by_type": dict(
            sorted(Counter(x["task_type"] for x in eligible).items())
        ),
        "excluded_by_type": dict(sorted(excluded.items())),
        "max_positions": max_positions,
    }


def predict_select(
    native: Any, rows: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            question, _ = question_and_target(row)
            try:
                answer = gliner25.score_question(native, row["state"], question)
                counts["valid"] += 1
            except gliner25.NativeContextOverflow as exc:
                answer = {
                    "type": question["type"],
                    "error": "context_overflow",
                    "native_input_tokens": exc.tokens,
                    "native_max_positions": exc.limit,
                }
                counts["context_overflow"] += 1
            stream.write(
                json.dumps({"id": row["id"], "answer": answer}, sort_keys=True)
            )
            stream.write("\n")
    return {"prediction_sha256": file_sha256(output), "counts": dict(counts)}


def score_select(rows: list[dict[str, Any]], predictions: Path) -> dict[str, Any]:
    with predictions.open(encoding="utf-8") as stream:
        answers = [json.loads(line) for line in stream]
    if len(answers) != len(rows) or [x["id"] for x in answers] != [
        x["id"] for x in rows
    ]:
        raise ValueError("SELECT prediction IDs or order mismatch")
    families: dict[str, list[int]] = defaultdict(list)
    types: dict[str, list[int]] = defaultdict(list)
    for row, prediction in zip(rows, answers):
        answer = prediction["answer"]
        question, target = question_and_target(row)
        if answer.get("type") != question["type"]:
            raise ValueError("Wrong prediction type")
        if "error" in answer:
            correct = 0
        elif question["type"] == "choice":
            correct = int(answer.get("choice") == target)
        elif question["type"] == "noul":
            probability = answer.get("noul")
            correct = int(
                isinstance(probability, (int, float))
                and ("yes" if probability > 0.5 else "no") == target
                and probability != 0.5
            )
        else:
            correct = int(answer.get("native_level") == int(target))
        families[row["family"]].append(correct)
        types[row["task_type"]].append(correct)
    family = _summary(families)
    return {
        "prediction_sha256": file_sha256(predictions),
        "correct": sum(sum(items) for items in types.values()),
        "total": len(rows),
        "family_macro_accuracy": sum(x["accuracy"] for x in family.values())
        / len(family),
        "by_type": _summary(types),
        "by_family": family,
    }


def _summary(groups: dict[str, list[int]]) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            "correct": sum(items),
            "total": len(items),
            "accuracy": sum(items) / len(items),
        }
        for name, items in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    build_command = command.add_parser("build")
    build_command.add_argument("--source", type=Path, required=True)
    build_command.add_argument("--train", type=Path, required=True)
    build_command.add_argument("--select", type=Path, required=True)
    build_command.add_argument("--output", type=Path, required=True)
    build_command.add_argument("--manifest", type=Path, required=True)
    predict_command = command.add_parser("predict")
    predict_command.add_argument("--checkpoint", type=Path, required=True)
    predict_command.add_argument("--select", type=Path, required=True)
    predict_command.add_argument("--output", type=Path, required=True)
    predict_command.add_argument("--receipt", type=Path, required=True)
    predict_command.add_argument("--expected-weights-sha256", required=True)
    score_command = command.add_parser("score")
    score_command.add_argument("--select", type=Path, required=True)
    score_command.add_argument("--predictions", type=Path, required=True)
    score_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        receipt = build(args.source, args.train, args.select, args.output)
        args.manifest.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    elif args.command == "predict":
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        checkpoint = args.checkpoint.resolve(strict=True)
        weights = checkpoint / "model.safetensors"
        if file_sha256(weights) != args.expected_weights_sha256:
            raise ValueError("Checkpoint weights hash mismatch")
        if args.expected_weights_sha256 == gliner25.MODEL_FILES["model.safetensors"]:
            native = gliner25.load_native(checkpoint, "cuda:0")
        else:
            import torch
            from gliner2.classification.engine import Classifier

            native = (
                Classifier.from_pretrained(
                    str(checkpoint), device="cuda:0", dtype=torch.float32
                )
                .to("cuda:0")
                .eval()
            )
        rows = load_partition(args.select, "select")
        receipt = predict_select(native, rows, args.output)
        receipt.update(
            {
                "contract": CONTRACT,
                "checkpoint_weights_sha256": args.expected_weights_sha256,
                "select_sha256": file_sha256(args.select),
                "adapter_version": gliner25.ADAPTER_VERSION,
                "library_commit": gliner25.LIBRARY_COMMIT,
            }
        )
        args.receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        rows = load_partition(args.select, "select")
        report = score_select(rows, args.predictions)
        report["select_sha256"] = file_sha256(args.select)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")
        receipt = report
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
