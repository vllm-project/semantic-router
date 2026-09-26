"""Freeze a gold-free parity roster and audit native LoRA/full predictions.

This is an inference-only source gate for the preregistered Score v6 English
pilot. It never opens a selector key or computes a benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition
from training.model.infer import load_prompts, prompt_input_sha256, question_to_row

PARENT_EN_SELECT_SHA = (
    "e41774cfc6f1dcea58fa0baa3c8a0cef3940af36fcd464f2b0d5ca01598a95ce"
)
SOURCE_MODEL_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
SEED = "decision2-score-v6-en-source-parity-20260927"
ROWS_PER_TYPE = 16
MAX_LENGTH = 1024
PROBABILITY_TOLERANCE = 1e-4


def _sha_order(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{SEED}\0{row['task_type']}\0{row['id']}".encode()
    ).hexdigest()


def _prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            "decision": {
                "type": row["task_type"],
                "instructions": row["instructions"],
                "criteria": {
                    option["key"]: option["description"] for option in row["options"]
                },
            }
        },
    }


def choose_roster(
    rows: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from training.model.decision_model import encode

    prompts: list[dict[str, Any]] = []
    encodings: list[dict[str, Any]] = []
    for kind in ("choice", "noul"):
        eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in sorted(
            (r for r in rows if r["language"] == "en" and r["task_type"] == kind),
            key=_sha_order,
        ):
            prompt = _prompt(row)
            native = question_to_row(
                prompt, "decision", prompt["questions"]["decision"]
            )
            try:
                encoded = encode(native, tokenizer, MAX_LENGTH)
            except ValueError as exc:
                if "exceeds max_length" in str(exc):
                    continue
                raise
            eligible.append((prompt, encoded))
            if len(eligible) == ROWS_PER_TYPE:
                break
        if len(eligible) != ROWS_PER_TYPE:
            raise ValueError(f"Fewer than {ROWS_PER_TYPE} eligible {kind} rows")
        prompts.extend(prompt for prompt, _ in eligible)
        encodings.extend(encoded for _, encoded in eligible)
    if len({row["id"] for row in prompts}) != 2 * ROWS_PER_TYPE:
        raise ValueError("Parity roster contains duplicate IDs")
    return prompts, encodings


def freeze_roster(select: Path, source: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError("Parity output directory already exists")
    if file_sha256(select) != PARENT_EN_SELECT_SHA:
        raise ValueError("Parent English SELECT SHA-256 mismatch")
    if not source.is_dir():
        raise ValueError("Pinned source tokenizer directory is missing")
    rows = load_partition(select, "select")
    if len(rows) != 588 or any(row["language"] != "en" for row in rows):
        raise ValueError("Parent English SELECT balance changed")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    prompts, encodings = choose_roster(rows, tokenizer)
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError("Interrupted pending parity roster exists")
    pending.mkdir(parents=True)
    roster_file = pending / "prompts.jsonl"
    with roster_file.open("x", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(
                json.dumps(prompt, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": "decision2-score-en-source-parity-roster/1",
        "source_model_sha256": SOURCE_MODEL_SHA,
        "select_sha256": PARENT_EN_SELECT_SHA,
        "roster_sha256": file_sha256(roster_file),
        "roster_rows": len(prompts),
        "choice_rows": ROWS_PER_TYPE,
        "noul_rows": ROWS_PER_TYPE,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "input_sha256": [encoded["prompt_sha256"] for encoded in encodings],
        "token_ids_sha256": [encoded["token_ids_sha256"] for encoded in encodings],
        "token_count": [len(encoded["ids"]) for encoded in encodings],
    }
    with (pending / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output)
    return manifest


def _read_prediction(
    path: Path,
    expected_model: str,
    roster_file: Path,
    roster: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("predictions_sha256") != file_sha256(path)
        or manifest.get("input_sha256") != file_sha256(roster_file)
        or manifest.get("model_sha256") != expected_model
        or manifest.get("max_length") != MAX_LENGTH
        or manifest.get("temperature") != 1.0
        or manifest.get("counts", {}).get("valid_questions") != 32
        or manifest.get("counts", {}).get("invalid_questions") != 0
        or manifest.get("counts", {}).get("truncated_questions") != 0
    ):
        raise ValueError("Native prediction manifest failed the frozen parity contract")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 32 or [row.get("id") for row in rows] != [
        item["id"] for item in roster
    ]:
        raise ValueError("Native predictions do not match the frozen roster")
    if any(
        row.get("model_sha256") != expected_model
        or row.get("input_sha256") != prompt_input_sha256(item)
        or row.get("truncated_questions") != 0
        for row, item in zip(rows, roster)
    ):
        raise ValueError("Native prediction rows differ from the roster or model")
    return rows, manifest


def _probabilities(row: dict[str, Any], kind: str) -> tuple[dict[str, float], str]:
    if row.get("adapter_status") != "ok" or row.get("adapter_errors"):
        raise ValueError("Native parity contains an invalid model response")
    answer = row.get("answers", {}).get("decision")
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError("Native parity response has the wrong answer type")
    if kind == "noul":
        truth = answer.get("noul")
        if (
            type(truth) not in (float, int)
            or not math.isfinite(truth)
            or not 0 <= truth <= 1
        ):
            raise ValueError("Invalid Noul probability")
        probs = {"false": 1.0 - truth, "true": float(truth)}
        winner = "true" if truth > 0.5 else "false" if truth < 0.5 else "tie"
    else:
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or any(
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in probs.values()
        ):
            raise ValueError("Invalid Choice probability map")
        winner = answer.get("choice")
        if winner not in probs:
            raise ValueError("Invalid Choice winner")
    return probs, winner


def compare(
    roster_dir: Path,
    source_path: Path,
    merged_path: Path,
    receipt_path: Path,
    source_predictions: Path,
    merged_predictions: Path,
    output: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    if output.exists():
        raise FileExistsError("Parity receipt already exists")
    roster_file = roster_dir / "prompts.jsonl"
    manifest = json.loads((roster_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("roster_sha256") != file_sha256(roster_file)
        or manifest.get("roster_rows") != 32
    ):
        raise ValueError("Parity roster integrity failed")
    materialization = json.loads(receipt_path.read_text(encoding="utf-8"))
    if materialization.get("source_model_sha256") != SOURCE_MODEL_SHA:
        raise ValueError(
            "Materialization source identity differs from the pinned BEST368"
        )
    roster = load_prompts(roster_file)
    for tokenizer_path in (source_path, merged_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        encodings = [
            encode(
                question_to_row(item, "decision", item["questions"]["decision"]),
                tokenizer,
                MAX_LENGTH,
            )
            for item in roster
        ]
        if [entry["prompt_sha256"] for entry in encodings] != manifest[
            "input_sha256"
        ] or [entry["token_ids_sha256"] for entry in encodings] != manifest[
            "token_ids_sha256"
        ]:
            raise ValueError("Source and merged tokenizer/prompt encodings differ")
    original, original_manifest = _read_prediction(
        source_predictions, SOURCE_MODEL_SHA, roster_file, roster
    )
    merged, merged_manifest = _read_prediction(
        merged_predictions, materialization["merged_model_sha256"], roster_file, roster
    )
    if original_manifest.get("adapter_sha256") != merged_manifest.get(
        "adapter_sha256"
    ) or original_manifest.get("execution") != merged_manifest.get("execution"):
        raise ValueError("Native adapter or execution contracts differ")
    max_drift = 0.0
    same = 0
    for index, (item, a, b) in enumerate(zip(roster, original, merged)):
        if (
            a.get("input_sha256") != b.get("input_sha256")
            or a.get("usage", {}).get("input_tokens")
            != b.get("usage", {}).get("input_tokens")
            or a.get("usage", {}).get("input_tokens") != manifest["token_count"][index]
        ):
            raise ValueError("Native inputs or token counts differ")
        kind = item["questions"]["decision"]["type"]
        a_probs, a_winner = _probabilities(a, kind)
        b_probs, b_winner = _probabilities(b, kind)
        if a_probs.keys() != b_probs.keys():
            raise ValueError("Native option sets differ")
        max_drift = max(
            max_drift, *(abs(a_probs[key] - b_probs[key]) for key in a_probs)
        )
        same += a_winner == b_winner
    result = {
        "schema_version": "decision2-score-en-native-source-parity/1",
        "source_model_sha256": SOURCE_MODEL_SHA,
        "merged_model_sha256": materialization["merged_model_sha256"],
        "materialization_receipt_sha256": file_sha256(receipt_path),
        "roster_sha256": file_sha256(roster_file),
        "source_predictions_sha256": file_sha256(source_predictions),
        "merged_predictions_sha256": file_sha256(merged_predictions),
        "rows": len(roster),
        "same_argmax": same,
        "max_absolute_option_probability_drift": max_drift,
        "tolerance": PROBABILITY_TOLERANCE,
        "status": (
            "PASS"
            if same == 32 and max_drift <= PROBABILITY_TOLERANCE
            else "BLOCKED_PARITY"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--select", type=Path, required=True)
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("compare")
    for name in (
        "roster-dir",
        "source-path",
        "merged-path",
        "receipt-path",
        "source-predictions",
        "merged-predictions",
        "output",
    ):
        audit.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        report = freeze_roster(args.select, args.source, args.output)
        print(
            json.dumps(
                {
                    "roster_sha256": report["roster_sha256"],
                    "rows": report["roster_rows"],
                }
            )
        )
    else:
        report = compare(
            args.roster_dir,
            args.source_path,
            args.merged_path,
            args.receipt_path,
            args.source_predictions,
            args.merged_predictions,
            args.output,
        )
        print(json.dumps(report, sort_keys=True))
        if report["status"] != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
