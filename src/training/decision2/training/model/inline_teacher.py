"""Generate private, gold-free native source probabilities on existing TRAIN rows.

The output has no prompt text or labels. Its identities, ordered roster and
source file hashes are verified again when the trainer attaches it in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .data import file_sha256, load_partition
from .decision_model import DecisionModel, collate, encode
from .inline_replay import SCHEMA, roster_sha256
from .source import source_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--materialization-receipt-sha256", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--select-sha256", required=True)
    parser.add_argument("--control-baseline", type=Path, required=True)
    parser.add_argument("--control-baseline-sha256", required=True)
    parser.add_argument("--parity-roster-sha256", required=True)
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_materialization(
    model_path: Path, receipt_path: Path, receipt_sha256: str
) -> tuple[dict[str, str], str]:
    if file_sha256(receipt_path) != receipt_sha256:
        raise ValueError("Materialization receipt hash differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("materialization_version") != "decision2-merged-peft-lora/1":
        raise ValueError("Unsupported source materialization")
    files = receipt.get("merged_model_files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source materialization lacks file inventory")
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or file_sha256(model_path / relative) != expected
        ):
            raise ValueError("Source materialization file differs")
    model_sha = receipt.get("merged_model_sha256")
    if not isinstance(model_sha, str) or len(model_sha) != 64:
        raise ValueError("Source materialization model hash is invalid")
    return source_fingerprint(model_path)["files_sha256"], model_sha


def write_once(path: Path, document: dict) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError(
            "Teacher output must be new inside an existing private directory"
        )
    pending = path.with_name(path.name + ".pending")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    except Exception:
        pending.unlink(missing_ok=True)
        raise


def native_probabilities(
    model: DecisionModel, tokenizer: object, row: dict, max_length: int
) -> tuple[dict, dict]:
    return native_probabilities_batch(model, tokenizer, [row], max_length)[0]


def native_probabilities_batch(
    model: DecisionModel, tokenizer: object, rows: list[dict], max_length: int
) -> list[tuple[dict, dict]]:
    items = [encode(row, tokenizer, max_length) for row in rows]
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Source tokenizer lacks pad and EOS tokens")
    batch = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in collate(items, pad_id).items()
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**batch)
    probabilities = logits.float().softmax(-1).cpu().tolist()
    return [
        (item, dict(zip(item["keys"], all_prob[: len(item["keys"])])))
        for item, all_prob in zip(items, probabilities)
    ]


def source_pairs_for_selection(
    rows: list[dict], selected_ids: set[str]
) -> list[list[dict]]:
    """Preserve historical batch mates and padding for selected SELECT rows."""
    pairs = []
    for start in range(0, len(rows), 2):
        pair = rows[start : start + 2]
        if any(row["id"] in selected_ids for row in pair):
            pairs.append(pair)
    return pairs


def verify_zero_step_parity(
    model: DecisionModel,
    tokenizer: object,
    *,
    select: Path,
    select_sha256: str,
    control_baseline: Path,
    control_baseline_sha256: str,
    parity_roster_sha256: str,
    max_length: int,
) -> dict:
    if file_sha256(select) != select_sha256:
        raise ValueError("SELECT bytes differ from frozen hash")
    if file_sha256(control_baseline) != control_baseline_sha256:
        raise ValueError("Control zero-step predictions differ from frozen hash")
    rows = load_partition(select, "select")
    chosen = sorted(
        rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )[:32]
    if len(chosen) != 32 or roster_sha256(chosen) != parity_roster_sha256:
        raise ValueError("Zero-step parity SELECT roster differs")
    references = {}
    with control_baseline.open(encoding="utf-8") as stream:
        for line in stream:
            reference = json.loads(line)
            identifier = reference["id"]
            if identifier in references:
                raise ValueError("Duplicate control zero-step ID")
            references[identifier] = reference
    if len(references) != len(rows):
        raise ValueError("Control zero-step coverage differs from SELECT")
    largest_drift = 0.0
    checked = 0
    selected_ids = {row["id"] for row in chosen}
    # The historical SELECT baseline was evaluated in batches of two in the
    # original partition order. Keep each selected row's original batch mate:
    # changing padding/shape can change BF16 probabilities even with identical
    # model weights and token IDs.
    for pair in source_pairs_for_selection(rows, selected_ids):
        actuals = native_probabilities_batch(model, tokenizer, pair, max_length)
        for row, (item, probabilities) in zip(pair, actuals):
            if row["id"] not in selected_ids:
                continue
            checked += 1
            reference = references[row["id"]]
            if (
                reference["prompt_sha256"] != item["prompt_sha256"]
                or reference["token_ids_sha256"] != item["token_ids_sha256"]
                or reference["task_type"] != row["task_type"]
            ):
                raise ValueError("Control zero-step native input differs")
            answer = reference["answer"]
            if row["task_type"] == "noul":
                reference_probabilities = {
                    "false": 1.0 - answer["noul"],
                    "true": answer["noul"],
                }
            else:
                reference_probabilities = answer["probabilities"]
            if set(reference_probabilities) != set(probabilities):
                raise ValueError("Control zero-step option keys differ")
            drift = max(
                abs(probabilities[key] - reference_probabilities[key])
                for key in probabilities
            )
            if not math.isfinite(drift):
                raise ValueError("Nonfinite zero-step probability drift")
            largest_drift = max(largest_drift, drift)
            maximum = max(probabilities.values())
            winners = [
                key
                for key, value in probabilities.items()
                if abs(value - maximum) <= 1e-8
            ]
            prediction = winners[0] if len(winners) == 1 else None
            if row["task_type"] == "noul":
                p_true = probabilities["true"]
                prediction = (
                    None if p_true == 0.5 else "true" if p_true > 0.5 else "false"
                )
            if prediction != reference["prediction_key"]:
                raise ValueError("Control zero-step categorical prediction differs")
    if checked != 32:
        raise ValueError("Zero-step parity checked an incomplete roster")
    if largest_drift > 1e-4:
        raise ValueError("Control zero-step probability drift exceeds 1e-4")
    return {
        "status": "PASS",
        "count": len(chosen),
        "max_absolute_probability_drift": largest_drift,
        "control_baseline_sha256": control_baseline_sha256,
        "parity_roster_sha256": parity_roster_sha256,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 source inference needs a supported GPU")
    if args.max_length != 8192 or args.expected_count < 1:
        raise ValueError("The frozen 2B source policy requires 8192 tokens")
    if len(set(args.family)) != len(args.family):
        raise ValueError("Duplicate selected family")
    if file_sha256(args.train) != args.train_sha256:
        raise ValueError("TRAIN bytes differ from frozen hash")
    source_files, model_sha = verify_materialization(
        args.model_path,
        args.materialization_receipt,
        args.materialization_receipt_sha256,
    )
    train_rows = load_partition(args.train, "train")
    chosen = [row for row in train_rows if row["family"] in set(args.family)]
    if (
        len(chosen) != args.expected_count
        or roster_sha256(chosen) != args.roster_sha256
    ):
        raise ValueError("Selected TRAIN roster differs from preregistration")
    chosen_groups = {row["group_id"] for row in chosen}
    if any(
        row["group_id"] in chosen_groups and row["family"] not in args.family
        for row in train_rows
    ):
        raise ValueError("Selected TRAIN source group is incomplete")

    model, tokenizer = DecisionModel.from_checkpoint(args.model_path)
    model = model.float().to(torch.device("cuda:0"))
    model.eval()
    model.backbone.config.use_cache = False
    results = []
    with torch.inference_mode():
        parity = verify_zero_step_parity(
            model,
            tokenizer,
            select=args.select,
            select_sha256=args.select_sha256,
            control_baseline=args.control_baseline,
            control_baseline_sha256=args.control_baseline_sha256,
            parity_roster_sha256=args.parity_roster_sha256,
            max_length=args.max_length,
        )
        for row in chosen:
            _, probabilities = native_probabilities(
                model, tokenizer, row, args.max_length
            )
            results.append(
                {
                    "id": row["id"],
                    "input_sha256": row["input_sha256"],
                    "teacher_probs": probabilities,
                }
            )
    document = {
        "schema_version": SCHEMA,
        "train_sha256": args.train_sha256,
        "source_files_sha256": source_files,
        "source_merged_model_sha256": model_sha,
        "materialization_receipt_sha256": args.materialization_receipt_sha256,
        "roster_sha256": args.roster_sha256,
        "max_length": args.max_length,
        "inference_precision": "FP32 model; BF16 backbone autocast; FP32 decision head",
        "generator_sha256": file_sha256(Path(__file__)),
        "zero_step_parity": parity,
        "rows": results,
    }
    write_once(args.output, document)
    print(
        json.dumps(
            {
                "status": "TEACHER_SEALED",
                "rows": len(results),
                "roster_sha256": args.roster_sha256,
                "teacher_sha256": file_sha256(args.output),
                "generator_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
