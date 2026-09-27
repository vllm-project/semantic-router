"""Compare a saved Decision checkpoint with its original native SELECT output.

This is a bounded technical reload check. SELECT is never used here to search
checkpoints or make a release claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _probabilities(row: dict[str, Any]) -> dict[str, float]:
    answer = row["answer"]
    kind = row["task_type"]
    if answer.get("type") != kind:
        raise ValueError("Native answer type mismatch")
    if kind == "noul":
        yes = answer.get("noul")
        probabilities = (
            {"false": 1 - yes, "true": yes} if type(yes) in (int, float) else {}
        )
    elif kind in {"choice", "score"}:
        probabilities = answer.get("probabilities")
    else:
        probabilities = None
    if (
        not isinstance(probabilities, dict)
        or len(probabilities) < 2
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in probabilities.values()
        )
        or abs(sum(probabilities.values()) - 1) > 0.01
        or row.get("prediction_key") not in probabilities
    ):
        raise ValueError("Invalid native probability answer")
    return probabilities


def compare(
    original: list[dict[str, Any]], reloaded: list[dict[str, Any]], n: int
) -> dict[str, Any]:
    if len(original) < n or len(reloaded) != n or n < 1:
        raise ValueError("Incomplete SELECT reload roster")
    left = original[:n]
    if (
        len({row.get("id") for row in left}) != n
        or len({row.get("id") for row in reloaded}) != n
    ):
        raise ValueError("Duplicate SELECT reload ID")
    drifts: list[float] = []
    changes = 0
    by_type: dict[str, int] = {}
    for a, b in zip(left, reloaded, strict=True):
        if any(
            a.get(field) != b.get(field)
            for field in ("id", "task_type", "prompt_sha256", "token_ids_sha256")
        ):
            raise ValueError("SELECT row identity or native input changed")
        kind = a["task_type"]
        by_type[kind] = by_type.get(kind, 0) + 1
        pa, pb = _probabilities(a), _probabilities(b)
        if pa.keys() != pb.keys():
            raise ValueError("Native candidate keys or order changed")
        changes += a["prediction_key"] != b["prediction_key"]
        drifts.append(max(abs(pa[key] - pb[key]) for key in pa))
    p99 = sorted(drifts)[math.ceil(0.99 * n) - 1]
    maximum = max(drifts)
    return {
        "items": n,
        "by_type": by_type,
        "categorical_changes": changes,
        "probability_p99_drift": p99,
        "probability_max_drift": maximum,
        "status": (
            "PASS" if changes == 0 and p99 <= 0.005 and maximum <= 0.02 else "FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-select-sha256", required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--items", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.max_length < 1 or args.items < 1:
        parser.error("Output must be fresh; length and item count must be positive")
    if (
        sha256_file(args.select) != args.expected_select_sha256
        or sha256_file(args.original) != args.expected_original_sha256
    ):
        raise ValueError("SELECT or original predictions changed")

    import torch
    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, encode
    from training.model.train import evaluate

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 accelerator is required")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.checkpoint, source_path=args.source_path
    )
    model = model.float().to(torch.device("cuda:0"))
    model.backbone.config.use_cache = False
    rows = load_partition(args.select, "select")[: args.items]
    if len(rows) != args.items:
        raise ValueError("SELECT has fewer rows than the frozen smoke")
    encoded = [encode(row, tokenizer, args.max_length) for row in rows]
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or EOS token")
    args.output.mkdir(parents=True)
    evaluate(
        model,
        encoded,
        pad_id=pad_id,
        batch_size=2,
        device=torch.device("cuda:0"),
        output=args.output,
        tag="reload",
    )
    torch.cuda.synchronize()
    predictions = args.output / "reload-predictions.jsonl"
    result = compare(_read(args.original), _read(predictions), args.items)
    result.update(
        {
            "select_sha256": args.expected_select_sha256,
            "original_sha256": args.expected_original_sha256,
            "reload_predictions_sha256": sha256_file(predictions),
            "checkpoint_receipt_sha256": sha256_file(
                args.checkpoint / "checkpoint.json"
            ),
        }
    )
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "PASS":
        raise RuntimeError("Native reload drift exceeded the frozen tolerance")


if __name__ == "__main__":
    main()
