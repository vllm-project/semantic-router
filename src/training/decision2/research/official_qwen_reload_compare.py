"""Compare a fixed native SELECT slice before and after checkpoint reload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from training.model.data import load_partition
from training.model.decision_model import (
    ARCHITECTURE,
    QWEN3_ARCHITECTURE,
    DecisionModel,
    encode,
)
from training.model.train import evaluate


def _predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return result


def audit(args: argparse.Namespace) -> dict:
    selected = load_partition(args.select, "select")[: args.rows]
    if len(selected) != args.rows or args.rows % args.batch_size:
        raise ValueError("Fixed slice must contain complete native batch pairs")
    checkpoint = args.run / "checkpoint-0000001"
    model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=args.source_path
    )
    if model.metadata["architecture"] not in (ARCHITECTURE, QWEN3_ARCHITECTURE):
        raise ValueError("Unexpected Decision architecture")
    model = model.float().to("cuda:0").eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or EOS token")
    items = [encode(row, tokenizer, args.max_length) for row in selected]
    evaluate(
        model,
        items,
        pad_id=pad_id,
        batch_size=args.batch_size,
        device=torch.device("cuda:0"),
        output=args.output,
        tag="reload",
    )
    original = _predictions(args.run / "select-step-0000001-predictions.jsonl")
    reloaded = _predictions(args.output / "reload-predictions.jsonl")
    if set(reloaded) != {row["id"] for row in selected}:
        raise ValueError("Reload slice IDs differ")
    changes = 0
    drifts: list[float] = []
    for row in selected:
        before, after = original[row["id"]], reloaded[row["id"]]
        if before["token_ids_sha256"] != after["token_ids_sha256"]:
            raise ValueError("Native token IDs changed")
        if before["answer"]["type"] != after["answer"]["type"]:
            raise ValueError("Native task type changed")
        changes += before["prediction_key"] != after["prediction_key"]
        left, right = before["answer"], after["answer"]
        if left["type"] == "noul":
            drifts.append(abs(left["noul"] - right["noul"]))
        else:
            if set(left["probabilities"]) != set(right["probabilities"]):
                raise ValueError("Candidate domain changed")
            drifts.extend(
                abs(left["probabilities"][key] - right["probabilities"][key])
                for key in left["probabilities"]
            )
    ordered = sorted(drifts)
    p99 = ordered[math.ceil((len(ordered) - 1) * 0.99)]
    maximum = ordered[-1]
    result = {
        "schema": "decision2-official-qwen-one-step-reload/1",
        "rows": args.rows,
        "batch_size": args.batch_size,
        "category_changes": changes,
        "p99_probability_drift": p99,
        "max_probability_drift": maximum,
        "status": (
            "PASS"
            if changes == 0 and p99 <= args.p99_limit and maximum <= args.max_limit
            else "HOLD_PARITY"
        ),
    }
    with (args.output / "reload-comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--select", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--p99-limit", type=float, default=0.005)
    parser.add_argument("--max-limit", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(audit(args), sort_keys=True))


if __name__ == "__main__":
    main()
