"""Compare a fixed SELECT32 checkpoint reload with in-process step output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.model.data import load_partition
from training.model.decision_model import DecisionModel, QWEN3_ARCHITECTURE, encode
from training.model.train import evaluate


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def audit(args: argparse.Namespace) -> dict:
    selected = sorted(
        load_partition(args.select, "select"),
        key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )[:32]
    checkpoint = args.run / "checkpoint-0000001"
    model, tokenizer = DecisionModel.from_checkpoint(checkpoint)
    if model.metadata["architecture"] != QWEN3_ARCHITECTURE:
        raise ValueError("Expected a full official-Qwen3-origin checkpoint")
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
        batch_size=2,
        device=torch.device("cuda:0"),
        output=args.output,
        tag="reload32",
    )
    original = _rows(args.run / "select-step-0000001-predictions.jsonl")
    reloaded = _rows(args.output / "reload32-predictions.jsonl")
    changed = 0
    maximum = 0.0
    for item in selected:
        before, after = original[item["id"]], reloaded[item["id"]]
        if before["token_ids_sha256"] != after["token_ids_sha256"]:
            raise ValueError("Checkpoint reload changed native token IDs")
        changed += before["prediction_key"] != after["prediction_key"]
        left, right = before["answer"], after["answer"]
        if left["type"] == "noul":
            maximum = max(maximum, abs(left["noul"] - right["noul"]))
        else:
            if set(left["probabilities"]) != set(right["probabilities"]):
                raise ValueError("Checkpoint reload changed candidate keys")
            maximum = max(
                maximum,
                *(
                    abs(left["probabilities"][k] - right["probabilities"][k])
                    for k in left["probabilities"]
                ),
            )
    result = {
        "schema": "decision2-qwen3-06-one-step-reload/1",
        "rows": 32,
        "category_changes": changed,
        "max_probability_drift": maximum,
        "status": "PASS" if changed == 0 and maximum <= 1e-4 else "HOLD_PARITY",
    }
    with (args.output / "reload32-comparison.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--select", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(audit(args), sort_keys=True))


if __name__ == "__main__":
    main()
