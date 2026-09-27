"""Gold-free second-load categorical parity for the fixed paired-order arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.model.data import file_sha256, load_partition

from . import paired_order, pilot, train


def categorical(answer: dict) -> tuple[str, object]:
    kind = answer.get("type")
    if "error" in answer:
        return kind, ("invalid", answer["error"])
    if kind == "choice":
        return kind, answer.get("choice")
    if kind == "noul":
        value = answer.get("noul")
        return kind, bool(value > 0.5) if isinstance(value, (int, float)) else None
    if kind == "score":
        return kind, answer.get("native_level")
    raise ValueError("Unknown native decision type")


def run(source: Path, adapter: Path, select: Path, predictions: Path) -> dict:
    if file_sha256(select) != pilot.SELECT_SHA:
        raise ValueError("SELECT prompts differ")
    rows = load_partition(select, "select")
    with predictions.open(encoding="utf-8") as stream:
        sealed = [json.loads(line) for line in stream]
    if len(rows) != 700 or [r["id"] for r in rows] != [r["id"] for r in sealed]:
        raise ValueError("Sealed SELECT prediction alignment differs")
    by_id = {row["id"]: row for row in rows}
    sealed_by_id = {record["id"]: record["answer"] for record in sealed}
    selected = sorted(
        by_id,
        key=lambda item_id: hashlib.sha256(
            f"{paired_order.CONTRACT}:smoke:{item_id}".encode()
        ).hexdigest(),
    )[: paired_order.SMOKE_ROWS]
    torch, tokenizer, model, no_id, yes_id = paired_order._load_candidate(
        source, adapter
    )
    mismatches = []
    for item_id in selected:
        row = by_id[item_id]
        answer, _ = train.predict_row(
            model,
            torch,
            tokenizer,
            row["state"],
            pilot.row_question(row),
            no_id,
            yes_id,
        )
        if categorical(answer) != categorical(sealed_by_id[item_id]):
            mismatches.append(item_id)
    result = {
        "contract": "qwen3-rerank06-paired-postsave-v1",
        "source": pilot.verify_source(source),
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "select_sha256": file_sha256(select),
        "sealed_prediction_sha256": file_sha256(predictions),
        "second_load_count": len(selected),
        "categorical_matches": len(selected) - len(mismatches),
        "mismatch_count": len(mismatches),
        "scope": "two independent loads of the saved adapter; original in-memory pre-save logits were not retained",
    }
    if mismatches:
        raise ValueError("Saved adapter second-load categorical parity failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.source, args.adapter, args.select, args.predictions)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
