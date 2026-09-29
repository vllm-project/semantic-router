"""Explicit streaming evaluation; never invoked on test data during training."""

# ruff: noqa: PLC0415

import argparse
import json
import time
from pathlib import Path

from .data import classification_metrics, file_receipts, read_records
from .model import load_model

LONG_BATCH_THRESHOLD = 2048


def evaluate_records(
    model,
    tokenizer,
    records,
    label_to_id,
    id_to_label,
    max_length,
    batch_size=16,
    dtype="bfloat16",
):
    import torch

    device = next(model.parameters()).device
    if model.config.problem_type != "single_label_classification":
        raise ValueError("Softmax evaluation requires a single-label classifier")
    was_training = model.training
    model.eval()
    encoded = [
        tokenizer(row["text"], truncation=False, padding=False) for row in records
    ]
    lengths = [len(row["input_ids"]) for row in encoded]
    if max(lengths) > max_length:
        raise ValueError(
            "Evaluation input exceeds the explicit budget; truncation is forbidden"
        )
    order = sorted(range(len(records)), key=lengths.__getitem__)
    results, timings, cursor = [None] * len(records), [], 0
    while cursor < len(order):
        size = 1 if lengths[order[cursor]] > LONG_BATCH_THRESHOLD else batch_size
        indices = order[cursor : cursor + size]
        if any(lengths[index] > LONG_BATCH_THRESHOLD for index in indices):
            indices = indices[:1]
        cursor += len(indices)
        batch = tokenizer.pad(
            [encoded[index] for index in indices], padding=True, return_tensors="pt"
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, dtype),
            enabled=device.type == "cuda" and dtype != "float32",
        ):
            logits = model(**batch).logits.float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite evaluation logits")
        probabilities = logits.softmax(-1).cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        for position, index in enumerate(indices):
            scores = probabilities[position]
            results[index] = {
                "id": records[index]["id"],
                "prediction": max(range(len(scores)), key=scores.__getitem__),
                "probabilities": scores,
                "tokens": lengths[index],
            }
    gold = [label_to_id[row["label"]] for row in records]
    predicted = [row["prediction"] for row in results]
    metrics = classification_metrics(gold, predicted, id_to_label)
    metrics["breakdowns"] = {}
    for field in ["source", "language", "length_bucket", "position"]:
        metrics["breakdowns"][field] = {}
        for group in sorted({str(row.get(field, "unspecified")) for row in records}):
            indices = [
                i
                for i, row in enumerate(records)
                if str(row.get(field, "unspecified")) == group
            ]
            metrics["breakdowns"][field][group] = classification_metrics(
                [gold[i] for i in indices], [predicted[i] for i in indices], id_to_label
            )
    metrics["actual_tokens"] = {"min": min(lengths), "max": max(lengths)}
    metrics["forward_batches_ms"] = timings
    metrics["precision"] = {
        "parameters": "float32",
        "forward_autocast": (
            dtype if device.type == "cuda" and dtype != "float32" else None
        ),
        "device_type": device.type,
    }
    if was_training:
        model.train()
    return metrics, results


def main():
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evaluation evidence")
    model, tokenizer, labels, ids = load_model(args.base, args.contract, args.adapter)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Evaluation budget exceeds checkpoint position capacity")
    model.to(args.device)
    rows = read_records(args.data, labels)
    metrics, predictions = evaluate_records(
        model, tokenizer, rows, labels, ids, args.max_length, dtype=args.dtype
    )
    metrics["data_files"] = file_receipts(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in ["rows", "accuracy", "macro_f1", "actual_tokens"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
