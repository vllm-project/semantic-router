# Heavy inference dependencies are lazy so data-contract tests stay dependency-light.
# ruff: noqa: PLC0415
"""Stream strict PII entity evaluation without a padded dataset-wide logit array."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from span_data import decode_entities, entity_metrics, load_label_contract, read_jsonl

PII_LABEL_COUNT = 35
LONG_EVAL_THRESHOLD = 2048


def load_model(base, config_path, adapter=None, trainable=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    label_to_id, id_to_label = load_label_contract(config_path)
    if len(label_to_id) != PII_LABEL_COUNT:
        raise ValueError("Expected the existing 35-label PII contract")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForTokenClassification.from_pretrained(
        base,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        reference_compile=False,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model, tokenizer, label_to_id, id_to_label


def evaluate_records(
    model,
    tokenizer,
    records,
    id_to_label,
    batch_size=16,
    max_length=32768,
    dtype="bfloat16",
    window_tokens=0,
    window_stride=64,
):
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if window_tokens and not 0 <= window_stride < window_tokens - 2:
        raise ValueError(
            "Window overlap must leave room for content and special tokens"
        )
    encodings, owners, original_lengths = [], [], []
    for owner, record in enumerate(records):
        encoding = tokenizer(
            record["full_text"],
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
        )
        if len(encoding["input_ids"]) > max_length:
            raise ValueError(
                f"Evaluation row {record.get('id')} exceeds the explicit budget"
            )
        original_lengths.append(len(encoding["input_ids"]))
        if window_tokens:
            windows = tokenizer(
                record["full_text"],
                truncation=True,
                max_length=window_tokens,
                stride=window_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            for index in range(len(windows["input_ids"])):
                encodings.append(
                    {
                        key: value[index]
                        for key, value in windows.items()
                        if key != "overflow_to_sample_mapping"
                    }
                )
                owners.append(owner)
        else:
            encodings.append(encoding)
            owners.append(owner)
    order = sorted(
        range(len(encodings)), key=lambda index: len(encodings[index]["input_ids"])
    )
    predictions = [set() for _ in records]
    elapsed = []
    cursor = 0
    while cursor < len(order):
        size = (
            1
            if len(encodings[order[cursor]]["input_ids"]) > LONG_EVAL_THRESHOLD
            else batch_size
        )
        indices = order[cursor : cursor + size]
        if any(
            len(encodings[index]["input_ids"]) > LONG_EVAL_THRESHOLD
            for index in indices
        ):
            indices = indices[:1]
        cursor += len(indices)
        inputs = tokenizer.pad(
            [
                {
                    key: value
                    for key, value in encodings[index].items()
                    if key != "offset_mapping"
                }
                for index in indices
            ],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, dtype),
            enabled=device.type == "cuda" and dtype != "float32",
        ):
            logits = model(**inputs).logits
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite evaluation logits")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
        classes = logits.argmax(-1).cpu().tolist()
        for row_index, class_ids in zip(indices, classes, strict=True):
            labels = [
                id_to_label[class_id]
                for class_id in class_ids[: len(encodings[row_index]["input_ids"])]
            ]
            owner = owners[row_index]
            predictions[owner].update(
                decode_entities(
                    records[owner]["full_text"],
                    encodings[row_index]["offset_mapping"],
                    labels,
                )
            )
        del logits, inputs
    predictions = [
        sorted(entities, key=lambda entity: (entity[1], entity[2], entity[0]))
        for entities in predictions
    ]
    entity_types = sorted({label[2:] for label in id_to_label.values() if label != "O"})
    metrics = entity_metrics(records, predictions, entity_types)
    metrics["breakdowns"] = {}
    for field in ("language", "kind", "position", "length_bucket"):
        groups = sorted({str(record.get(field, "unspecified")) for record in records})
        metrics["breakdowns"][field] = {}
        for group in groups:
            indices = [
                index
                for index, record in enumerate(records)
                if str(record.get(field, "unspecified")) == group
            ]
            metrics["breakdowns"][field][group] = entity_metrics(
                [records[index] for index in indices],
                [predictions[index] for index in indices],
                entity_types,
            )
    metrics["forward_batches_ms"] = elapsed
    metrics["actual_tokens"] = {
        "min": min(original_lengths),
        "max": max(original_lengths),
    }
    metrics["inference"] = {
        "mode": "hf-token-overlap-windows" if window_tokens else "full-sequence",
        "window_tokens": window_tokens or None,
        "window_stride": window_stride if window_tokens else None,
        "forward_sequences": len(encodings),
        "max_forward_tokens": max(len(encoding["input_ids"]) for encoding in encodings),
        "parameters": "float32",
        "autocast": dtype if device.type == "cuda" and dtype != "float32" else None,
        "runtime_policy_parity": False,
    }
    if was_training:
        model.train()
    return metrics, predictions


def main():
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-tokens", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evaluation evidence")
    model, tokenizer, _, id_to_label = load_model(args.base, args.config, args.adapter)
    model.to(args.device)
    records = read_jsonl(args.data)
    if not records:
        raise ValueError("An empty evaluation set cannot pass")
    metrics, predictions = evaluate_records(
        model,
        tokenizer,
        records,
        id_to_label,
        args.batch_size,
        dtype=args.dtype,
        window_tokens=args.window_tokens,
        window_stride=args.window_stride,
    )
    metrics["data_file"] = {
        "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "rows": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(
            json.dumps({"id": record["id"], "entities": entities}, ensure_ascii=False)
            + "\n"
            for record, entities in zip(records, predictions, strict=True)
        )
    )
    print(
        json.dumps(
            {
                "rows": metrics["rows"],
                "micro": metrics["micro"],
                "complete_email": metrics["complete_email"],
                "negative_documents": metrics["negative_documents"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
