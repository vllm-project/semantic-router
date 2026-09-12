"""Masked multi-label training primitives and evaluation for Vela Hazard."""

# Heavy dependencies remain optional for corpus validation.
# ruff: noqa: PLC0415

import argparse
import json
import time
from pathlib import Path

LONG_BATCH_THRESHOLD = 2048
MIN_CALIBRATION_SUPPORT = 10


def read_rows(paths, count):
    rows, ids = [], set()
    for path in paths:
        for line in Path(path).read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not row.get("id")
                or row["id"] in ids
                or not row.get("group_id")
                or not row.get("text")
            ):
                raise ValueError("Hazard rows need unique IDs, groups and text")
            targets, mask = row.get("targets"), row.get("label_mask")
            if (
                not isinstance(targets, list)
                or not isinstance(mask, list)
                or len(targets) != count
                or len(mask) != count
            ):
                raise ValueError("Hazard targets/masks must match output dimension")
            if any(value not in {0, 1} for value in targets + mask) or not any(mask):
                raise ValueError(
                    "Targets/masks must be binary and have observed supervision"
                )
            if any(
                target and not observed
                for target, observed in zip(targets, mask, strict=True)
            ):
                raise ValueError("Positive targets cannot be masked")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Empty hazard dataset")
    return rows


def masked_loss(logits, targets, mask, accumulation=1):
    """Mean of per-example mean observed BCE; each example has equal weight."""
    from torch.nn import functional

    if accumulation <= 0 or bool((mask.sum(dim=-1) == 0).any()):
        raise ValueError("Each example needs observed labels and positive accumulation")
    losses = functional.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    return ((losses * mask).sum(dim=-1) / mask.sum(dim=-1)).mean() / accumulation


def score_predictions(rows, probabilities, labels, thresholds=None):
    import numpy as np
    from sklearn.metrics import average_precision_score

    probabilities = np.asarray(probabilities)
    target = np.asarray([row["targets"] for row in rows])
    mask = np.asarray([row["label_mask"] for row in rows], dtype=bool)
    thresholds = np.asarray(
        thresholds if thresholds is not None else [0.5] * len(labels)
    )
    if (
        probabilities.shape != target.shape
        or thresholds.shape != (len(labels),)
        or not np.isfinite(probabilities).all()
        or not np.isfinite(thresholds).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError(
            "Hazard probabilities and thresholds must match the label contract"
        )
    selected = probabilities >= thresholds
    details = {}
    for i, label in enumerate(labels):
        gold, pred = target[mask[:, i], i], selected[mask[:, i], i]
        scores = probabilities[mask[:, i], i]
        tp = int((gold.astype(bool) & pred).sum())
        fp = int((~gold.astype(bool) & pred).sum())
        fn = int((gold.astype(bool) & ~pred).sum())
        details[label] = {
            "positive_support": int(gold.sum()),
            "negative_support": int(len(gold) - gold.sum()),
            "unknown": int((~mask[:, i]).sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": (
                fp / (len(gold) - int(gold.sum())) if len(gold) > gold.sum() else None
            ),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "ap": (
                float(average_precision_score(gold, scores))
                if gold.sum() and gold.sum() < len(gold)
                else None
            ),
            "threshold": float(thresholds[i]),
        }
    ap = [item["ap"] for item in details.values() if item["ap"] is not None]
    safe = np.asarray([row.get("label") == "safe" for row in rows])
    return {
        "rows": len(rows),
        "macro_ap": sum(ap) / len(ap) if ap else None,
        "macro_f1": sum(item["f1"] for item in details.values()) / len(labels),
        "observed_exact_match": float(
            ((selected == target) | ~mask).all(axis=1).mean()
        ),
        "safe_support": int(safe.sum()),
        "safe_any_hazard_rate": (
            float(selected[safe].any(axis=1).mean()) if safe.any() else None
        ),
        "per_label": details,
    }


def calibrate_thresholds(rows, probabilities, labels, minimum=MIN_CALIBRATION_SUPPORT):
    """Development-only maximum-F1 operating points; not posterior calibration."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    values = np.asarray(probabilities)
    thresholds, details = [], {}
    for index, label in enumerate(labels):
        observed = [i for i, row in enumerate(rows) if row["label_mask"][index]]
        gold = np.asarray([rows[i]["targets"][index] for i in observed])
        positive, negative = int(gold.sum()), int(len(gold) - gold.sum())
        threshold, status = (
            0.5,
            "insufficient development support; default not calibrated",
        )
        if min(positive, negative) >= minimum:
            precision, recall, candidates = precision_recall_curve(
                gold, values[observed, index]
            )
            f1 = (
                2
                * precision[:-1]
                * recall[:-1]
                / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            )
            # A tied F1 uses the higher threshold to reduce false positives.
            best = np.flatnonzero(f1 == f1.max())[-1]
            threshold, status = float(candidates[best]), "development maximum F1"
        thresholds.append(threshold)
        details[label] = {
            "threshold": threshold,
            "status": status,
            "positive": positive,
            "negative": negative,
        }
    return thresholds, details


def evaluate(
    model,
    tokenizer,
    rows,
    labels,
    max_length,
    batch_size=8,
    thresholds=None,
    dtype="bfloat16",
):
    import torch

    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Hazard evaluation requires independent sigmoid outputs")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    encoded = [tokenizer(row["text"], truncation=False, padding=False) for row in rows]
    lengths = [len(row["input_ids"]) for row in encoded]
    if max(lengths) > max_length:
        raise ValueError(
            "Hazard evaluation input exceeds explicit budget; no silent truncation"
        )
    order = sorted(range(len(rows)), key=lengths.__getitem__)
    probabilities, timings, cursor = [None] * len(rows), [], 0
    with torch.inference_mode():
        while cursor < len(order):
            indices = order[cursor : cursor + batch_size]
            if any(lengths[index] > LONG_BATCH_THRESHOLD for index in indices):
                indices = indices[:1]
            cursor += len(indices)
            batch = tokenizer.pad(
                [encoded[index] for index in indices], padding=True, return_tensors="pt"
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(
                device.type,
                dtype=getattr(torch, dtype),
                enabled=device.type == "cuda" and dtype != "float32",
            ):
                logits = model(**batch).logits.float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite Hazard evaluation logits")
            scores = logits.sigmoid().cpu().tolist()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            for position, index in enumerate(indices):
                probabilities[index] = scores[position]
    if was_training:
        model.train()
    metrics = score_predictions(rows, probabilities, labels, thresholds)
    metrics["actual_tokens"] = {"min": min(lengths), "max": max(lengths)}
    metrics["forward_batches_ms"] = timings
    metrics["precision"] = {
        "parameters": "float32",
        "forward_autocast": (
            dtype if device.type == "cuda" and dtype != "float32" else None
        ),
        "device_type": device.type,
    }
    metrics["breakdowns"] = {}
    for field in ["source", "language", "length_bucket", "position"]:
        metrics["breakdowns"][field] = {}
        for group in sorted({str(row.get(field, "unspecified")) for row in rows}):
            indices = [
                i
                for i, row in enumerate(rows)
                if str(row.get(field, "unspecified")) == group
            ]
            metrics["breakdowns"][field][group] = score_predictions(
                [rows[i] for i in indices],
                [probabilities[i] for i in indices],
                labels,
                thresholds,
            )
    return metrics, probabilities


def main():
    import torch

    from ..sequence_repair.data import file_receipts
    from ..sequence_repair.model import load_model

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
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibrate-development", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (args.thresholds and args.calibrate_development):
        raise ValueError(
            "Refusing overwrite or simultaneous calibration and frozen thresholds"
        )
    model, tokenizer, _, ids = load_model(args.base, args.contract, args.adapter)
    if not 0 < args.max_length <= model.config.max_position_embeddings:
        raise ValueError("Hazard budget must be within checkpoint position capacity")
    model.to(args.device)
    labels = [ids[i] for i in range(len(ids))]
    rows = read_rows(args.data, len(labels))
    thresholds = None
    if args.thresholds:
        saved = json.loads(args.thresholds.read_text())
        if saved["labels"] != labels:
            raise ValueError("Frozen threshold order differs from model label contract")
        thresholds = saved["thresholds"]
    metrics, probabilities = evaluate(
        model,
        tokenizer,
        rows,
        labels,
        args.max_length,
        thresholds=thresholds,
        dtype=args.dtype,
    )
    metrics["data_files"] = file_receipts(args.data)
    metrics["fixed_threshold_05_metrics"] = score_predictions(
        rows, probabilities, labels
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.calibrate_development:
        thresholds, details = calibrate_thresholds(rows, probabilities, labels)
        metrics["development_fitted_threshold_metrics"] = score_predictions(
            rows, probabilities, labels, thresholds
        )
        args.output.with_suffix(".thresholds.json").write_text(
            json.dumps(
                {
                    "labels": labels,
                    "thresholds": thresholds,
                    "details": details,
                    "data_files": metrics["data_files"],
                    "scope": "development operating points; heldout evaluation required",
                },
                indent=2,
            )
            + "\n"
        )
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(
            json.dumps({"id": row["id"], "probabilities": scores}) + "\n"
            for row, scores in zip(rows, probabilities, strict=True)
        )
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in [
                    "rows",
                    "macro_ap",
                    "macro_f1",
                    "safe_any_hazard_rate",
                    "actual_tokens",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
