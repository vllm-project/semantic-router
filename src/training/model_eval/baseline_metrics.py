"""Inference and metric summarisation for the Router Model quality baseline.

The router consumes a score and compares it to a configured threshold, so this
module keeps the probability vector rather than collapsing to an argmax, and
reports calibration and threshold behaviour alongside accuracy.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from calibration import (
    abstention_curve,
    expected_calibration_error,
    length_slices,
    multiclass_brier_score,
    reliability_bins,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def predict(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    warmup_batches: int = 1,
) -> tuple[np.ndarray, list[float]]:
    """Return per-row probabilities and per-row latency in milliseconds.

    The first batches pay kernel autotuning and allocator warmup, which lands
    entirely in p99 if it is timed. They are replayed after warmup so every row
    is still scored exactly once.
    """
    for index in range(min(warmup_batches, max(1, len(texts) // max(batch_size, 1)))):
        warmup = texts[index * batch_size : (index + 1) * batch_size]
        if not warmup:
            break
        inputs = tokenizer(
            warmup,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    probabilities: list[np.ndarray] = []
    latencies: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Divide by the number of rows in the batch. Indexing a batch dict by
        # len() counts columns instead, which is how per-row latency ends up
        # scaled by an unrelated constant.
        latencies.extend([elapsed_ms / len(batch)] * len(batch))
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities, axis=0), latencies


def summarise(
    probabilities: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    mapping: dict[str, int],
    latencies: list[float],
    bin_count: int,
    thresholds: tuple[float, ...],
    peak_memory_mb: float,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    index_to_label = {index: name for name, index in mapping.items()}
    label_indices = list(range(len(mapping)))

    bins = reliability_bins(confidences, correct, bin_count)
    ece, mce = expected_calibration_error(bins)
    latency = np.array(latencies, dtype=np.float64)

    return {
        "metrics": {
            "rows": int(labels.shape[0]),
            "accuracy": float(correct.mean()),
            "macro_f1": _macro_f1(labels, predictions, label_indices),
            "weighted_f1": float(
                f1_score(
                    labels,
                    predictions,
                    labels=label_indices,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "per_label": _per_label(labels, predictions, index_to_label),
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=label_indices
            )
            .astype(int)
            .tolist(),
        },
        "calibration": {
            "bin_count": bin_count,
            "ece": ece,
            "mce": mce,
            "brier": multiclass_brier_score(probabilities, labels),
            "bins": bins,
        },
        "abstention": {"curve": abstention_curve(confidences, correct, thresholds)},
        "slices": _length_slice_metrics(
            labels, predictions, correct, texts, label_indices
        ),
        "performance": {
            "latency_ms": {
                "mean": float(latency.mean()),
                "p50": float(np.percentile(latency, 50)),
                "p95": float(np.percentile(latency, 95)),
                "p99": float(np.percentile(latency, 99)),
            },
            "peak_memory_mb": peak_memory_mb,
            "throughput_rows_per_s": float(1000.0 / latency.mean()),
        },
    }


def peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux and bytes on macOS.
    divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
    return float(usage / divisor)


def _per_label(
    labels: np.ndarray, predictions: np.ndarray, index_to_label: dict[int, str]
) -> dict[str, dict[str, Any]]:
    label_indices = sorted(index_to_label)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=label_indices, zero_division=0
    )
    return {
        index_to_label[index]: {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(support[position]),
        }
        for position, index in enumerate(label_indices)
    }


def _macro_f1(
    labels: np.ndarray, predictions: np.ndarray, label_indices: list[int]
) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=label_indices,
            average="macro",
            zero_division=0,
        )
    )


def _length_slice_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    correct: np.ndarray,
    texts: list[str],
    label_indices: list[int],
) -> list[dict[str, Any]]:
    """Report each length band separately, so the aggregate cannot hide one."""
    slices = []
    for name, mask in length_slices(texts).items():
        rows = int(mask.sum())
        slices.append(
            {
                "name": name,
                "kind": "length",
                "rows": rows,
                "accuracy": float(correct[mask].mean()) if rows else None,
                "macro_f1": (
                    _macro_f1(labels[mask], predictions[mask], label_indices)
                    if rows
                    else None
                ),
            }
        )
    return slices
