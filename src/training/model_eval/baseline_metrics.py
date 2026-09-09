"""Inference and metric summarisation for the Router Model quality baseline.

The router consumes a score and compares it to a configured threshold, so this
module keeps the probability vector rather than collapsing to an argmax, and
reports calibration and threshold behaviour alongside accuracy.

The metric definitions themselves come from ``provenance.metrics``, so a number
this harness writes into an evaluation manifest is computed the same way as the
same number written by the training run.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from provenance.metrics import (
    abstention_curve,
    calibration_metrics,
    classification_metrics,
    discrimination,
    latency_percentiles,
    operating_points,
)
from sklearn.metrics import confusion_matrix

DEFAULT_BIN_COUNT = 10
# 0.0 is the open gate: the reference point a configured threshold is judged
# against, since it answers every row.
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
# Length is the cheapest proxy for the long-context behaviour that separates the
# 8K and 32K artifact families, so it is reported by default.
LENGTH_BOUNDARIES = (64, 256, 1024)


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
    positive_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    truth = labels.tolist()
    predicted = predictions.tolist()
    scores = confidences.tolist()
    label_indices = list(range(len(mapping)))

    metrics = classification_metrics(truth, predicted, mapping)
    metrics["confusion_matrix"] = (
        confusion_matrix(labels, predictions, labels=label_indices).astype(int).tolist()
    )
    abstention = abstention_curve(truth, predicted, scores, thresholds)
    separation = None
    if positive_labels:
        rows = probabilities.tolist()
        abstention["operating_points"] = operating_points(
            truth, rows, mapping, positive_labels, thresholds
        )
        separation = discrimination(truth, rows, mapping, positive_labels)

    latency = latency_percentiles(latencies)
    return {
        "metrics": metrics,
        "discrimination": separation,
        "calibration": calibration_metrics(truth, predicted, scores, bin_count),
        "abstention": abstention,
        "slices": _length_slice_metrics(truth, predicted, texts, mapping),
        "performance": {
            "latency_ms": latency,
            "peak_memory_mb": peak_memory_mb,
            "throughput_rows_per_s": 1000.0 / latency["mean"],
        },
    }


def peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux and bytes on macOS.
    divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
    return float(usage / divisor)


def _length_slice_metrics(
    truth: list[int],
    predicted: list[int],
    texts: list[str],
    mapping: dict[str, int],
) -> list[dict[str, Any]]:
    """Report each length band separately, so the aggregate cannot hide one."""
    lengths = [len(text) for text in texts]
    slices: list[dict[str, Any]] = []
    previous = 0
    for bound in (*LENGTH_BOUNDARIES, None):
        if bound is None:
            name = f"chars>={previous}"
            members = [index for index, size in enumerate(lengths) if size >= previous]
        else:
            name = f"chars<{bound}"
            members = [
                index for index, size in enumerate(lengths) if previous <= size < bound
            ]
            previous = bound
        if not members:
            slices.append(
                {
                    "name": name,
                    "kind": "length",
                    "rows": 0,
                    "accuracy": None,
                    "macro_f1": None,
                }
            )
            continue
        sliced = classification_metrics(
            [truth[index] for index in members],
            [predicted[index] for index in members],
            mapping,
        )
        slices.append(
            {
                "name": name,
                "kind": "length",
                "rows": len(members),
                "accuracy": sliced["accuracy"],
                "macro_f1": sliced["macro_f1"],
            }
        )
    return slices
