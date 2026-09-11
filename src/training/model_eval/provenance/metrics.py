"""Metrics an evaluation manifest has to carry.

These are plain functions over labels, predictions and confidences so a
producer can be tested without loading a model, and so two producers report the
same numbers the same way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

MIN_BIN_COUNT = 2
MAX_BIN_COUNT = 100

__all__ = [
    "abstention_curve",
    "calibration_metrics",
    "classification_metrics",
    "latency_percentiles",
]


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_mapping: dict[str, int],
) -> dict[str, Any]:
    """Accuracy, macro and weighted F1, and the per label breakdown."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("an evaluation needs at least one row")

    per_label: dict[str, dict[str, Any]] = {}
    for name, index in label_mapping.items():
        true_positive = sum(
            1
            for true, pred in zip(y_true, y_pred, strict=True)
            if true == index and pred == index
        )
        predicted = sum(1 for pred in y_pred if pred == index)
        support = sum(1 for true in y_true if true == index)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        per_label[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    rows = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true == pred)
    scores = [entry["f1"] for entry in per_label.values()]
    weights = [entry["support"] for entry in per_label.values()]
    weighted = sum(f1 * support for f1, support in zip(scores, weights, strict=True))
    return {
        "rows": rows,
        "accuracy": correct / rows,
        "macro_f1": sum(scores) / len(scores) if scores else 0.0,
        "weighted_f1": weighted / rows,
        "per_label": per_label,
    }


def calibration_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float],
    bin_count: int = 10,
) -> dict[str, Any]:
    """Expected and maximum calibration error, Brier score, and the bins.

    Confidence is the probability the model gave the label it predicted, so a
    bin's accuracy is how often that prediction was right.
    """
    if not MIN_BIN_COUNT <= bin_count <= MAX_BIN_COUNT:
        raise ValueError(
            f"bin_count must be between {MIN_BIN_COUNT} and {MAX_BIN_COUNT}"
        )
    if not (len(y_true) == len(y_pred) == len(confidences)):
        raise ValueError("labels, predictions and confidences must align")
    if not y_true:
        raise ValueError("an evaluation needs at least one row")

    hits = [1 if true == pred else 0 for true, pred in zip(y_true, y_pred, strict=True)]
    bins: list[dict[str, Any]] = []
    expected = 0.0
    maximum = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            position
            for position, confidence in enumerate(confidences)
            if (confidence > lower or (index == 0 and confidence >= lower))
            and confidence <= upper
        ]
        if not members:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = sum(confidences[position] for position in members) / len(
            members
        )
        accuracy = sum(hits[position] for position in members) / len(members)
        gap = abs(accuracy - mean_confidence)
        expected += gap * len(members) / len(y_true)
        maximum = max(maximum, gap)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    brier = sum(
        (confidence - hit) ** 2
        for confidence, hit in zip(confidences, hits, strict=True)
    ) / len(y_true)
    return {
        "bin_count": bin_count,
        "ece": expected,
        "mce": maximum,
        "brier": brier,
        "bins": bins,
    }


def abstention_curve(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float],
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> dict[str, Any]:
    """Coverage and selective accuracy at each threshold the router could apply."""
    if not (len(y_true) == len(y_pred) == len(confidences)):
        raise ValueError("labels, predictions and confidences must align")
    if not y_true:
        raise ValueError("an evaluation needs at least one row")

    rows = len(y_true)
    curve = []
    for threshold in thresholds:
        kept = [
            position
            for position, confidence in enumerate(confidences)
            if confidence >= threshold
        ]
        selective = (
            sum(1 for position in kept if y_true[position] == y_pred[position])
            / len(kept)
            if kept
            else None
        )
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(kept) / rows,
                "selective_accuracy": selective,
                "abstained": rows - len(kept),
            }
        )
    return {"curve": curve}


def latency_percentiles(samples_ms: Sequence[float]) -> dict[str, float]:
    """Mean and nearest rank percentiles of the per row latencies."""
    if not samples_ms:
        raise ValueError("latency needs at least one sample")
    ordered = sorted(samples_ms)

    def percentile(fraction: float) -> float:
        rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
        return ordered[rank - 1]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }
