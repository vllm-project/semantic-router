"""Calibration, abstention, and slice metrics for Router Model evaluation.

The router does not consume an argmax. It consumes a score and compares it to a
configured threshold, so accuracy alone cannot say whether a threshold is safe.
These functions report the two things a threshold decision needs: how well the
reported confidence matches observed accuracy, and what coverage and selective
accuracy look like across the threshold range.

All functions take plain numpy arrays so they can be unit tested without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "abstention_curve",
    "expected_calibration_error",
    "length_slices",
    "multiclass_brier_score",
    "reliability_bins",
]

DEFAULT_BIN_COUNT = 10
# Fewer than two bins cannot show a confidence-to-accuracy trend.
MIN_BIN_COUNT = 2
# A probability vector is two dimensional: rows by labels.
PROBABILITY_NDIM = 2
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, bin_count: int = DEFAULT_BIN_COUNT
) -> list[dict[str, Any]]:
    """Bin predictions by confidence and report mean confidence against accuracy.

    Bins are half-open ``[lower, upper)`` except the last, which includes 1.0 so
    that a perfectly confident prediction is never dropped.
    """
    _validate_pair(confidences, correct)
    if bin_count < MIN_BIN_COUNT:
        raise ValueError(f"bin_count must be at least {MIN_BIN_COUNT}")

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        if index == bin_count - 1:
            selected = (confidences >= lower) & (confidences <= upper)
        else:
            selected = (confidences >= lower) & (confidences < upper)
        count = int(selected.sum())
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": float(confidences[selected].mean()) if count else None,
                "accuracy": float(correct[selected].mean()) if count else None,
            }
        )
    return bins


def expected_calibration_error(bins: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Return the sample-weighted (ECE) and worst-bin (MCE) calibration errors."""
    total = sum(int(entry["count"]) for entry in bins)
    if total == 0:
        raise ValueError("cannot compute calibration error over zero predictions")

    weighted = 0.0
    worst = 0.0
    for entry in bins:
        count = int(entry["count"])
        if count == 0:
            continue
        gap = abs(float(entry["confidence"]) - float(entry["accuracy"]))
        weighted += (count / total) * gap
        worst = max(worst, gap)
    return float(weighted), float(worst)


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot truth.

    Normalised by 2 so the result stays in ``[0, 1]`` and stays comparable across
    tasks with different label counts.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != PROBABILITY_NDIM:
        raise ValueError("probabilities must be a 2-D array of shape (rows, labels)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must describe the same rows")
    if probabilities.shape[0] == 0:
        raise ValueError("cannot score zero predictions")
    if labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("labels fall outside the probability columns")

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)) / 2.0)


def abstention_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Coverage and selective accuracy for each threshold the router might use.

    A prediction is answered when its confidence is at or above the threshold.
    ``selective_accuracy`` is the accuracy over answered rows only, so a rising
    threshold that does not raise it means the score carries no useful ranking.
    """
    _validate_pair(confidences, correct)
    rows = confidences.shape[0]
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold {threshold} is outside [0, 1]")
        answered = confidences >= threshold
        answered_count = int(answered.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(answered_count / rows),
                "selective_accuracy": (
                    float(correct[answered].mean()) if answered_count else None
                ),
                "abstained": rows - answered_count,
            }
        )
    return curve


def length_slices(
    texts: Sequence[str], boundaries: Sequence[int] = (64, 256, 1024)
) -> dict[str, np.ndarray]:
    """Group row indices by input character length.

    Length is the cheapest proxy for the long-context behaviour that separates
    the 8K and 32K artifact families, so it is reported by default.
    """
    lengths = np.array([len(text) for text in texts], dtype=np.int64)
    ordered = sorted({int(bound) for bound in boundaries})
    masks: dict[str, np.ndarray] = {}
    previous = 0
    for bound in ordered:
        masks[f"chars<{bound}"] = (lengths >= previous) & (lengths < bound)
        previous = bound
    masks[f"chars>={previous}"] = lengths >= previous
    return masks


def _validate_pair(confidences: np.ndarray, correct: np.ndarray) -> None:
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    if confidences.ndim != 1:
        raise ValueError("confidences must be one dimensional")
    if confidences.shape[0] == 0:
        raise ValueError("cannot summarise zero predictions")
    if float(confidences.min()) < 0.0 or float(confidences.max()) > 1.0:
        raise ValueError("confidences must lie in [0, 1]")
