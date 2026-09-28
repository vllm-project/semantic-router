"""Training-data helpers for the modality-routing trainer.

Pure functions with no torch or transformers dependency, so they can be tested and
reused on their own. The class-weight, focal-gamma and oversampling rules mirror
modality_routing_bert_finetuning_lora.main(); they are duplicated here because that
script does not expose them as functions and is too large to refactor in this change.
"""

import json
import math
import random
from dataclasses import dataclass

SEVERE_IMBALANCE_RATIO = 3.0
MILD_IMBALANCE_RATIO = 1.5
OVERSAMPLE_IMBALANCE_RATIO = 2.0
MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 3.0


@dataclass(frozen=True)
class ClassStats:
    """Class balance of a training set and the loss settings derived from it.

    Attributes:
        label_counts: Number of rows per label id.
        class_weights: Loss weight per label id, in label id order.
        focal_gamma: Focusing parameter of the Focal Loss.
        imbalance_ratio: Largest class count over the smallest.
    """

    label_counts: dict[int, int]
    class_weights: list[float]
    focal_gamma: float
    imbalance_ratio: float


def load_jsonl(path: str) -> list[dict]:
    """Load rows written by export_modality_dataset.py.

    Args:
        path: Path to a JSONL file with one {"text", "label", "label_name"} object per line.

    Returns:
        The parsed rows, in file order.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_class_stats(labels: list[int], num_classes: int) -> ClassStats:
    """Compute class weights and the focal-loss gamma from the training labels.

    Weights are inverse frequency with sqrt dampening, clamped to [0.5, 3.0]. The
    gamma grows with the imbalance ratio.

    Args:
        labels: Integer label id of every training row.
        num_classes: Number of classes.

    Returns:
        The class counts, weights, gamma and imbalance ratio.

    Raises:
        ValueError: If labels is empty.
    """
    if not labels:
        raise ValueError("cannot compute class statistics of an empty training set")
    label_counts: dict[int, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        raw_weight = total / (num_classes * count)
        weights.append(
            max(MIN_CLASS_WEIGHT, min(math.sqrt(raw_weight), MAX_CLASS_WEIGHT))
        )

    imbalance_ratio = max(label_counts.values()) / max(min(label_counts.values()), 1)
    if imbalance_ratio > SEVERE_IMBALANCE_RATIO:
        focal_gamma = 3.0
    elif imbalance_ratio > MILD_IMBALANCE_RATIO:
        focal_gamma = 2.0
    else:
        focal_gamma = 1.5
    return ClassStats(label_counts, weights, focal_gamma, imbalance_ratio)


def oversample_minority_classes(
    rows: list[dict], label_counts: dict[int, int], rng: random.Random
) -> list[dict]:
    """Repeat minority-class rows until every class matches the largest one.

    Does nothing if the imbalance ratio is at most OVERSAMPLE_IMBALANCE_RATIO.

    Args:
        rows: Training rows with an integer "label".
        label_counts: Number of rows per label id in rows.
        rng: Random source, so the result is reproducible from a seed.

    Returns:
        The rows to train on, shuffled if any class was oversampled.
    """
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    if max_count / max(min_count, 1) <= OVERSAMPLE_IMBALANCE_RATIO:
        return rows

    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["label"], []).append(row)

    oversampled: list[dict] = []
    for items in buckets.values():
        if len(items) < max_count:
            repeats, remainder = divmod(max_count, len(items))
            oversampled.extend(items * repeats + rng.sample(items, remainder))
        else:
            oversampled.extend(items)
    rng.shuffle(oversampled)
    return oversampled
