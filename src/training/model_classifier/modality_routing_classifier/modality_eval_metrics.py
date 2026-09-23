"""Metrics and report assembly for the three-way modality evaluation.

Pure functions over numpy arrays and plain dicts: no model loading and no torch, so
the contamination check, the metrics, the agreement counts and the report layout can
be tested without a GPU or a checkpoint.
"""

import hashlib
import os
import re
from datetime import datetime

import numpy as np
from modality_label_mapping import MODALITY_LABELS
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

MODEL_KEYS = ["published_baseline", "clean_baseline", "candidate"]
AGREEMENT_PAIRS = [
    ("candidate", "clean_baseline"),
    ("candidate", "published_baseline"),
    ("clean_baseline", "published_baseline"),
]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_text(text: str) -> str:
    """Lowercase text and collapse whitespace, for near-duplicate matching.

    Args:
        text: Raw prompt text.

    Returns:
        The normalized text.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_of(text: str) -> str:
    """Hash a prompt so per-example records can be joined without shipping the text.

    Args:
        text: Raw prompt text.

    Returns:
        Hex sha256 of the UTF-8 encoded text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flag_test_contamination(
    train_texts: list[str], val_texts: list[str], test_rows: list[dict]
) -> tuple[list[int], dict[str, float]]:
    """Find test rows whose text also appears in the train or validation split.

    A test row is contaminated if its exact text or its normalized text (see
    normalize_text) also appears in train.jsonl or validation.jsonl.

    Args:
        train_texts: Prompt texts from train.jsonl.
        val_texts: Prompt texts from validation.jsonl.
        test_rows: Rows from test.jsonl.

    Returns:
        (contaminated_row_indices, contamination_rate_by_class).
    """
    train_val_exact = set(train_texts) | set(val_texts)
    train_val_normalized = {normalize_text(t) for t in train_texts + val_texts}

    contaminated_indices = []
    counts_by_class: dict[str, int] = {}
    contaminated_by_class: dict[str, int] = {}

    for idx, row in enumerate(test_rows):
        label_name = row["label_name"]
        counts_by_class[label_name] = counts_by_class.get(label_name, 0) + 1
        text = row["text"]
        is_contaminated = (
            text in train_val_exact or normalize_text(text) in train_val_normalized
        )
        if is_contaminated:
            contaminated_indices.append(idx)
            contaminated_by_class[label_name] = (
                contaminated_by_class.get(label_name, 0) + 1
            )

    rate_by_class = {
        label_name: round(contaminated_by_class.get(label_name, 0) / count, 4)
        for label_name, count in counts_by_class.items()
    }
    return contaminated_indices, rate_by_class


def portable_path(path: str) -> str:
    """Return a path as it should be recorded in a checked-in report.

    Absolute paths would leak the machine's layout and username. Hub repo ids
    ("org/name") are not absolute and are kept as they are.

    Args:
        path: A local path or a Hub repo id.

    Returns:
        The path relative to this directory, or just its name if it lies outside it.
    """
    if not os.path.isabs(path):
        return path
    relative = os.path.relpath(path, _THIS_DIR)
    if not relative.startswith(".."):
        return relative
    return os.path.basename(path.rstrip("/"))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, weighted F1, per-class scores and the confusion matrix.

    Args:
        y_true: Ground-truth canonical label ids.
        y_pred: Predicted canonical label ids.

    Returns:
        Dict with accuracy, f1_weighted, per_class and confusion_matrix.
    """
    labels = list(range(len(MODALITY_LABELS)))
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_weighted = float(
        f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    )
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class = {
        MODALITY_LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }
        for i in range(len(MODALITY_LABELS))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_routing_agreement(
    y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray
) -> dict:
    """Compare two models' predictions example by example.

    Also records which model matched the ground truth where they disagree. The
    order of a and b only affects the labels in the result, not the counts.

    Args:
        y_true: Ground-truth canonical label ids.
        preds_a: Predictions of the first model.
        preds_b: Predictions of the second model.

    Returns:
        Dict with the agreement rate, the disagreement breakdown and the agreement
        rate per ground-truth label.
    """
    n = len(y_true)
    agree_mask = preds_a == preds_b
    num_agree = int(agree_mask.sum())
    num_disagree = n - num_agree

    a_matched_truth_b_wrong = 0
    b_matched_truth_a_wrong = 0
    both_wrong_different_labels = 0
    for i in range(n):
        if agree_mask[i]:
            continue
        a_correct = preds_a[i] == y_true[i]
        b_correct = preds_b[i] == y_true[i]
        if a_correct and not b_correct:
            a_matched_truth_b_wrong += 1
        elif b_correct and not a_correct:
            b_matched_truth_a_wrong += 1
        else:
            both_wrong_different_labels += 1

    def rate_of_disagreements(count: int) -> float | None:
        """Return a count as a share of all disagreements, or None if there are none."""
        return round(count / num_disagree, 4) if num_disagree else None

    agreement_by_true_label = {}
    for i, label_name in enumerate(MODALITY_LABELS):
        mask = y_true == i
        if mask.sum() > 0:
            agreement_by_true_label[label_name] = float(agree_mask[mask].mean())
        else:
            agreement_by_true_label[label_name] = None

    return {
        "agreement_rate": float(num_agree / n) if n else None,
        "num_agree": num_agree,
        "num_disagree": num_disagree,
        "disagreement_breakdown": {
            "a_matched_truth_b_wrong": {
                "count": a_matched_truth_b_wrong,
                "rate_of_disagreements": rate_of_disagreements(a_matched_truth_b_wrong),
            },
            "b_matched_truth_a_wrong": {
                "count": b_matched_truth_a_wrong,
                "rate_of_disagreements": rate_of_disagreements(b_matched_truth_a_wrong),
            },
            "both_wrong_different_labels": {
                "count": both_wrong_different_labels,
                "rate_of_disagreements": rate_of_disagreements(
                    both_wrong_different_labels
                ),
            },
        },
        "agreement_rate_by_ground_truth_label": agreement_by_true_label,
    }


def evaluate_subset(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> dict:
    """Score every model and every agreement pair on one subset of the test set.

    Args:
        y_true: Ground-truth canonical label ids for the subset.
        preds: Predictions per model key, aligned with y_true.

    Returns:
        Dict with the per-model metrics and the pairwise agreement blocks.
    """
    metrics = {
        key: compute_classification_metrics(y_true, preds[key]) for key in MODEL_KEYS
    }
    agreements = {}
    for a, b in AGREEMENT_PAIRS:
        agreements[f"{a}_vs_{b}"] = compute_routing_agreement(
            y_true, preds[a], preds[b]
        )
    return {"metrics": metrics, "routing_agreement": agreements}


def build_per_example_records(
    test_rows: list[dict],
    preds: dict[str, np.ndarray],
    contaminated: set[int],
    include_full_text: bool,
) -> list[dict]:
    """Build one record per test row with every model's prediction.

    Records carry a hash of the prompt so they can be joined to other runs without
    shipping the text.

    Args:
        test_rows: Rows from test.jsonl.
        preds: Canonical label ids per model key, aligned with test_rows.
        contaminated: Indices of contaminated test rows.
        include_full_text: Whether to store each prompt's text as well.

    Returns:
        The per-example records, in row order.
    """
    records = []
    for idx, row in enumerate(test_rows):
        record = {
            "row_index": idx,
            "input_hash_sha256": sha256_of(row["text"]),
            "ground_truth": row["label_name"],
            "contaminated": idx in contaminated,
        }
        for key in MODEL_KEYS:
            record[f"{key}_pred"] = MODALITY_LABELS[int(preds[key][idx])]
        if include_full_text:
            record["text"] = row["text"]
        records.append(record)
    return records


def build_report(
    test_rows: list[dict],
    preds: dict[str, np.ndarray],
    contamination: tuple[list[int], dict[str, float]],
    *,
    paths: dict[str, str],
    model_paths: dict[str, str],
    generated_at: datetime,
    include_full_text: bool = False,
) -> dict:
    """Assemble the full evaluation report.

    Args:
        test_rows: Rows from test.jsonl.
        preds: Canonical label ids per model key, aligned with test_rows.
        contamination: Result of flag_test_contamination.
        paths: The "test_file", "train_file" and "val_file" paths to record.
        model_paths: Checkpoint path or repo id per model key.
        generated_at: Timestamp to record; passed in so reports are reproducible.
        include_full_text: Whether to store each prompt's text in the records.

    Returns:
        The JSON-serialisable report.
    """
    contaminated_indices, contamination_rate_by_class = contamination
    contaminated = set(contaminated_indices)
    y_true = np.array([r["label"] for r in test_rows], dtype=np.int64)

    clean_idx = [i for i in range(len(test_rows)) if i not in contaminated]
    filtered_report = None
    if clean_idx:
        filtered_report = evaluate_subset(
            y_true[clean_idx], {key: preds[key][clean_idx] for key in MODEL_KEYS}
        )

    return {
        "metadata": {
            **{name: portable_path(path) for name, path in paths.items()},
            "num_test_examples": len(test_rows),
            "model_paths": {k: portable_path(v) for k, v in model_paths.items()},
            "generated_at_utc": generated_at.isoformat(),
            "label_order": MODALITY_LABELS,
        },
        "contamination_check": {
            "num_contaminated": len(contaminated_indices),
            "contamination_rate": (
                round(len(contaminated_indices) / len(test_rows), 4)
                if test_rows
                else None
            ),
            "contamination_rate_by_ground_truth_label": contamination_rate_by_class,
            "note": "A test row is flagged if its exact or whitespace/case-normalized "
            "text also appears in train.jsonl/validation.jsonl. See DECISION_RECORD.md "
            "for the source-dataset duplicate-rate EDA that motivated this check.",
        },
        "full_test_set": evaluate_subset(y_true, preds),
        "contamination_filtered_test_set": filtered_report,
        "per_example_records": build_per_example_records(
            test_rows, preds, contaminated, include_full_text
        ),
        "pending_dependencies": {
            "same_run_p99_latency": "PENDING -- blocked on #3856 (same-run evaluation "
            "harness); not computed by this script.",
            "fixed_policy_control_comparison": "PENDING -- blocked on #3857 (fixed-policy "
            "controls); not computed by this script.",
        },
    }
