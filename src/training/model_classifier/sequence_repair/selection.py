"""Development-only binary operating points under an explicit false-positive budget."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction

BINARY_FP_SELECTION = "binary-fp-budget-recall"
BINARY_LABEL_COUNT = 2


def validate_binary_selection(label_to_id, positive_label, budget):
    """Require an unambiguous binary ontology and a finite, explicit FP budget."""
    if (
        len(label_to_id) != BINARY_LABEL_COUNT
        or set(label_to_id.values()) != {0, 1}
        or any(type(index) is not int for index in label_to_id.values())
    ):
        raise ValueError("FP-budget selection requires exactly two label IDs: 0 and 1")
    if positive_label not in label_to_id:
        raise ValueError("FP-budget selection requires an exact positive label")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or not 0 <= budget <= 1
    ):
        raise ValueError("Selection false-positive budget must be finite in [0, 1]")


def validate_selection_options(selection, label_to_id, positive_label, budget):
    if selection == BINARY_FP_SELECTION:
        validate_binary_selection(label_to_id, positive_label, budget)
    elif positive_label is not None or budget is not None:
        raise ValueError("Binary selection options require binary-fp-budget-recall")


def binary_selection_support(records, label_to_id, positive_label):
    counts = Counter(row["label"] for row in records)
    if set(counts) != set(label_to_id):
        raise ValueError("FP-budget development data must observe both binary labels")
    if any(not row.get("id") for row in records) or len(
        {row["id"] for row in records}
    ) != len(records):
        raise ValueError("Development IDs must be nonempty and unique")
    return {
        "positive": counts[positive_label],
        "negative": len(records) - counts[positive_label],
    }


def binary_checkpoint_key(operating_point):
    """Tie policy also applies across checkpoints; exact ties keep the earlier step."""
    if not operating_point["feasible"]:
        return None
    return (
        operating_point["recall"],
        -operating_point["confusion"]["fp"],
        operating_point["threshold"],
    )


def fit_binary_operating_point(
    records, predictions, label_to_id, positive_label, budget
):
    """Sweep complete tied-score groups with p(positive) >= threshold in [0, 1]."""
    validate_binary_selection(label_to_id, positive_label, budget)
    support = binary_selection_support(records, label_to_id, positive_label)
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in records}:
        raise ValueError("Development prediction IDs must match each row exactly once")
    positive_id = label_to_id[positive_label]
    grouped = defaultdict(lambda: [0, 0])
    evidence = []
    for row in records:
        scores = by_id[row["id"]]["probabilities"]
        if (
            len(scores) != BINARY_LABEL_COUNT
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
                for score in scores
            )
            or not math.isclose(sum(scores), 1.0, rel_tol=0, abs_tol=1e-5)
        ):
            raise ValueError("Expected finite normalized binary probabilities")
        score = scores[positive_id]
        is_positive = row["label"] == positive_label
        grouped[score][0 if is_positive else 1] += 1
        evidence.append(
            {"id": row["id"], "label": row["label"], "probabilities": scores}
        )
    # Decimal budget arithmetic avoids floor(0.3 * 10) rounding surprises without
    # an epsilon that could admit an extra false positive.
    maximum_fp = math.floor(Fraction(str(budget)) * support["negative"])
    receipt = {
        "metric": BINARY_FP_SELECTION,
        "positive_label": positive_label,
        "positive_label_id": positive_id,
        "negative_label": next(
            label for label in label_to_id if label != positive_label
        ),
        "false_positive_budget": budget,
        "maximum_false_positives": maximum_fp,
        "support": support,
        "comparison": "positive_probability >= threshold",
        "threshold_range": [0, 1],
        "tie_break": ["maximum_recall", "fewer_false_positives", "higher_threshold"],
        "evidence_sha256": hashlib.sha256(
            json.dumps(
                sorted(evidence, key=lambda row: row["id"]),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "test_used": False,
        "feasible": False,
        "threshold": None,
        "recall": None,
        "precision": None,
        "false_positive_rate": None,
        "confusion": None,
    }
    best_key, tp, fp = None, 0, 0
    for threshold in sorted({0.0, 1.0, *grouped}, reverse=True):
        positive_count, negative_count = grouped[threshold]
        tp += positive_count
        fp += negative_count
        if fp > maximum_fp:
            continue
        key = (tp, -fp, threshold)
        if best_key is not None and key <= best_key:
            continue
        best_key = key
        receipt.update(
            {
                "feasible": True,
                "threshold": threshold,
                "recall": tp / support["positive"],
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "false_positive_rate": fp / support["negative"],
                "confusion": {
                    "tp": tp,
                    "fp": fp,
                    "fn": support["positive"] - tp,
                    "tn": support["negative"] - fp,
                },
            }
        )
    if not receipt["feasible"]:
        receipt["infeasible_reason"] = (
            "Negative examples at probability 1 exceed the budget even at threshold 1"
        )
    return receipt
