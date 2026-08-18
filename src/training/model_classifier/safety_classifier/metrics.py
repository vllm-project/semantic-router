"""Dependency-free metrics for the reconstructed safety classifiers.

The public ``compute_level*_metrics`` helpers intentionally return only scalar
floats so they can be passed straight to a Hugging Face ``Trainer``.  The
``*_classification_metrics`` functions retain confusion matrices, per-class
metrics, support, and threshold metadata for evaluation reports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

BINARY_CLASS_COUNT = 2


def _as_list(values: Any, name: str) -> list[Any]:
    """Materialize Python and array-like inputs without importing NumPy."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable, not a string")
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc


def _integer_vector(values: Any, name: str) -> list[int]:
    result: list[int] = []
    for index, raw_value in enumerate(_as_list(values, name)):
        value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        try:
            numeric = float(value)
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}[{index}] is not an integer label") from exc
        if not math.isfinite(numeric) or numeric != integer:
            raise ValueError(f"{name}[{index}] is not an integer label")
        result.append(integer)
    return result


def _float_vector(values: Any, name: str) -> list[float]:
    result: list[float] = []
    for index, raw_value in enumerate(_as_list(values, name)):
        value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}[{index}] is not numeric") from exc
        if not math.isfinite(score):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(score)
    return result


def _float_matrix(values: Any, name: str) -> list[list[float]]:
    rows = _as_list(values, name)
    matrix = [_float_vector(row, f"{name}[{index}]") for index, row in enumerate(rows)]
    if not matrix:
        return []
    width = len(matrix[0])
    if width == 0:
        raise ValueError(f"{name} rows must not be empty")
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _stable_softmax(row: Sequence[float]) -> list[float]:
    maximum = max(row)
    exponentials = [math.exp(value - maximum) for value in row]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def probabilities_from_logits(logits: Any) -> list[list[float]]:
    """Convert a finite, rectangular logit matrix to softmax probabilities."""
    return [_stable_softmax(row) for row in _float_matrix(logits, "logits")]


def probabilities_from_logits_or_probabilities(values: Any) -> list[list[float]]:
    """Preserve normalized probabilities, otherwise apply stable softmax."""
    matrix = _float_matrix(values, "logits_or_probabilities")
    if all(
        all(0.0 <= value <= 1.0 for value in row)
        and math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        for row in matrix
    ):
        return matrix
    return [_stable_softmax(row) for row in matrix]


def predictions_from_scores(unsafe_scores: Any, threshold: float = 0.5) -> list[int]:
    """Apply the frozen binary rule: ``unsafe_score >= threshold`` is unsafe."""
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return [
        int(score >= threshold)
        for score in _float_vector(unsafe_scores, "unsafe_scores")
    ]


def _validate_lengths(
    labels: Sequence[Any], values: Sequence[Any], value_name: str
) -> None:
    if len(labels) != len(values):
        raise ValueError(
            f"labels and {value_name} must have the same length "
            f"({len(labels)} != {len(values)})"
        )


def _binary_curve_areas(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[float | None, float | None]:
    """Return tie-aware ROC AUC and average precision (reported as AUPRC)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None, None

    ordered = sorted(
        zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True
    )
    true_positives = 0
    false_positives = 0
    previous_tpr = 0.0
    previous_fpr = 0.0
    previous_recall = 0.0
    auroc = 0.0
    average_precision = 0.0
    index = 0

    while index < len(ordered):
        score = ordered[index][0]
        group_positives = 0
        group_negatives = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                group_positives += 1
            else:
                group_negatives += 1
            index += 1

        true_positives += group_positives
        false_positives += group_negatives
        true_positive_rate = true_positives / positives
        false_positive_rate = false_positives / negatives
        auroc += (
            (false_positive_rate - previous_fpr)
            * (true_positive_rate + previous_tpr)
            / 2.0
        )

        precision = true_positives / (true_positives + false_positives)
        average_precision += (true_positive_rate - previous_recall) * precision
        previous_tpr = true_positive_rate
        previous_fpr = false_positive_rate
        previous_recall = true_positive_rate

    return auroc, average_precision


def _binary_inputs(
    labels: Any,
    predictions: Any | None,
    unsafe_scores: Any | None,
    threshold: float,
) -> tuple[list[int], list[int], list[float] | None]:
    label_values = _integer_vector(labels, "labels")
    if any(label not in (0, 1) for label in label_values):
        raise ValueError("binary labels must contain only 0 (safe) and 1 (unsafe)")

    score_values = (
        _float_vector(unsafe_scores, "unsafe_scores")
        if unsafe_scores is not None
        else None
    )
    if score_values is not None:
        _validate_lengths(label_values, score_values, "unsafe_scores")

    if predictions is None:
        if score_values is None:
            raise ValueError("predictions or unsafe_scores must be provided")
        prediction_values = predictions_from_scores(score_values, threshold)
    else:
        prediction_values = _integer_vector(predictions, "predictions")
        _validate_lengths(label_values, prediction_values, "predictions")
        if any(prediction not in (0, 1) for prediction in prediction_values):
            raise ValueError("binary predictions must contain only 0 and 1")
    return label_values, prediction_values, score_values


def binary_classification_metrics(
    labels: Any,
    predictions: Any | None = None,
    *,
    unsafe_scores: Any | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the complete Level-1 binary classification report.

    Undefined divisions use zero. AUROC and AUPRC are ``None`` when either
    class is absent. AUPRC is average precision (a step-wise PR integral).
    """
    label_values, prediction_values, score_values = _binary_inputs(
        labels,
        predictions,
        unsafe_scores,
        threshold,
    )

    true_negative = false_positive = false_negative = true_positive = 0
    for label, prediction in zip(label_values, prediction_values, strict=True):
        if label == 0 and prediction == 0:
            true_negative += 1
        elif label == 0:
            false_positive += 1
        elif prediction == 0:
            false_negative += 1
        else:
            true_positive += 1

    safe_support = true_negative + false_positive
    unsafe_support = true_positive + false_negative
    total = len(label_values)

    unsafe_precision = _safe_divide(true_positive, true_positive + false_positive)
    unsafe_recall = _safe_divide(true_positive, unsafe_support)
    unsafe_f1 = _f1(unsafe_precision, unsafe_recall)
    safe_precision = _safe_divide(true_negative, true_negative + false_negative)
    safe_recall = _safe_divide(true_negative, safe_support)
    safe_f1 = _f1(safe_precision, safe_recall)

    precision_macro = (safe_precision + unsafe_precision) / 2.0
    recall_macro = (safe_recall + unsafe_recall) / 2.0
    f1_macro = (safe_f1 + unsafe_f1) / 2.0
    precision_weighted = _safe_divide(
        safe_precision * safe_support + unsafe_precision * unsafe_support, total
    )
    recall_weighted = _safe_divide(
        safe_recall * safe_support + unsafe_recall * unsafe_support, total
    )
    f1_weighted = _safe_divide(
        safe_f1 * safe_support + unsafe_f1 * unsafe_support, total
    )
    auroc, auprc = (None, None)
    if score_values is not None:
        auroc, auprc = _binary_curve_areas(label_values, score_values)

    return {
        "accuracy": _safe_divide(true_negative + true_positive, total),
        "precision_unsafe": unsafe_precision,
        "recall_unsafe": unsafe_recall,
        "f1_unsafe": unsafe_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "false_positive_rate": _safe_divide(false_positive, safe_support),
        "false_negative_rate": _safe_divide(false_negative, unsafe_support),
        "specificity": safe_recall,
        "auroc": auroc,
        "auprc": auprc,
        "label_order": ["safe", "unsafe"],
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "support": {
            "total": total,
            "safe": safe_support,
            "unsafe": unsafe_support,
        },
        "per_class": {
            "safe": {
                "precision": safe_precision,
                "recall": safe_recall,
                "f1": safe_f1,
                "support": safe_support,
            },
            "unsafe": {
                "precision": unsafe_precision,
                "recall": unsafe_recall,
                "f1": unsafe_f1,
                "support": unsafe_support,
            },
        },
        "threshold": float(threshold),
        "threshold_comparison": "unsafe_score >= threshold",
    }


def freeze_binary_threshold(
    labels: Any,
    unsafe_scores: Any,
    *,
    minimum_recall: float = 0.95,
) -> dict[str, Any]:
    """Select and describe a deterministic validation threshold.

    The primary objective is maximum specificity among observed-score
    thresholds whose unsafe recall is at least ``minimum_recall``. Ties prefer
    higher recall, then the higher threshold. If validation has no unsafe
    examples, the explicit fallback predicts every example safe.
    """
    label_values = _integer_vector(labels, "labels")
    score_values = _float_vector(unsafe_scores, "unsafe_scores")
    _validate_lengths(label_values, score_values, "unsafe_scores")
    if not label_values:
        raise ValueError("threshold selection requires at least one example")
    if any(label not in (0, 1) for label in label_values):
        raise ValueError("binary labels must contain only 0 and 1")
    minimum_recall = float(minimum_recall)
    if not math.isfinite(minimum_recall) or not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")

    tie_break = "maximum specificity, then maximum recall, then highest threshold"
    if sum(label_values) == 0:
        threshold = math.nextafter(max(score_values), math.inf)
        return {
            "threshold": threshold,
            "minimum_recall": minimum_recall,
            "recall_unsafe": 0.0,
            "specificity": 1.0,
            "false_positive_rate": 0.0,
            "fallback": "no unsafe validation examples; predict all examples safe",
            "tie_break": tie_break,
            "threshold_comparison": "unsafe_score >= threshold",
        }

    candidates: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for threshold in sorted(set(score_values)):
        report = binary_classification_metrics(
            label_values,
            unsafe_scores=score_values,
            threshold=threshold,
        )
        recall = report["recall_unsafe"]
        if recall >= minimum_recall:
            candidates.append(
                (
                    (report["specificity"], recall, threshold),
                    report,
                )
            )

    # The minimum observed score predicts every example unsafe, so a positive
    # validation set always has a feasible threshold for minimum_recall <= 1.
    _, selected = max(candidates, key=lambda item: item[0])
    return {
        "threshold": selected["threshold"],
        "minimum_recall": minimum_recall,
        "recall_unsafe": selected["recall_unsafe"],
        "specificity": selected["specificity"],
        "false_positive_rate": selected["false_positive_rate"],
        "fallback": None,
        "tie_break": tie_break,
        "threshold_comparison": "unsafe_score >= threshold",
    }


def _class_report(
    confusion: Sequence[Sequence[int]], class_index: int
) -> dict[str, float | int]:
    true_positive = confusion[class_index][class_index]
    support = sum(confusion[class_index])
    predicted = sum(row[class_index] for row in confusion)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "support": support,
    }


def _top_k_predictions(
    class_scores: Sequence[Sequence[float]], k: int
) -> list[list[int]]:
    return [
        sorted(range(len(row)), key=lambda class_id: (-row[class_id], class_id))[:k]
        for row in class_scores
    ]


def _multiclass_inputs(
    labels: Any,
    predictions: Any | None,
    class_scores: Any | None,
    class_names: Sequence[str] | None,
) -> tuple[list[int], list[int], list[list[float]] | None, list[str], int]:
    label_values = _integer_vector(labels, "labels")
    score_values = (
        _float_matrix(class_scores, "class_scores")
        if class_scores is not None
        else None
    )
    if score_values is not None:
        _validate_lengths(label_values, score_values, "class_scores")
    supplied_predictions = (
        _integer_vector(predictions, "predictions") if predictions is not None else None
    )

    if class_names is not None:
        names = [str(name) for name in class_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("class_names must be non-empty and unique")
        class_count = len(names)
    elif score_values:
        class_count = len(score_values[0])
        names = [str(index) for index in range(class_count)]
    else:
        observed = label_values + (supplied_predictions or [])
        if not observed:
            raise ValueError("class_names are required for an empty report")
        class_count = max(observed) + 1
        names = [str(index) for index in range(class_count)]

    if score_values is not None and any(
        len(row) != class_count for row in score_values
    ):
        raise ValueError("class_scores width must match class_names")
    if supplied_predictions is None:
        if score_values is None:
            raise ValueError("predictions or class_scores must be provided")
        prediction_values = [top[0] for top in _top_k_predictions(score_values, 1)]
    else:
        prediction_values = supplied_predictions
        _validate_lengths(label_values, prediction_values, "predictions")

    if any(not 0 <= label < class_count for label in label_values):
        raise ValueError("labels contain an ID outside the configured classes")
    if any(not 0 <= prediction < class_count for prediction in prediction_values):
        raise ValueError("predictions contain an ID outside the configured classes")
    return label_values, prediction_values, score_values, names, class_count


def _strict_subset_report(
    strict_single_target_mask: Any | None,
    label_values: list[int],
    prediction_values: list[int],
    score_values: list[list[float]] | None,
    names: list[str],
    total: int,
) -> dict[str, Any] | None:
    if strict_single_target_mask is None:
        return None
    mask = [
        bool(value)
        for value in _as_list(
            strict_single_target_mask,
            "strict_single_target_mask",
        )
    ]
    _validate_lengths(label_values, mask, "strict_single_target_mask")
    indices = [index for index, include in enumerate(mask) if include]
    subset = multiclass_classification_metrics(
        [label_values[index] for index in indices],
        [prediction_values[index] for index in indices],
        class_scores=(
            [score_values[index] for index in indices]
            if score_values is not None
            else None
        ),
        class_names=names,
    )
    subset["coverage"] = _safe_divide(len(indices), total)
    return subset


def multiclass_classification_metrics(
    labels: Any,
    predictions: Any | None = None,
    *,
    class_scores: Any | None = None,
    class_names: Sequence[str] | None = None,
    strict_single_target_mask: Any | None = None,
) -> dict[str, Any]:
    """Compute the complete Level-2 report, optionally with a strict subset."""
    label_values, prediction_values, score_values, names, class_count = (
        _multiclass_inputs(labels, predictions, class_scores, class_names)
    )

    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(label_values, prediction_values, strict=True):
        confusion[label][prediction] += 1

    reports = [
        _class_report(confusion, class_index) for class_index in range(class_count)
    ]
    total = len(label_values)
    accuracy = _safe_divide(
        sum(confusion[index][index] for index in range(class_count)), total
    )
    macro_precision = (
        sum(float(report["precision"]) for report in reports) / class_count
    )
    macro_recall = sum(float(report["recall"]) for report in reports) / class_count
    macro_f1 = sum(float(report["f1"]) for report in reports) / class_count
    weighted_precision = _safe_divide(
        sum(float(report["precision"]) * int(report["support"]) for report in reports),
        total,
    )
    weighted_recall = _safe_divide(
        sum(float(report["recall"]) * int(report["support"]) for report in reports),
        total,
    )
    weighted_f1 = _safe_divide(
        sum(float(report["f1"]) * int(report["support"]) for report in reports),
        total,
    )
    supported_recalls = [
        float(report["recall"]) for report in reports if int(report["support"]) > 0
    ]
    balanced_accuracy = _safe_divide(sum(supported_recalls), len(supported_recalls))

    top2_accuracy: float | None = None
    if score_values is not None:
        top2 = _top_k_predictions(score_values, min(2, class_count))
        top2_accuracy = _safe_divide(
            sum(
                label in choices
                for label, choices in zip(label_values, top2, strict=True)
            ),
            total,
        )

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "top2_accuracy": top2_accuracy,
        "label_order": names,
        "confusion_matrix": confusion,
        "support": {
            "total": total,
            "by_class": {
                name: int(report["support"])
                for name, report in zip(names, reports, strict=True)
            },
        },
        "per_class": dict(zip(names, reports, strict=True)),
        "balanced_accuracy_definition": "mean recall over classes present in labels",
        "top2_tie_break": "descending score, then ascending class ID",
    }

    subset = _strict_subset_report(
        strict_single_target_mask,
        label_values,
        prediction_values,
        score_values,
        names,
        total,
    )
    if subset is not None:
        result["strict_single_target"] = subset

    return result


_SCALAR_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision_unsafe",
    "recall_unsafe",
    "f1_unsafe",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "false_positive_rate",
    "false_negative_rate",
    "specificity",
    "auroc",
    "auprc",
    "top2_accuracy",
)


def _flat_float_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(report[name])
        for name in _SCALAR_METRIC_NAMES
        if name in report and report[name] is not None
    }


def _binary_probabilities(
    logits_or_probabilities: Any,
) -> tuple[list[list[float]], list[float]]:
    values = _as_list(logits_or_probabilities, "logits_or_probabilities")
    if not values:
        return [], []
    first = values[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    if isinstance(first, Iterable) and not isinstance(first, (str, bytes)):
        matrix = _float_matrix(values, "logits_or_probabilities")
        if any(len(row) != BINARY_CLASS_COUNT for row in matrix):
            raise ValueError("Level-1 logits must have exactly two columns")
        probabilities = probabilities_from_logits_or_probabilities(matrix)
        return probabilities, [row[1] for row in probabilities]

    one_dimensional = _float_vector(values, "logits_or_probabilities")
    if all(0.0 <= value <= 1.0 for value in one_dimensional):
        unsafe = one_dimensional
    else:
        unsafe = [
            1.0 / (1.0 + math.exp(-max(min(value, 709.0), -709.0)))
            for value in one_dimensional
        ]
    return [[1.0 - score, score] for score in unsafe], unsafe


def classification_metrics(
    task_name: str,
    labels: Any,
    logits_or_probabilities: Any,
    *,
    threshold: float = 0.5,
    class_names: Sequence[str] | None = None,
    strict_single_target_mask: Any | None = None,
) -> dict[str, Any]:
    """Return a complete report from Trainer-style logits for either task."""
    if task_name == "level1":
        _, unsafe_scores = _binary_probabilities(logits_or_probabilities)
        predictions = predictions_from_scores(unsafe_scores, threshold)
        return binary_classification_metrics(
            labels,
            predictions,
            unsafe_scores=unsafe_scores,
            threshold=threshold,
        )
    if task_name == "level2":
        probabilities = probabilities_from_logits_or_probabilities(
            logits_or_probabilities
        )
        return multiclass_classification_metrics(
            labels,
            class_scores=probabilities,
            class_names=class_names,
            strict_single_target_mask=strict_single_target_mask,
        )
    raise ValueError("task_name must be level1 or level2")


def compute_level1_metrics(
    labels: Any,
    logits_or_probabilities: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return Trainer-safe scalar Level-1 metrics from raw two-class logits."""
    report = classification_metrics(
        "level1",
        labels,
        logits_or_probabilities,
        threshold=threshold,
    )
    return _flat_float_metrics(report)


def compute_level2_metrics(
    labels: Any,
    logits_or_probabilities: Any,
    *,
    class_names: Sequence[str] | None = None,
    strict_single_target_mask: Any | None = None,
) -> dict[str, float]:
    """Return Trainer-safe scalar Level-2 metrics from raw multiclass logits."""
    report = classification_metrics(
        "level2",
        labels,
        logits_or_probabilities,
        class_names=class_names,
        strict_single_target_mask=strict_single_target_mask,
    )
    flat = _flat_float_metrics(report)
    if "strict_single_target" in report:
        flat.update(
            {
                f"strict_{name}": value
                for name, value in _flat_float_metrics(
                    report["strict_single_target"]
                ).items()
            }
        )
    return flat
