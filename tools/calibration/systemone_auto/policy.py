"""Small statistical action heads and independent monotone quality calibration."""

from __future__ import annotations

from bisect import bisect_left
from statistics import mean

from .metrics import FEATURE_NAMES

PIVOT_TOLERANCE = 1e-12


def ridge(
    features: list[list[float]], targets: list[float], regularization: float = 1.0
) -> list[float]:
    """Solve the regularized normal equations with partial pivoting, no ML runtime."""
    if not features or len(features) != len(targets) or regularization <= 0:
        raise ValueError("ridge needs paired data and positive regularization")
    width = len(FEATURE_NAMES)
    if any(len(row) != width for row in features):
        raise ValueError("feature contract mismatch")
    matrix = [
        [
            sum(row[i] * row[j] for row in features) + (regularization if i == j else 0)
            for j in range(width)
        ]
        + [sum(row[i] * y for row, y in zip(features, targets, strict=True))]
        for i in range(width)
    ]
    for index in range(width):
        pivot = max(range(index, width), key=lambda i: abs(matrix[i][index]))
        matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
        divisor = matrix[index][index]
        if abs(divisor) < PIVOT_TOLERANCE:
            raise ValueError("singular ridge system")
        matrix[index] = [value / divisor for value in matrix[index]]
        for other in range(width):
            if other != index:
                multiple = matrix[other][index]
                matrix[other] = [
                    a - multiple * b
                    for a, b in zip(matrix[other], matrix[index], strict=True)
                ]
    return [row[-1] for row in matrix]


def predict(weights: list[float], features: list[float]) -> float:
    return sum(a * b for a, b in zip(weights, features, strict=True))


def terminal_result(source: dict, action: dict) -> dict:
    """Learned policy retains a previously accepted bundle if an upgrade fails."""
    if delivery_eligible(action):
        return action
    return source if delivery_eligible(source) else unresolved_result(action)


def delivery_eligible(result: dict) -> bool:
    """Match the common whole-bundle native top_probability >= 0 floor."""
    return bool(
        result["valid"]
        and result.get("questions")
        and all(
            q.get("valid") and q.get("top_probability") is not None
            for q in result["questions"].values()
        )
    )


def unresolved_result(result: dict) -> dict:
    """An undeliverable bundle fails as a whole, even with valid point answers."""
    return {
        **result,
        "valid": False,
        "correct": False,
        "loss": 1.0,
        "delivery_error": "no_accepted_native_bundle",
        "questions": {
            key: {
                **value,
                "valid": False,
                "correct": False,
                "loss": 1.0,
                "prediction": None,
                "top_probability": None,
                "entropy": None,
                "margin": None,
            }
            for key, value in result["questions"].items()
        },
    }


def fit_heads(
    train: list[dict], matrix: dict, native: list[str], provenance: dict
) -> dict:
    heads = {}
    for source in native:
        x = [matrix[row["id"]][source]["features"] for row in train]
        heads[source] = {}
        for action in native:
            if action == source:
                continue
            gains = [
                float(
                    terminal_result(
                        matrix[row["id"]][source]["result"],
                        matrix[row["id"]][action]["result"],
                    )["correct"]
                )
                - float(
                    terminal_result(
                        matrix[row["id"]][source]["result"],
                        matrix[row["id"]][source]["result"],
                    )["correct"]
                )
                for row in train
            ]
            heads[source][action] = {
                "weights": ridge(x, gains),
                "training_mean_cost_ms": mean(
                    matrix[row["id"]][action]["policy_cost_ms"] for row in train
                ),
            }
    return {
        "schema_version": "systemone-policy/v1",
        "feature_names": FEATURE_NAMES,
        "heads": heads,
        "stop_value": 0.0,
        "training": {
            **provenance,
            "split": "train",
            "regularization": 1.0,
            "target": "source minus action whole-bundle error; not a quality certificate",
            "delivery_contract": "whole bundle requires top_probability >= 0 for every native answer; failed upgrade retains an already accepted first answer",
            "conditioning": "all public training source groups; source-stage observation only, no accumulated history",
            "train_groups": len({row["group_id"] for row in train}),
        },
    }


def fit_calibrator(points: list[tuple[float, bool]]) -> dict:
    """PAVA on tied scores, fitted only to held-separate calibration labels."""
    if not points:
        raise ValueError("quality calibration requires data")
    tied = {}
    for score, correct in points:
        values = tied.setdefault(score, [0.0, 0])
        values[0] += float(correct)
        values[1] += 1
    blocks = []
    for score, (total, count) in sorted(tied.items()):
        blocks.append({"upper": score, "sum": total, "count": count})
        while (
            len(blocks) > 1
            and blocks[-2]["sum"] / blocks[-2]["count"]
            > blocks[-1]["sum"] / blocks[-1]["count"]
        ):
            right, left = blocks.pop(), blocks.pop()
            blocks.append(
                {
                    "upper": right["upper"],
                    "sum": left["sum"] + right["sum"],
                    "count": left["count"] + right["count"],
                }
            )
    return {
        "method": "pava",
        "feature": "min_top_probability",
        "target": "whole_bundle_correct",
        "split": "calibration",
        "upper_bounds": [block["upper"] for block in blocks],
        "probabilities": [block["sum"] / block["count"] for block in blocks],
        "counts": [block["count"] for block in blocks],
    }


def calibrated_quality(calibrator: dict, score: float) -> float:
    index = min(
        bisect_left(calibrator["upper_bounds"], score),
        len(calibrator["probabilities"]) - 1,
    )
    return calibrator["probabilities"][index]
