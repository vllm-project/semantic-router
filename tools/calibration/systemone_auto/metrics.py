"""Typed losses and observable features; no invented probabilities for Chat."""

from __future__ import annotations

import math
from statistics import mean

from .artifacts import canonical, finite

NORMALIZATION_TOLERANCE = 1e-4
NOUL_DECISION_THRESHOLD = 0.5

FEATURE_NAMES = [
    "bias",
    "min_top_probability",
    "mean_top_probability",
    "max_entropy",
    "min_margin",
    "log_state_bytes",
    "log_questions",
    "fraction_choice",
    "fraction_noul",
    "fraction_score",
    "invalid_response",
]


def _probabilities(
    answer: dict, question: dict, *, native: bool
) -> dict[str, float] | None:
    kind = question["type"]
    if kind == "noul":
        value = answer.get("noul")
        if not native and type(value) is bool:
            return None
        probability = finite(value)
        if probability > 1:
            raise ValueError("Noul probability exceeds one")
        return {"false": 1 - probability, "true": probability}
    if not native:
        return None
    value = answer.get("probabilities")
    keys = (
        list(question["criteria"])
        if kind == "choice"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    if not isinstance(value, dict) or set(value) != set(keys):
        return None
    try:
        result = {key: finite(value[key]) for key in keys}
    except ValueError:
        return None
    if (
        any(p > 1 for p in result.values())
        or abs(sum(result.values()) - 1) > NORMALIZATION_TOLERANCE
    ):
        return None
    return result


def _question_result(answer: dict, question: dict, gold: dict, *, native: bool) -> dict:
    kind = question["type"]
    if answer.get("type") != kind or answer.get("error"):
        raise ValueError("missing, wrong-type or errored answer")
    if (
        native
        and question.get("require_full_input")
        and answer.get("input_coverage") != "complete"
    ):
        raise ValueError("missing complete-input coverage proof")
    probabilities = _probabilities(answer, question, native=native)
    if kind == "choice":
        prediction = answer.get("choice")
        if prediction not in question["criteria"]:
            raise ValueError("Choice is outside the declared options")
        loss = float(prediction != gold["label"])
    elif kind == "noul":
        value = answer.get("noul")
        if not native and type(value) is bool:
            prediction = str(value).lower()
        else:
            prediction = str(finite(value) >= NOUL_DECISION_THRESHOLD).lower()
        loss = float(prediction != gold["label"])
    elif kind == "score":
        score = finite(answer.get("score"))
        maximum = len(question["criteria"]) - 1
        if score > maximum:
            raise ValueError("Score outside declared levels")
        prediction = str(math.floor(score + 0.5))
        loss = abs(score - int(gold["label"])) / maximum
    else:
        raise ValueError("pilot supports Choice, Noul and Score only")
    result = {
        "valid": True,
        "type": kind,
        "prediction": prediction,
        "target": gold["label"],
        "correct": prediction == gold["label"],
        "loss": loss,
        "brier": None,
        "nll": None,
        "top_probability": None,
        "entropy": None,
        "margin": None,
        "probability_correct": None,
    }
    if probabilities is not None:
        ordered = sorted(probabilities.values(), reverse=True)
        entropy = -sum(p * math.log(p) for p in ordered if p > 0) / math.log(
            len(ordered)
        )
        target = gold.get("distribution") or {
            key: float(key == gold["label"]) for key in probabilities
        }
        if set(target) != set(probabilities):
            raise ValueError("gold distribution differs from declared options")
        # Probability calibration uses the modal answer, not a rounded expected Score.
        modal = max(probabilities, key=probabilities.get)
        result.update(
            brier=sum((probabilities[key] - target[key]) ** 2 for key in probabilities),
            nll=-math.log(max(probabilities[gold["label"]], 1e-15)),
            top_probability=ordered[0],
            entropy=entropy,
            margin=ordered[0] - ordered[1],
            probability_correct=modal == gold["label"],
        )
    return result


def evaluate_response(row: dict, response: dict, *, native: bool = True) -> dict:
    questions = row["request"]["questions"]
    answers = response.get("answers") if isinstance(response, dict) else None
    results = {}
    for key, question in questions.items():
        try:
            answer = answers.get(key) if isinstance(answers, dict) else None
            if not isinstance(answer, dict):
                raise ValueError("missing answer")
            results[key] = _question_result(
                answer, question, row["labels"][key], native=native
            )
        except (ValueError, TypeError, KeyError) as error:
            results[key] = {
                "valid": False,
                "type": question["type"],
                "error": str(error),
                "target": row["labels"][key]["label"],
                "prediction": None,
                "loss": 1.0,
                "correct": False,
                "top_probability": None,
                "entropy": None,
                "margin": None,
            }
    valid = all(item["valid"] for item in results.values())
    return {
        "valid": valid,
        "loss": mean(item["loss"] for item in results.values()) if valid else 1.0,
        "correct": valid and all(item["correct"] for item in results.values()),
        "questions": results,
    }


def features(row: dict, result: dict) -> list[float]:
    """Whole bundle v1; every declared question contributes to the denominator."""
    questions = row["request"]["questions"]
    values = [result["questions"].get(key, {}) for key in questions]
    valid = [
        item.get("valid", False) and item.get("top_probability") is not None
        for item in values
    ]
    top = [
        item["top_probability"] if ok else 0.0
        for item, ok in zip(values, valid, strict=True)
    ]
    entropy = [
        item["entropy"] if ok else 1.0 for item, ok in zip(values, valid, strict=True)
    ]
    margin = [
        item["margin"] if ok else 0.0 for item, ok in zip(values, valid, strict=True)
    ]
    return [
        1.0,
        min(top),
        mean(top),
        max(entropy),
        min(margin),
        math.log1p(len(canonical(row["request"]["state"]).encode())),
        math.log1p(len(questions)),
        *(
            sum(q["type"] == kind for q in questions.values()) / len(questions)
            for kind in ("choice", "noul", "score")
        ),
        float(not all(valid)),
    ]


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}
    per_type = {}
    for kind in ("choice", "noul", "score"):
        values = [
            q for row in results for q in row["questions"].values() if q["type"] == kind
        ]
        if not values:
            continue
        calibrated = [q for q in values if q["top_probability"] is not None]
        ece = 0.0
        for index in range(10):
            bucket = [
                q for q in calibrated if min(int(q["top_probability"] * 10), 9) == index
            ]
            if bucket:
                ece += (
                    len(bucket)
                    * abs(
                        mean(q["top_probability"] for q in bucket)
                        - mean(q["probability_correct"] for q in bucket)
                    )
                    / len(calibrated)
                )
        per_type[kind] = {
            "questions": len(values),
            "valid_questions": sum(q["valid"] for q in values),
            "accuracy": mean(q["correct"] for q in values),
            "loss": mean(q["loss"] for q in values),
            "probability_count": len(calibrated),
            "ece_10_equal_width": ece if calibrated else None,
            "brier": mean(q["brier"] for q in calibrated) if calibrated else None,
            "nll": mean(q["nll"] for q in calibrated) if calibrated else None,
        }
    return {
        "count": len(results),
        "invalid_count": sum(not row["valid"] for row in results),
        "bundle_accuracy": mean(row["correct"] for row in results),
        "bundle_error": mean(not row["correct"] for row in results),
        "mean_typed_loss": mean(row["loss"] for row in results),
        "per_type": per_type,
    }
