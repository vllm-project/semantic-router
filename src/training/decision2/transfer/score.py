"""Score native typed-choice predictions on the CSS human-label transfer panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build import EVALUATION_TASKS, PANEL_VERSION, PILOT_TASKS, sha_file


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            item_id = row.get("id")
            if not isinstance(item_id, str) or item_id in rows:
                raise ValueError(f"{path}:{line_number}: missing or duplicate id")
            rows[item_id] = row
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def finite_probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def evaluate(gold: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    if prediction is None:
        return {"valid": False, "reason": "missing_prediction", "choice": None}
    answers = prediction.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"label"}:
        return {"valid": False, "reason": "answer_map", "choice": None}
    answer = answers["label"]
    if not isinstance(answer, dict) or answer.get("type") not in (None, "choice"):
        return {"valid": False, "reason": "answer_type", "choice": None}
    labels = gold["labels"]
    choice, probs = answer.get("choice"), answer.get("probabilities")
    if not isinstance(choice, str) or choice not in labels:
        return {"valid": False, "reason": "choice", "choice": None}
    if (
        not isinstance(probs, dict)
        or set(probs) != set(labels)
        or any(not finite_probability(value) for value in probs.values())
        or abs(sum(probs.values()) - 1.0) > 0.02
    ):
        return {"valid": False, "reason": "probabilities", "choice": None}
    if probs[choice] < max(probs.values()) - 0.02:
        return {"valid": False, "reason": "choice_probability_conflict", "choice": None}
    original_sum = sum(probs.values())
    normalized = {label: probs[label] / original_sum for label in labels}
    brier_sum = sum(
        (normalized[label] - (label == gold["gold"])) ** 2 for label in labels
    )
    confidence = answer.get("confidence")
    if confidence is not None and not finite_probability(confidence):
        confidence = None
    return {
        "valid": True,
        "choice": choice,
        "correct": choice == gold["gold"],
        "pmax": max(normalized.values()),
        "native_confidence": confidence,
        "brier_sum": brier_sum,
        "gold_probability": normalized[gold["gold"]],
        "option_sum_abs_delta": abs(original_sum - 1.0),
    }


def macro_f1(golds: list[str], choices: list[str | None], labels: list[str]) -> float:
    if not golds:
        raise ValueError("macro-F1 requires nonempty gold")
    scores = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(golds, choices))
        false_positive = sum(g != label and p == label for g, p in zip(golds, choices))
        false_negative = sum(g == label and p != label for g, p in zip(golds, choices))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.mean(scores)


def ece_15(rows: list[dict[str, Any]], confidence_key: str) -> float | None:
    eligible = [row for row in rows if row["valid"] and row[confidence_key] is not None]
    if not eligible:
        return None
    bins: list[list[dict[str, Any]]] = [[] for _ in range(15)]
    for row in eligible:
        bins[min(int(row[confidence_key] * 15), 14)].append(row)
    return sum(
        len(bucket)
        / len(eligible)
        * abs(
            statistics.mean(float(row["correct"]) for row in bucket)
            - statistics.mean(row[confidence_key] for row in bucket)
        )
        for bucket in bins
        if bucket
    )


def option_sum_diagnostics(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    deltas = sorted(row["option_sum_abs_delta"] for row in rows if row["valid"])

    def quantile(fraction: float) -> float | None:
        if not deltas:
            return None
        position = (len(deltas) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return deltas[low] + (deltas[high] - deltas[low]) * (position - low)

    return {
        "n": len(deltas),
        "mean": statistics.mean(deltas) if deltas else None,
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": deltas[-1] if deltas else None,
        "over_0_001_n": sum(delta > 0.001 for delta in deltas),
        "over_0_01_n": sum(delta > 0.01 for delta in deltas),
    }


def task_metrics(
    gold_rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(gold_rows) != len(results):
        raise ValueError("gold/results length mismatch")
    labels = gold_rows[0]["labels"]
    if any(row["labels"] != labels for row in gold_rows):
        raise ValueError("task label set changes across rows")
    if set(row["gold"] for row in gold_rows) != set(labels):
        raise ValueError("task labels absent from gold split")
    golds = [row["gold"] for row in gold_rows]
    choices = [row["choice"] for row in results]
    valid = [row for row in results if row["valid"]]
    valid_golds = [g for g, row in zip(golds, results) if row["valid"]]
    return {
        "n": len(results),
        "valid_n": len(valid),
        "invalid_or_missing_n": len(results) - len(valid),
        "invalid_reasons": dict(
            sorted(
                Counter(row["reason"] for row in results if not row["valid"]).items()
            )
        ),
        "correct_n": sum(row.get("correct", False) for row in results),
        "accuracy_all": sum(row.get("correct", False) for row in results)
        / len(results),
        "macro_f1_all": macro_f1(golds, choices, labels),
        "accuracy_valid": (
            statistics.mean(float(row["correct"]) for row in valid) if valid else None
        ),
        "macro_f1_valid": (
            macro_f1(valid_golds, [row["choice"] for row in valid], labels)
            if valid
            else None
        ),
        "option_probability_sum_abs_delta": option_sum_diagnostics(results),
        "brier_sum": (
            statistics.mean(row["brier_sum"] for row in valid) if valid else None
        ),
        "nll": (
            statistics.mean(
                -math.log(max(row["gold_probability"], 1e-12)) for row in valid
            )
            if valid
            else None
        ),
        "ece_pmax_15": ece_15(results, "pmax"),
        "ece_native_confidence_15": ece_15(results, "native_confidence"),
        "native_confidence_n": sum(
            row["native_confidence"] is not None for row in valid
        ),
    }


def score(gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    gold = read_jsonl(gold_path)
    predictions = read_jsonl(predictions_path)
    if set(predictions) - set(gold):
        raise ValueError("prediction file has IDs outside the gold panel")
    by_task_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_id, gold_row in gold.items():
        if gold_row.get("panel_version") != PANEL_VERSION:
            raise ValueError(f"{item_id}: wrong panel version")
        task, role = gold_row["task"], gold_row["role"]
        expected_role = (
            "pilot"
            if task in PILOT_TASKS
            else "evaluation" if task in EVALUATION_TASKS else None
        )
        if role != expected_role:
            raise ValueError(f"{item_id}: invalid task/role")
        prediction = predictions.get(item_id)
        if (
            prediction is not None
            and prediction.get("source_input_sha256") != gold_row["input_sha256"]
        ):
            raise ValueError(
                f"{item_id}: prediction input hash differs from gold manifest"
            )
        by_task_gold[task].append(gold_row)
        by_task_results[task].append(evaluate(gold_row, prediction))
    tasks = {}
    for task in sorted(by_task_gold):
        role = by_task_gold[task][0]["role"]
        tasks[task] = {
            "role": role,
            **task_metrics(by_task_gold[task], by_task_results[task]),
        }
    roles = {}
    for role in ("pilot", "evaluation"):
        names = [task for task, value in tasks.items() if value["role"] == role]
        subset = [tasks[task] for task in names]
        if not subset:
            continue
        role_results = [row for task in names for row in by_task_results[task]]
        roles[role] = {
            "tasks": len(subset),
            "items": sum(value["n"] for value in subset),
            "valid_items": sum(value["valid_n"] for value in subset),
            "option_probability_sum_abs_delta": option_sum_diagnostics(role_results),
            "micro_accuracy_all": sum(value["correct_n"] for value in subset)
            / sum(value["n"] for value in subset),
            "median_task_macro_f1_all": statistics.median(
                value["macro_f1_all"] for value in subset
            ),
            "median_task_accuracy_all": statistics.median(
                value["accuracy_all"] for value in subset
            ),
            "median_task_brier_sum": (
                statistics.median(
                    value["brier_sum"]
                    for value in subset
                    if value["brier_sum"] is not None
                )
                if any(value["brier_sum"] is not None for value in subset)
                else None
            ),
            "median_task_ece_pmax_15": (
                statistics.median(
                    value["ece_pmax_15"]
                    for value in subset
                    if value["ece_pmax_15"] is not None
                )
                if any(value["ece_pmax_15"] is not None for value in subset)
                else None
            ),
        }
    return {
        "score_schema_version": "css-transfer-score/2",
        "panel_version": PANEL_VERSION,
        "gold_sha256": sha_file(gold_path),
        "predictions_sha256": sha_file(predictions_path),
        "metric_policy": {
            "macro_f1_all": "Unweighted mean of class F1 over gold classes; invalid/missing predictions count as misses.",
            "probability_acceptance": "Finite per-option probabilities in [0,1], original sum within 0.02 of one, and returned choice within 0.02 of the original maximum option probability.",
            "probability_metrics": "For accepted maps, divide each option probability by the original sum before Brier, NLL, and pmax ECE. Native confidence ECE uses the returned scalar unchanged.",
            "option_probability_sum_abs_delta": "Absolute original option-probability sum minus one, for valid answers only.",
            "brier_sum": "Mean normalized multiclass squared error (sum over classes), using the CSS replication scale.",
            "ece_pmax_15": "15 equal-width bins of normalized maximum option probability; comparable across typed models. The paper's headline ECE instead uses each model's native confidence field.",
            "ece_native_confidence_15": "15 equal-width bins of the returned confidence field when available; native field definitions differ among models.",
            "role": "The three pilot tasks are excluded from evaluation summaries and may only be used for predeclared calibration or adapter checks.",
        },
        "roles": roles,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = score(args.gold, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"roles": report["roles"], "output": str(args.output)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
