"""Paired item bootstrap for two models on the frozen CSS human-label panel.

Items are resampled within each task, with the same sampled IDs applied to
both models. Evaluation headlines reproduce transfer.score's median over its
15 evaluation tasks; the three pilot tasks never enter that headline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .build import EVALUATION_TASKS, PANEL_VERSION, PILOT_TASKS, sha_file
from .score import evaluate, read_jsonl, score

COMPARISON_VERSION = "css-paired-item-bootstrap/1"


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile needs nonempty values and a fraction in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval95(values: list[float]) -> dict[str, float]:
    return {"low": percentile(values, 0.025), "high": percentile(values, 0.975)}


def _coded_task(
    rows: list[dict[str, Any]],
    predictions_a: dict[str, dict[str, Any]],
    predictions_b: dict[str, dict[str, Any]],
) -> tuple[list[int], list[int], list[int], int]:
    labels = rows[0]["labels"]
    if any(row["labels"] != labels for row in rows):
        raise ValueError("Task option labels vary across items")
    index = {label: position for position, label in enumerate(labels)}
    if len(index) != len(labels) or set(row["gold"] for row in rows) != set(labels):
        raise ValueError("Task gold must cover each unique option label")
    gold: list[int] = []
    a: list[int] = []
    b: list[int] = []
    for row in rows:
        gold.append(index[row["gold"]])
        for predictions, target in ((predictions_a, a), (predictions_b, b)):
            result = evaluate(row, predictions.get(row["id"]))
            # -1 is a miss in both accuracy and every class's false negatives.
            target.append(index[result["choice"]] if result["valid"] else -1)
    return gold, a, b, len(labels)


def _sample_pair(
    gold: list[int],
    a: list[int],
    b: list[int],
    nlabels: int,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    gold_count = [0] * nlabels
    a_pred = [0] * nlabels
    b_pred = [0] * nlabels
    a_tp = [0] * nlabels
    b_tp = [0] * nlabels
    a_correct = b_correct = 0
    count = len(gold)
    for _ in range(count):
        item = rng.randrange(count)
        truth, choice_a, choice_b = gold[item], a[item], b[item]
        gold_count[truth] += 1
        if choice_a >= 0:
            a_pred[choice_a] += 1
            if choice_a == truth:
                a_tp[truth] += 1
                a_correct += 1
        if choice_b >= 0:
            b_pred[choice_b] += 1
            if choice_b == truth:
                b_tp[truth] += 1
                b_correct += 1
    a_f1 = (
        sum(
            (
                2 * tp / (gold_count[label] + a_pred[label])
                if gold_count[label] + a_pred[label]
                else 0.0
            )
            for label, tp in enumerate(a_tp)
        )
        / nlabels
    )
    b_f1 = (
        sum(
            (
                2 * tp / (gold_count[label] + b_pred[label])
                if gold_count[label] + b_pred[label]
                else 0.0
            )
            for label, tp in enumerate(b_tp)
        )
        / nlabels
    )
    return a_f1, b_f1, a_correct / count, b_correct / count


def compare(
    gold_path: Path,
    predictions_a_path: Path,
    predictions_b_path: Path,
    *,
    model_a: str,
    model_b: str,
    replicates: int = 5000,
    seed: int = 20260926,
) -> dict[str, Any]:
    if not model_a or not model_b or model_a == model_b:
        raise ValueError("Supply two distinct nonempty model labels")
    if replicates < 100 or type(seed) is not int:
        raise ValueError("Use at least 100 replicates and an integer seed")
    # Reuse the frozen scorer for all gold, input-hash, role and validity checks.
    scored_a = score(gold_path, predictions_a_path)
    scored_b = score(gold_path, predictions_b_path)
    if (
        scored_a["panel_version"] != scored_b["panel_version"]
        or scored_a["tasks"].keys() != scored_b["tasks"].keys()
    ):
        raise ValueError("The two predictions do not cover the same frozen panel")
    tasks = scored_a["tasks"]
    found_evaluation = {
        name for name, result in tasks.items() if result["role"] == "evaluation"
    }
    if found_evaluation != set(EVALUATION_TASKS):
        raise ValueError(
            f"Expected all {len(EVALUATION_TASKS)} frozen evaluation tasks; found {len(found_evaluation)}"
        )
    if any(name not in PILOT_TASKS and name not in EVALUATION_TASKS for name in tasks):
        raise ValueError("Unexpected task in CSS panel")
    gold = read_jsonl(gold_path)
    predictions_a = read_jsonl(predictions_a_path)
    predictions_b = read_jsonl(predictions_b_path)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in tasks}
    for row in gold.values():
        grouped[row["task"]].append(row)
    task_reports: dict[str, dict[str, Any]] = {}
    task_draws: dict[str, dict[str, list[float]]] = {}
    for task in sorted(tasks):
        rows = sorted(grouped[task], key=lambda row: row["id"])
        coded = _coded_task(rows, predictions_a, predictions_b)
        # Stable task-specific streams make the result invariant to input row order
        # and to the presence of optional pilot tasks.
        task_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{task}".encode()).digest(), "big"
        )
        rng = random.Random(task_seed)
        draws = {"a_f1": [], "b_f1": [], "a_accuracy": [], "b_accuracy": []}
        for _ in range(replicates):
            a_f1, b_f1, a_accuracy, b_accuracy = _sample_pair(*coded, rng)
            draws["a_f1"].append(a_f1)
            draws["b_f1"].append(b_f1)
            draws["a_accuracy"].append(a_accuracy)
            draws["b_accuracy"].append(b_accuracy)
        task_draws[task] = draws
        a_point, b_point = scored_a["tasks"][task], scored_b["tasks"][task]
        f1_differences = [a - b for a, b in zip(draws["a_f1"], draws["b_f1"])]
        accuracy_differences = [
            a - b for a, b in zip(draws["a_accuracy"], draws["b_accuracy"])
        ]
        task_reports[task] = {
            "role": a_point["role"],
            "n": a_point["n"],
            "invalid_or_missing_a": a_point["invalid_or_missing_n"],
            "invalid_or_missing_b": b_point["invalid_or_missing_n"],
            "macro_f1_all": {
                "a": a_point["macro_f1_all"],
                "b": b_point["macro_f1_all"],
                "difference_a_minus_b": a_point["macro_f1_all"]
                - b_point["macro_f1_all"],
                "difference_interval95": interval95(f1_differences),
            },
            "accuracy_all": {
                "a": a_point["accuracy_all"],
                "b": b_point["accuracy_all"],
                "difference_a_minus_b": a_point["accuracy_all"]
                - b_point["accuracy_all"],
                "difference_interval95": interval95(accuracy_differences),
            },
        }

    evaluation_names = sorted(EVALUATION_TASKS)
    headline: dict[str, Any] = {"task_count": len(evaluation_names)}
    for metric, draw_a, draw_b in (
        ("macro_f1_all", "a_f1", "b_f1"),
        ("accuracy_all", "a_accuracy", "b_accuracy"),
    ):
        point_a = statistics.median(
            scored_a["tasks"][task][metric] for task in evaluation_names
        )
        point_b = statistics.median(
            scored_b["tasks"][task][metric] for task in evaluation_names
        )
        bootstrap_a = [
            statistics.median(
                task_draws[task][draw_a][replicate] for task in evaluation_names
            )
            for replicate in range(replicates)
        ]
        bootstrap_b = [
            statistics.median(
                task_draws[task][draw_b][replicate] for task in evaluation_names
            )
            for replicate in range(replicates)
        ]
        headline[metric] = {
            "median_a": point_a,
            "median_b": point_b,
            "difference_a_minus_b": point_a - point_b,
            "median_a_interval95": interval95(bootstrap_a),
            "median_b_interval95": interval95(bootstrap_b),
            "difference_interval95": interval95(
                [a - b for a, b in zip(bootstrap_a, bootstrap_b)]
            ),
        }
    return {
        "comparison_version": COMPARISON_VERSION,
        "panel_version": PANEL_VERSION,
        "gold_sha256": sha_file(gold_path),
        "predictions_a_sha256": sha_file(predictions_a_path),
        "predictions_b_sha256": sha_file(predictions_b_path),
        "compare_code_sha256": sha_file(Path(__file__)),
        "score_code_sha256": sha_file(Path(__file__).with_name("score.py")),
        "model_a": model_a,
        "model_b": model_b,
        "bootstrap": {
            "method": "paired item percentile bootstrap within each task; independent task streams",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "unit": "item ID; the same draw is applied to both models",
            "fixed_label_universe": "all original task option labels; unsampled classes score F1=0",
            "invalid_policy": "missing or invalid predictions are misses (choice=None)",
        },
        "evaluation_median_over_15_tasks": headline,
        "tasks": task_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(
        args.gold,
        args.predictions_a,
        args.predictions_b,
        model_a=args.model_a,
        model_b=args.model_b,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluation_median_over_15_tasks": report[
                    "evaluation_median_over_15_tasks"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
