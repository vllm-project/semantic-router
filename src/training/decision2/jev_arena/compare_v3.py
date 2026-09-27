"""Joint paired bootstrap for the two-axis JevArena v3 sealed core.

The typed axis resamples independent four-variant groups within each family.
The CSS axis resamples evaluation tasks and then their items. Every draw uses
the same sampled units for the two models. This is deliberately separate from
the component comparisons: the reported interval is for the *joint* v3 score.

Only the sealed gold holder should run this command. The JSON report contains
digests and aggregate results, never the gold records or prediction contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from benchmark.score import DataError, evaluate_answer, load_jsonl, score_suite
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, sha_file
from transfer.compare import interval95
from transfer.score import evaluate as evaluate_css
from transfer.score import read_jsonl
from transfer.score import score as score_css

from jev_arena.arena_v3 import SCORER_SOURCE_PATHS

SCHEMA_VERSION = "jevarena-v3-paired-aggregate/1"
DEFAULT_REPLICATES = 5000
DEFAULT_SEED = 20260927


def _typed_groups(
    gold: dict[str, dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[int, int, int]]]:
    """Return paired correct-answer counts and denominators per independent group."""
    groups: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for item_id, item in gold.items():
        n = len(item["questions"])
        counts = [0, 0]
        for side, predictions in enumerate((left, right)):
            prediction = predictions.get(item_id)
            answers = prediction.get("answers") if prediction is not None else None
            if not isinstance(answers, dict) or set(answers) != set(item["questions"]):
                continue
            counts[side] = sum(
                evaluate_answer(question, item["gold"][key], answers[key]).get(
                    "correct", False
                )
                for key, question in item["questions"].items()
            )
        groups[(item["family"], item["group_id"])].append((counts[0], counts[1], n))
    by_family: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for (family, _group_id), items in sorted(groups.items()):
        if len(items) != 4:
            raise DataError(f"{family}: expected four variants per independent group")
        by_family[family].append(tuple(sum(row[i] for row in items) for i in range(3)))
    if set(by_family) != set(FINAL_FAMILIES) or any(
        len(by_family[family]) != 100 for family in FINAL_FAMILIES
    ):
        raise DataError("Typed FINAL requires four families of 100 independent groups")
    return dict(by_family)


def _css_tasks(
    gold: dict[str, dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, tuple[list[tuple[int, int, int]], int]]:
    """Code gold and both native choices with invalid/missing mapped to -1."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold.values():
        if row["role"] == "evaluation":
            grouped[row["task"]].append(row)
    if set(grouped) != set(EVALUATION_TASKS) or sum(map(len, grouped.values())) != 6547:
        raise ValueError(
            "CSS sealed panel must contain 15 tasks and 6547 evaluation items"
        )
    coded = {}
    for task, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["id"])
        labels = rows[0]["labels"]
        if any(row["labels"] != labels for row in rows) or len(set(labels)) != len(
            labels
        ):
            raise ValueError(f"{task}: inconsistent or duplicate option labels")
        index = {label: i for i, label in enumerate(labels)}
        if set(row["gold"] for row in rows) != set(labels):
            raise ValueError(f"{task}: original gold does not cover every task label")
        values = []
        for row in rows:
            choices = []
            for predictions in (left, right):
                result = evaluate_css(row, predictions.get(row["id"]))
                choices.append(index[result["choice"]] if result["valid"] else -1)
            values.append((index[row["gold"]], choices[0], choices[1]))
        coded[task] = values, len(labels)
    return coded


def _task_f1(
    rows: list[tuple[int, int, int]], nlabels: int, rng: random.Random
) -> tuple[float, float]:
    """One paired item draw; preserve the original label universe in macro-F1."""
    gold_count = [0] * nlabels
    predicted_left = [0] * nlabels
    predicted_right = [0] * nlabels
    tp_left = [0] * nlabels
    tp_right = [0] * nlabels
    for _ in range(len(rows)):
        truth, left, right = rows[rng.randrange(len(rows))]
        gold_count[truth] += 1
        if left >= 0:
            predicted_left[left] += 1
            tp_left[truth] += left == truth
        if right >= 0:
            predicted_right[right] += 1
            tp_right[truth] += right == truth
    scores = []
    for predicted, true_positive in (
        (predicted_left, tp_left),
        (predicted_right, tp_right),
    ):
        scores.append(
            sum(
                (
                    2 * true_positive[i] / (gold_count[i] + predicted[i])
                    if gold_count[i] + predicted[i]
                    else 0.0
                )
                for i in range(nlabels)
            )
            / nlabels
        )
    return scores[0], scores[1]


def _point(typed: float, css: float) -> dict[str, float]:
    return {"T": typed, "H": css, "score": 100 * math.sqrt(typed * css)}


def _panel_digest(typed_sha: str, css_sha: str) -> str:
    payload = json.dumps(
        {"typed_gold_sha256": typed_sha, "css_gold_sha256": css_sha},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare(
    typed_gold_path: Path,
    css_gold_path: Path,
    left_typed_path: Path,
    left_css_path: Path,
    right_typed_path: Path,
    right_css_path: Path,
    *,
    left_name: str,
    right_name: str,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if (
        not isinstance(left_name, str)
        or not isinstance(right_name, str)
        or not left_name.strip()
        or not right_name.strip()
        or left_name == right_name
    ):
        raise ValueError("Provide two distinct nonempty model names")
    if type(replicates) is not int or replicates < 5000 or type(seed) is not int:
        raise ValueError("JevArena v3 requires at least 5000 draws and an integer seed")

    # Invoke both canonical scorers first. They enforce source-input hashes,
    # native-output validity and full-denominator metrics on the original panel.
    typed_reports = (
        score_suite(typed_gold_path, left_typed_path, left_name, "paired", "native"),
        score_suite(typed_gold_path, right_typed_path, right_name, "paired", "native"),
    )
    css_reports = (
        score_css(css_gold_path, left_css_path),
        score_css(css_gold_path, right_css_path),
    )
    if any(
        report["split"] != "final" or report["items"] != 1600
        for report in typed_reports
    ):
        raise ValueError("Typed paired comparison requires the 1600-item FINAL panel")
    if any(
        report["panel_version"] != PANEL_VERSION
        or report["roles"].get("evaluation", {}).get("items") != 6547
        or report["roles"]["evaluation"]["tasks"] != 15
        for report in css_reports
    ):
        raise ValueError(
            "CSS paired comparison requires 15 evaluation tasks / 6547 items"
        )

    typed_gold = load_jsonl(typed_gold_path)
    typed_left = load_jsonl(left_typed_path)
    typed_right = load_jsonl(right_typed_path)
    css_gold = read_jsonl(css_gold_path)
    css_left = read_jsonl(left_css_path)
    css_right = read_jsonl(right_css_path)
    families = _typed_groups(typed_gold, typed_left, typed_right)
    tasks = _css_tasks(css_gold, css_left, css_right)
    observed_left = _point(
        typed_reports[0]["macro_family_accuracy"],
        css_reports[0]["roles"]["evaluation"]["median_task_macro_f1_all"],
    )
    observed_right = _point(
        typed_reports[1]["macro_family_accuracy"],
        css_reports[1]["roles"]["evaluation"]["median_task_macro_f1_all"],
    )
    for side, point in enumerate((observed_left, observed_right)):
        reconstructed = statistics.mean(
            sum(group[side] for group in groups) / sum(group[2] for group in groups)
            for groups in families.values()
        )
        if not math.isclose(point["T"], reconstructed, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                "Typed grouped outcomes disagree with the canonical scorer"
            )

    rng = random.Random(seed)
    task_names = tuple(sorted(EVALUATION_TASKS))
    draws: dict[str, list[float]] = {
        "left_T": [],
        "right_T": [],
        "delta_T": [],
        "left_H": [],
        "right_H": [],
        "delta_H": [],
        "left_score": [],
        "right_score": [],
        "delta_score": [],
    }
    for _ in range(replicates):
        sampled_families = []
        for family in FINAL_FAMILIES:
            groups = families[family]
            counts = [0, 0, 0]
            for _group in groups:
                chosen = groups[rng.randrange(len(groups))]
                for side in range(3):
                    counts[side] += chosen[side]
            sampled_families.append((counts[0] / counts[2], counts[1] / counts[2]))
        typed_left_axis = statistics.mean(row[0] for row in sampled_families)
        typed_right_axis = statistics.mean(row[1] for row in sampled_families)
        sampled_tasks = [[], []]
        for _task in task_names:
            name = task_names[rng.randrange(len(task_names))]
            task_rows, nlabels = tasks[name]
            pair = _task_f1(task_rows, nlabels, rng)
            sampled_tasks[0].append(pair[0])
            sampled_tasks[1].append(pair[1])
        left = _point(typed_left_axis, statistics.median(sampled_tasks[0]))
        right = _point(typed_right_axis, statistics.median(sampled_tasks[1]))
        for key in ("T", "H", "score"):
            draws[f"left_{key}"].append(left[key])
            draws[f"right_{key}"].append(right[key])
            draws[f"delta_{key}"].append(left[key] - right[key])

    typed_sha = typed_reports[0]["gold_sha256"]
    css_sha = css_reports[0]["gold_sha256"]
    if (
        typed_sha != typed_reports[1]["gold_sha256"]
        or css_sha != css_reports[1]["gold_sha256"]
    ):
        raise ValueError("Left and right models have different sealed panels")
    return {
        "schema_version": SCHEMA_VERSION,
        "models": {"left": left_name, "right": right_name},
        "panel_sha256": _panel_digest(typed_sha, css_sha),
        "typed_gold_sha256": typed_sha,
        "css_gold_sha256": css_sha,
        "predictions_sha256": {
            "left": {
                "typed": typed_reports[0]["predictions_sha256"],
                "css": css_reports[0]["predictions_sha256"],
            },
            "right": {
                "typed": typed_reports[1]["predictions_sha256"],
                "css": css_reports[1]["predictions_sha256"],
            },
        },
        "source_sha256": {
            name: sha_file(path) for name, path in SCORER_SOURCE_PATHS.items()
        },
        "coverage": {
            "typed_items": 1600,
            "typed_independent_groups": 400,
            "css_evaluation_items": 6547,
            "css_evaluation_tasks": 15,
        },
        "bootstrap": {
            "unit": "paired typed groups within family; CSS tasks then paired items within sampled task",
            "fixed_css_label_universe": True,
            "invalid_or_missing": "counted as incorrect; never removed from draws",
            "confidence_level": 0.95,
        },
        "replicates": replicates,
        "seed": seed,
        "point": {
            "left": observed_left,
            "right": observed_right,
            "delta": {
                key: observed_left[key] - observed_right[key]
                for key in ("T", "H", "score")
            },
        },
        "ci95": interval95(draws["delta_score"]),
        "axis_ci95": {
            key: {
                side: interval95(draws[f"{side}_{key}"])
                for side in ("left", "right", "delta")
            }
            for key in ("T", "H")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "typed-gold",
        "css-gold",
        "left-typed",
        "left-css",
        "right-typed",
        "right-css",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(
        args.typed_gold,
        args.css_gold,
        args.left_typed,
        args.left_css,
        args.right_typed,
        args.right_css,
        left_name=args.left_name,
        right_name=args.right_name,
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
    print(json.dumps({"output": str(args.output), "ci95": report["ci95"]}))


if __name__ == "__main__":
    main()
