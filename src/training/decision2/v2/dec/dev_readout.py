"""Development readout: typed DEV ``T``, CSS pilot ``H``, proxy and paired intervals.

Per-answer correctness comes from the shared scorers' own answer functions
(``benchmark.score.evaluate_answer`` and ``transfer.score.evaluate``), so point
values match their reports. ``T`` is the typed family-macro accuracy, ``H`` the
median CSS task macro-F1 and the proxy ``100 * sqrt(T * H)``. Paired
bootstrap resamples typed generator groups within family and CSS items within
task, identically for both compared arms.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer
from transfer.score import evaluate as css_evaluate
from transfer.score import macro_f1

DRAWS = 5000
SEED = 20260928


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def typed_answers(gold_path: Path, pred_path: Path) -> list[dict[str, Any]]:
    gold, predictions = read_jsonl(gold_path), read_jsonl(pred_path)
    records = []
    for item_id, item in gold.items():
        answers = (predictions.get(item_id) or {}).get("answers") or {}
        for question_id, question in item["questions"].items():
            result = (
                evaluate_answer(
                    question, item["gold"][question_id], answers[question_id]
                )
                if question_id in answers
                else {"status": "missing", "correct": False}
            )
            records.append(
                {
                    "group": item["group_id"],
                    "family": item["family"],
                    "type": question["type"],
                    "correct": bool(result.get("correct"))
                    and result.get("status") == "ok",
                    "valid": result.get("status") == "ok",
                    "point": result.get("point"),
                    "gold": item["gold"][question_id]["value"],
                }
            )
    return records


def css_answers(gold_path: Path, pred_path: Path) -> dict[str, dict[str, Any]]:
    gold, predictions = read_jsonl(gold_path), read_jsonl(pred_path)
    tasks: dict[str, dict[str, Any]] = {}
    for item_id, row in gold.items():
        result = css_evaluate(row, predictions.get(item_id))
        task = tasks.setdefault(
            row["task"], {"labels": row["labels"], "gold": [], "choice": [], "valid": 0}
        )
        task["gold"].append(row["gold"])
        task["choice"].append(result["choice"] if result["valid"] else None)
        task["valid"] += int(result["valid"])
    return tasks


def typed_T(
    records: list[dict[str, Any]], weights: dict[str, int] | None = None
) -> float:
    by_family: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        w = 1 if weights is None else weights.get(record["group"], 0)
        by_family[record["family"]][0] += w * record["correct"]
        by_family[record["family"]][1] += w
    return statistics.mean(c / n for c, n in by_family.values() if n)


def css_H(
    tasks: dict[str, dict[str, Any]], indices: dict[str, list[int]] | None = None
) -> float:
    scores = []
    for name, task in sorted(tasks.items()):
        idx = indices[name] if indices is not None else range(len(task["gold"]))
        scores.append(
            macro_f1(
                [task["gold"][i] for i in idx],
                [task["choice"][i] for i in idx],
                task["labels"],
            )
        )
    return statistics.median(scores)


def summarize(
    records: list[dict[str, Any]], tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    T, H = typed_T(records), css_H(tasks)
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    score_points: Counter = Counter()
    score_gold: Counter = Counter()
    for record in records:
        by_type[record["type"]]["n"] += 1
        by_type[record["type"]]["correct"] += record["correct"]
        by_type[record["type"]]["invalid"] += not record["valid"]
        by_family[record["family"]]["n"] += 1
        by_family[record["family"]]["correct"] += record["correct"]
        if record["type"] == "score":
            score_points[str(record["point"])] += 1
            score_gold[str(record["gold"])] += 1
    return {
        "T": T,
        "H": H,
        "proxy": 100 * math.sqrt(T * H),
        "typed_correct": sum(r["correct"] for r in records),
        "typed_answers": len(records),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "score_predicted_levels": dict(sorted(score_points.items())),
        "score_gold_levels": dict(sorted(score_gold.items())),
        "css_task_macro_f1": {
            name: macro_f1(task["gold"], task["choice"], task["labels"])
            for name, task in sorted(tasks.items())
        },
        "css_valid": {name: task["valid"] for name, task in sorted(tasks.items())},
        "css_items": {name: len(task["gold"]) for name, task in sorted(tasks.items())},
    }


def paired_bootstrap(
    a: tuple[list[dict[str, Any]], dict[str, Any]],
    b: tuple[list[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Interval for b minus a on T, H and proxy."""
    rng = random.Random(SEED)
    groups_by_family: dict[str, list[str]] = defaultdict(list)
    for record in a[0]:
        if record["group"] not in groups_by_family[record["family"]]:
            groups_by_family[record["family"]].append(record["group"])
    sizes = {name: len(task["gold"]) for name, task in a[1].items()}
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(DRAWS):
        weights: Counter = Counter()
        for groups in groups_by_family.values():
            for _ in groups:
                weights[rng.choice(groups)] += 1
        indices = {
            name: [rng.randrange(n) for _ in range(n)] for name, n in sizes.items()
        }
        values = []
        for records, tasks in (a, b):
            T, H = typed_T(records, weights), css_H(tasks, indices)
            values.append((T, H, 100 * math.sqrt(T * H)))
        for name, i in (("T", 0), ("H", 1), ("proxy", 2)):
            deltas[name].append(values[1][i] - values[0][i])
    result = {}
    for name, values in deltas.items():
        values.sort()
        result[name] = {
            "lower95": values[int(0.025 * DRAWS)],
            "upper95": values[int(0.975 * DRAWS) - 1],
            "positive_fraction": sum(v > 0 for v in values) / DRAWS,
        }
    return {"draws": DRAWS, "seed": SEED, "delta_b_minus_a": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-gold", type=Path, required=True)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name=typed_predictions,css_predictions",
    )
    parser.add_argument(
        "--compare", action="append", default=[], help="a:b computes b minus a"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        typed_path, css_path = paths.split(",")
        arms[name] = (
            typed_answers(args.typed_gold, Path(typed_path)),
            css_answers(args.css_gold, Path(css_path)),
        )
    report = {
        "schema_version": "dec-dev-readout/1",
        "role": "development readout (typed DEV + CSS pilot); not a release or post-key score",
        "arms": {name: summarize(*value) for name, value in arms.items()},
        "comparisons": {},
    }
    for spec in args.compare:
        left, right = spec.split(":")
        report["comparisons"][f"{right}-minus-{left}"] = paired_bootstrap(
            arms[left], arms[right]
        )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, value in report["arms"].items():
        print(
            name,
            round(value["T"], 6),
            round(value["H"], 6),
            round(value["proxy"], 4),
            value["by_type"],
            value["score_predicted_levels"],
        )
    for name, value in report["comparisons"].items():
        print(name, json.dumps(value["delta_b_minus_a"]))


if __name__ == "__main__":
    main()
