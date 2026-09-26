"""Paired group bootstrap for two models on the same frozen benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .score import DataError, evaluate_answer, load_jsonl, percentile, validate_suite


def outcome(item: dict[str, Any], prediction: dict[str, Any] | None) -> float:
    answers = prediction.get("answers") if prediction is not None else None
    if not isinstance(answers, dict) or answers.keys() != item["questions"].keys():
        return 0.0
    return statistics.mean(
        float(
            evaluate_answer(question, item["gold"][key], answers[key]).get(
                "correct", False
            )
        )
        for key, question in item["questions"].items()
    )


def score_groups(
    items: dict[str, dict[str, Any]],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    if set(left) - set(items) or set(right) - set(items):
        raise DataError("Predictions contain unknown benchmark IDs")
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for item_id, item in items.items():
        grouped[(item["family"], item["group_id"])].append(
            (outcome(item, left.get(item_id)), outcome(item, right.get(item_id)))
        )
    by_family: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (family, _group_id), values in sorted(grouped.items()):
        if len(values) != 4:
            raise DataError(f"{family}: each group must have four items")
        by_family[family].append(
            tuple(statistics.mean(pair[index] for pair in values) for index in (0, 1))
        )
    return dict(by_family)


def percentile_interval(values: list[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def compare(
    gold_path: Path,
    left_path: Path,
    right_path: Path,
    *,
    left_name: str,
    right_name: str,
    iterations: int = 5000,
    seed: int = 20260926,
) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    items = load_jsonl(gold_path)
    split, _pairs = validate_suite(items)
    left, right = load_jsonl(left_path), load_jsonl(right_path)
    families = score_groups(items, left, right)
    rng = random.Random(seed)
    observed_by_family = {
        family: [statistics.mean(group[index] for group in groups) for index in (0, 1)]
        for family, groups in sorted(families.items())
    }
    observed = [
        statistics.mean(row[index] for row in observed_by_family.values())
        for index in (0, 1)
    ]
    boot = []
    per_family_boot: dict[str, list[list[float]]] = {family: [] for family in families}
    for _ in range(iterations):
        sampled = {}
        for family, groups in families.items():
            chosen = [groups[rng.randrange(len(groups))] for _ in groups]
            row = [
                statistics.mean(group[index] for group in chosen) for index in (0, 1)
            ]
            sampled[family] = row
            per_family_boot[family].append([row[0], row[1], row[0] - row[1]])
        pair = [
            statistics.mean(row[index] for row in sampled.values()) for index in (0, 1)
        ]
        boot.append([pair[0], pair[1], pair[0] - pair[1]])
    return {
        "schema_version": "typed-decision-comparison/1",
        "split": split,
        "models": {"left": left_name, "right": right_name},
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "left_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "right_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        "paired_cluster": "four-variant group, resampled within family",
        "iterations": iterations,
        "seed": seed,
        "groups_by_family": {
            family: len(groups) for family, groups in sorted(families.items())
        },
        "family_macro": {
            "left": observed[0],
            "right": observed[1],
            "delta_left_minus_right": observed[0] - observed[1],
            "left_ci95": percentile_interval([row[0] for row in boot]),
            "right_ci95": percentile_interval([row[1] for row in boot]),
            "delta_ci95": percentile_interval([row[2] for row in boot]),
        },
        "by_family": {
            family: {
                "left": observed_by_family[family][0],
                "right": observed_by_family[family][1],
                "delta_left_minus_right": observed_by_family[family][0]
                - observed_by_family[family][1],
                "delta_ci95": percentile_interval(
                    [row[2] for row in per_family_boot[family]]
                ),
            }
            for family in sorted(families)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(
        args.gold,
        args.left,
        args.right,
        left_name=args.left_name,
        right_name=args.right_name,
        iterations=args.iterations,
        seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
