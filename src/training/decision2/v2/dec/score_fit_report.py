"""Score-only overfit verdict from ``eval_rows`` predictions on the fitted rows.

PASS needs overall accuracy >= 0.95, accuracy >= 0.90 within every level count,
and every gold level of every level count predicted at least once. The report
also gives the gold x predicted confusion per level count.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MIN_OVERALL = 0.95
MIN_PER_LEVEL_COUNT = 0.90


def fit_report(
    rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    confusion: dict[int, Counter] = defaultdict(Counter)
    correct: Counter = Counter()
    total: Counter = Counter()
    for row in rows:
        count = len(row["options"])
        gold = int(row["options"][row["label"]]["key"])
        record = predictions[row["id"]]
        predicted = record["prediction_key"]
        predicted = int(predicted) if predicted is not None else -1
        confusion[count][(gold, predicted)] += 1
        total[count] += 1
        correct[count] += int(predicted == gold)
    by_count = {}
    missing_levels = {}
    for count in sorted(total):
        predicted_levels = {p for (_, p) in confusion[count] if p >= 0}
        gold_levels = {g for (g, _) in confusion[count]}
        missing = sorted(gold_levels - predicted_levels)
        if missing:
            missing_levels[f"L{count}"] = missing
        by_count[f"L{count}"] = {
            "n": total[count],
            "accuracy": correct[count] / total[count],
            "predicted_levels": sorted(predicted_levels),
            "confusion": {
                f"{g}->{p}": n for (g, p), n in sorted(confusion[count].items())
            },
        }
    overall = sum(correct.values()) / sum(total.values())
    passed = (
        overall >= MIN_OVERALL
        and all(v["accuracy"] >= MIN_PER_LEVEL_COUNT for v in by_count.values())
        and not missing_levels
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "criteria": {
            "min_overall": MIN_OVERALL,
            "min_per_level_count": MIN_PER_LEVEL_COUNT,
            "every_gold_level_predicted": True,
        },
        "n": sum(total.values()),
        "overall_accuracy": overall,
        "gold_levels_never_predicted": missing_levels,
        "by_level_count": by_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8")]
    predictions = {
        record["id"]: record
        for record in map(json.loads, args.predictions.open(encoding="utf-8"))
    }
    report = fit_report(rows, predictions)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "n", "overall_accuracy")}))


if __name__ == "__main__":
    main()
