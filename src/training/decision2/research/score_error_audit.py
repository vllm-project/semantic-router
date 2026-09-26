"""Aggregate frozen typed-DEV errors without exporting private question text.

This is a diagnostic over existing predictions, never a selection or release
score. Missing and invalid answers remain failures. The output contains only
counts, input SHA-256 values and aggregate error slices.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer, load_jsonl, validate_suite


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    gold: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    pairs: dict[str, list[tuple[str, str, str]]],
) -> dict[str, Any]:
    if set(predictions) - set(gold):
        raise ValueError("predictions contain IDs outside the frozen gold panel")
    family = collections.defaultdict(collections.Counter)
    by_gold = collections.defaultdict(collections.Counter)
    score_confusion = collections.defaultdict(collections.Counter)
    answers: dict[str, bool] = {}
    invalid_reasons = collections.Counter()
    for item_id, item in gold.items():
        question = item["questions"]["decision"]
        target = item["gold"]["decision"]
        prediction = predictions.get(item_id, {})
        answer = prediction.get("answers", {}).get("decision")
        result = evaluate_answer(question, target, answer)
        correct = result["status"] == "ok" and bool(result["correct"])
        answers[item_id] = correct
        family[item["family"]]["n"] += 1
        family[item["family"]]["correct"] += int(correct)
        family[item["family"]]["invalid"] += int(result["status"] != "ok")
        by_gold[str(target["value"])]["n"] += 1
        by_gold[str(target["value"])]["correct"] += int(correct)
        if result["status"] != "ok":
            invalid_reasons[result["reason"]] += 1
        if question["type"] == "score":
            predicted = (
                str(result["point"])
                if result["status"] == "ok" and result["point"] is not None
                else "invalid_or_tie"
            )
            score_confusion[str(target["value"])][predicted] += 1
    paired = collections.defaultdict(collections.Counter)
    for members in pairs.values():
        first, second = members
        relation = first[1]
        paired[relation]["n"] += 1
        paired[relation]["both_correct"] += int(
            answers[first[0]] and answers[second[0]]
        )
    groups = collections.defaultdict(list)
    for item_id, item in gold.items():
        groups[item["group_id"]].append(answers[item_id])
    return {
        "n": len(gold),
        "predicted_n": len(predictions),
        "correct": sum(answers.values()),
        "all_four_groups": sum(all(values) for values in groups.values()),
        "group_n": len(groups),
        "family": {key: dict(value) for key, value in sorted(family.items())},
        "gold_level": {key: dict(value) for key, value in sorted(by_gold.items())},
        "score_confusion": {
            key: dict(value) for key, value in sorted(score_confusion.items())
        },
        "pairs": {key: dict(value) for key, value in sorted(paired.items())},
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "row_correct": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", required=True, metavar="NAME=FILE"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = load_jsonl(args.gold)
    split, pairs = validate_suite(gold)
    if split != "dev":
        raise ValueError("this diagnostic is restricted to typed DEV")
    parsed: dict[str, Path] = {}
    for raw in args.prediction:
        name, separator, file = raw.partition("=")
        if not separator or not name or not file or name in parsed:
            raise ValueError("each prediction needs a unique NAME=FILE")
        parsed[name] = Path(file)
    report: dict[str, Any] = {
        "status": "exploratory_dev_diagnostic_not_release_score",
        "gold_sha256": digest(args.gold),
        "models": {},
        "pairwise": {},
    }
    row_correct: dict[str, dict[str, bool]] = {}
    for name, path in parsed.items():
        result = audit(gold, load_jsonl(path), pairs)
        row_correct[name] = result.pop("row_correct")
        result["prediction_sha256"] = digest(path)
        report["models"][name] = result
    names = list(parsed)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            by_family = collections.defaultdict(collections.Counter)
            for item_id, item in gold.items():
                lhs, rhs = row_correct[left][item_id], row_correct[right][item_id]
                counter = by_family[item["family"]]
                counter["n"] += 1
                counter["both"] += int(lhs and rhs)
                counter["left_only"] += int(lhs and not rhs)
                counter["right_only"] += int(rhs and not lhs)
                counter["neither"] += int(not lhs and not rhs)
            report["pairwise"][f"{left}_vs_{right}"] = {
                key: dict(value) for key, value in sorted(by_family.items())
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
