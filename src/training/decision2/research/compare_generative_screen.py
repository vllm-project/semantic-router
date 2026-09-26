"""Compare a fixed sparse generative screen using the standard per-answer rule.

The full typed benchmark scorer intentionally rejects incomplete four-variant
groups. This exploratory screen samples individual items, so it reuses only
``evaluate_answer`` and does not claim suite or pair metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmark.score import evaluate_answer, load_jsonl
from inference.run import file_digest


def compare(gold_path: Path, head_path: Path, source_path: Path) -> dict:
    gold = load_jsonl(gold_path)
    head = load_jsonl(head_path)
    source = load_jsonl(source_path)
    if set(gold) != set(head) or set(gold) != set(source):
        raise ValueError("Gold and both prediction sets must have identical IDs")

    by_type: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "head_correct": 0,
            "source_correct": 0,
            "head_invalid": 0,
            "source_invalid": 0,
            "head_only": 0,
            "source_only": 0,
        }
    )
    disagreements = []
    groups = set()
    for item_id, item in gold.items():
        questions = item.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError(f"{item_id}: expected exactly one question")
        key, question = next(iter(questions.items()))
        expected_hash = item["provenance"]["payload_sha256"]
        results = {}
        for name, predictions in (("head", head), ("source", source)):
            prediction = predictions[item_id]
            if prediction.get("source_input_sha256") != expected_hash:
                raise ValueError(f"{item_id}: {name} input hash differs")
            answers = prediction.get("answers")
            answer = (
                answers.get(key)
                if isinstance(answers, dict) and set(answers) == {key}
                else None
            )
            results[name] = evaluate_answer(question, item["gold"][key], answer)
        kind = question["type"]
        if kind not in ("choice", "noul", "score"):
            raise ValueError(f"{item_id}: unsupported question type")
        counts = by_type[kind]
        counts["n"] += 1
        groups.add(item["group_id"])
        correct = {}
        for name, result in results.items():
            correct[name] = result["status"] == "ok" and bool(result["correct"])
            counts[f"{name}_correct"] += int(correct[name])
            counts[f"{name}_invalid"] += int(result["status"] != "ok")
        if correct["head"] != correct["source"]:
            winner = "head" if correct["head"] else "source"
            counts[f"{winner}_only"] += 1
            disagreements.append({"id": item_id, "type": kind, "winner": winner})

    for kind in ("choice", "noul", "score"):
        counts = by_type[kind]
        counts["gain"] = counts["source_correct"] - counts["head_correct"]
    gate = (by_type["score"]["gain"] >= 5 or by_type["noul"]["gain"] >= 5) and by_type[
        "choice"
    ]["gain"] >= -5
    return {
        "protocol": "decision2-qwen38-generative-sparse-screen/1",
        "metric": "benchmark.score.evaluate_answer point correctness; invalid/missing wrong",
        "sha256": {
            "gold": file_digest(gold_path),
            "head_predictions": file_digest(head_path),
            "source_predictions": file_digest(source_path),
        },
        "items": len(gold),
        "independent_groups": len(groups),
        "by_type": dict(sorted(by_type.items())),
        "gate_for_full_dev": gate,
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.gold, args.head, args.source)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, sort_keys=True, indent=2)
        target.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "disagreements"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
