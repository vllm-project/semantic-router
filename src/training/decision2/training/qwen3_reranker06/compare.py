"""Group-paired typed DEV contrast for the fixed 0.6B reranker pilot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer, load_jsonl, validate_suite

from training.model.data import file_sha256

SEED = 20260927
REPEATS = 10_000


def bootstrap(groups: dict[str, list[float]], repeats: int = REPEATS) -> dict[str, Any]:
    if not groups or any(not scores for scores in groups.values()):
        raise ValueError("Every family requires independent groups")
    rng = random.Random(SEED)
    by_family = {
        name: statistics.mean(values) for name, values in sorted(groups.items())
    }
    draws = []
    for _ in range(repeats):
        family_means = []
        for values in groups.values():
            family_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        draws.append(statistics.mean(family_means))
    draws.sort()
    return {
        "family_macro_delta": statistics.mean(by_family.values()),
        "ci95": [draws[int(0.025 * repeats)], draws[int(0.975 * repeats) - 1]],
        "by_family_delta": by_family,
        "independent_groups": sum(map(len, groups.values())),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": SEED,
    }


def compare(gold: Path, source: Path, candidate: Path) -> dict[str, Any]:
    items = load_jsonl(gold)
    split, _ = validate_suite(items)
    if split != "dev" or len(items) != 1600:
        raise ValueError("Expected exact typed DEV1600")
    left = load_jsonl(source)
    right = load_jsonl(candidate)
    if set(left) != set(items) or set(right) != set(items):
        raise ValueError("Predictions must cover the same 1,600 rows")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wins = losses = 0
    for item_id, item in items.items():
        correct = []
        for predictions in (left, right):
            record = predictions[item_id]
            if (
                record.get("source_input_sha256")
                != item["provenance"]["payload_sha256"]
            ):
                raise ValueError("Prediction prompt identity differs")
            answers = record.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(item["questions"]):
                raise ValueError("Question set differs")
            result = []
            for key, question in item["questions"].items():
                scored = evaluate_answer(question, item["gold"][key], answers[key])
                result.append(
                    int(scored.get("status") == "ok" and scored.get("correct"))
                )
            correct.append(statistics.mean(result))
        wins += int(correct[1] > correct[0])
        losses += int(correct[1] < correct[0])
        grouped[item["family"]][item["group_id"]].append(correct[1] - correct[0])
    family_groups = {
        family: [statistics.mean(rows) for rows in groups.values()]
        for family, groups in grouped.items()
    }
    return {
        "gold_sha256": file_sha256(gold),
        "source_prediction_sha256": file_sha256(source),
        "candidate_prediction_sha256": file_sha256(candidate),
        "rows": len(items),
        "candidate_wins": wins,
        "candidate_losses": losses,
        **bootstrap(family_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "source", "candidate", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.gold, args.source, args.candidate)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
