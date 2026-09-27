"""Paired bootstrap for the three open CSS pilot tasks only.

The sealed 15-task evaluation remains inaccessible. This report is diagnostic
and must not be cited as JevArena v3 release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from transfer.build import PILOT_TASKS, sha_file
from transfer.compare import _coded_task, _sample_pair, interval95
from transfer.score import read_jsonl, score


def compare_pilot(
    *,
    gold: Path,
    source: Path,
    treatment: Path,
    output: Path,
    replicates: int = 5000,
    seed: int = 20260927,
) -> dict:
    if output.exists() or replicates < 100:
        raise ValueError("Output exists or bootstrap budget invalid")
    left = score(gold, source)
    right = score(gold, treatment)
    if set(left["tasks"]) != set(PILOT_TASKS) or set(right["tasks"]) != set(
        PILOT_TASKS
    ):
        raise ValueError("Expected exactly three open pilot tasks")
    gold_rows = read_jsonl(gold)
    source_rows = read_jsonl(source)
    treatment_rows = read_jsonl(treatment)
    names = sorted(PILOT_TASKS)
    point_left = statistics.median(
        left["tasks"][name]["macro_f1_all"] for name in names
    )
    point_right = statistics.median(
        right["tasks"][name]["macro_f1_all"] for name in names
    )
    task_draws = {}
    task_results = {}
    for name in names:
        rows = sorted(
            (row for row in gold_rows.values() if row["task"] == name),
            key=lambda row: row["id"],
        )
        coded = _coded_task(rows, source_rows, treatment_rows)
        task_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{name}".encode()).digest(), "big"
        )
        rng = random.Random(task_seed)
        draws = [_sample_pair(*coded, rng) for _ in range(replicates)]
        task_draws[name] = draws
        differences = [a - b for a, b, _, _ in draws]
        task_results[name] = {
            "n": len(rows),
            "source_macro_f1": left["tasks"][name]["macro_f1_all"],
            "treatment_macro_f1": right["tasks"][name]["macro_f1_all"],
            "source_minus_treatment_ci95": interval95(differences),
            "source_invalid": left["tasks"][name]["invalid_or_missing_n"],
            "treatment_invalid": right["tasks"][name]["invalid_or_missing_n"],
        }
    median_differences = [
        statistics.median(task_draws[name][index][0] for name in names)
        - statistics.median(task_draws[name][index][1] for name in names)
        for index in range(replicates)
    ]
    result = {
        "protocol": "joyfox08-open-css-pilot-paired-bootstrap-v1",
        "role": "pilot_only_not_release",
        "gold_sha256": sha_file(gold),
        "source_predictions_sha256": sha_file(source),
        "treatment_predictions_sha256": sha_file(treatment),
        "compare_code_sha256": sha_file(Path(__file__)),
        "replicates": replicates,
        "seed": seed,
        "task_count": 3,
        "source_median_macro_f1": point_left,
        "treatment_median_macro_f1": point_right,
        "source_minus_treatment_ci95": interval95(median_differences),
        "tasks": task_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "source", "treatment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    print(json.dumps(compare_pilot(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
