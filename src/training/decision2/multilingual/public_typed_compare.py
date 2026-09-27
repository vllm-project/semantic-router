"""Paired, group-aware comparison on the exposed multilingual DEV panel.

This diagnostic is not an independent release evaluation. The available group
IDs are only observable proxies for independent source questions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from inference.run import file_digest

from multilingual import public_typed_dev as panel_api

SEED = 20260927
RESAMPLES = 3000


def _correct(prediction: dict[str, Any], target: dict[str, Any]) -> int:
    native = prediction.get("answers")
    if not isinstance(native, dict) or set(native) != {"decision"}:
        return 0
    return int(panel_api._native_choice(native["decision"], target)[1])


def _section(
    pairs: list[tuple[str, int, int]], *, seed: int, resamples: int
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Empty comparison section")
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for group, candidate, baseline in pairs:
        counts = groups[group]
        counts[0] += 1
        counts[1] += candidate
        counts[2] += baseline
    rows = [groups[key] for key in sorted(groups)]
    n = len(pairs)
    candidate_correct = sum(candidate for _, candidate, _ in pairs)
    baseline_correct = sum(baseline for _, _, baseline in pairs)
    result = {
        "questions": n,
        "observable_groups": len(rows),
        "candidate_correct": candidate_correct,
        "baseline_correct": baseline_correct,
        "candidate_only_correct": sum(a == 1 and b == 0 for _, a, b in pairs),
        "baseline_only_correct": sum(a == 0 and b == 1 for _, a, b in pairs),
        "delta_accuracy": (candidate_correct - baseline_correct) / n,
    }
    if not resamples:
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        chosen = rng.choices(rows, k=len(rows))
        denominator = sum(row[0] for row in chosen)
        samples.append(sum(row[1] - row[2] for row in chosen) / denominator)
    samples.sort()
    result["cluster_bootstrap_95_percentile"] = [
        samples[int(0.025 * resamples)],
        samples[int(0.975 * resamples) - 1],
    ]
    return result


def compare(
    panel: Path,
    candidate_path: Path,
    baseline_path: Path,
    *,
    seed: int = SEED,
    resamples: int = RESAMPLES,
) -> dict[str, Any]:
    if resamples < 0:
        raise ValueError("resamples must be nonnegative")
    # Reuse the exact panel/version/input/model validation of the native scorer.
    candidate_score = panel_api.score(panel, candidate_path)
    baseline_score = panel_api.score(panel, baseline_path)
    targets = panel_api._read_jsonl(panel / "targets.private.jsonl")
    candidate = {r["id"]: r for r in panel_api._read_jsonl(candidate_path)}
    baseline = {r["id"]: r for r in panel_api._read_jsonl(baseline_path)}
    ids = {row["id"] for row in targets}
    if set(candidate) != ids or set(baseline) != ids:
        raise ValueError("Both predictions must cover the entire frozen panel")
    sections: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for target in targets:
        item_id = target["id"]
        pair = (
            target["group_id"],
            _correct(candidate[item_id], target),
            _correct(baseline[item_id], target),
        )
        sections["all"].append(pair)
        sections[f"language/{target['language']}"].append(pair)
        sections[f"language_type/{target['language']}/{target['task_type']}"].append(
            pair
        )
        sections[f"task/{target['task']}"].append(pair)
    result = {
        "schema_version": "decision2-public-multilingual-typed-paired/1",
        "scope": "exposed supplementary DEV only; group IDs are incomplete source proxies",
        "panel_manifest_sha256": candidate_score["manifest_sha256"],
        "candidate_predictions_sha256": file_digest(candidate_path),
        "baseline_predictions_sha256": file_digest(baseline_path),
        "candidate_model": candidate_score["model"],
        "baseline_model": baseline_score["model"],
        "comparison_source_sha256": file_digest(Path(__file__)),
        "bootstrap": {
            "unit": "observable group_id within each section",
            "method": "with-replacement cluster bootstrap; question-weighted delta; percentile interval",
            "seed": seed,
            "resamples": resamples,
            "independence_certified": False,
        },
        "sections": {
            label: _section(rows, seed=seed, resamples=resamples)
            for label, rows in sorted(sections.items())
        },
    }
    if result["sections"]["all"]["candidate_correct"] != sum(
        row["correct"] for row in candidate_score["by_task"].values()
    ) or result["sections"]["all"]["baseline_correct"] != sum(
        row["correct"] for row in baseline_score["by_task"].values()
    ):
        raise ValueError("Paired/native score disagreement")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.panel, args.candidate, args.baseline)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report_sha256": file_digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
