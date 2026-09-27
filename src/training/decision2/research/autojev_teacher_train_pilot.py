"""Bounded native AutoJev teacher screen on disjoint TRAIN groups only.

This produces private aggregate diagnostics, never student targets or eval scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from inference.autojev27 import (
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_REVISION,
    _admission_reason,
    _native_answer,
    verify_release,
)
from research.eikos_teacher_train_pilot import (
    KINDS,
    probabilities,
    question,
    roster,
    roster_sha256,
    write_once,
)
from training.model.data import file_sha256, load_partition

Decision = Callable[[Any, dict[str, Any]], tuple[dict[str, Any], int]]


def aggregate(rows: list[dict[str, Any]], decide: Decision) -> dict[str, Any]:
    """Count exact native answers; only documented admission failures are invalid."""
    results: dict[str, dict[str, float | int]] = {
        kind: {
            "valid": 0,
            "invalid": 0,
            "ties": 0,
            "correct": 0,
            "gold_probability_sum": 0.0,
            "brier_sum": 0.0,
        }
        for kind in KINDS
    }
    for row in rows:
        stats = results[row["task_type"]]
        try:
            answer, _ = decide(row["state"], question(row))
            probs = probabilities(row, answer)
        except ValueError as error:
            if _admission_reason(error) is None:
                raise
            stats["invalid"] += 1
            continue
        gold = row["options"][row["label"]]["key"]
        maximum = max(probs.values())
        winners = [key for key, value in probs.items() if abs(value - maximum) <= 1e-8]
        stats["valid"] += 1
        stats["ties"] += int(len(winners) != 1)
        stats["correct"] += int(len(winners) == 1 and winners[0] == gold)
        stats["gold_probability_sum"] += probs[gold]
        stats["brier_sum"] += (
            sum((value - int(key == gold)) ** 2 for key, value in probs.items()) / 2
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if file_sha256(args.train) != args.train_sha256:
        raise ValueError("Frozen TRAIN bytes differ")
    selected = roster(load_partition(args.train, "train"))
    identity = roster_sha256(selected)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": len(selected),
                    "groups": len({row["group_id"] for row in selected}),
                    "roster_sha256": identity,
                }
            )
        )
        return
    if (
        args.revision != MODEL_REVISION
        or args.roster_sha256 != identity
        or args.model_path is None
        or args.source_path is None
        or args.output is None
    ):
        raise ValueError("Teacher source, roster, model path and output must be frozen")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Private output must be fresh and its directory must exist")
    release = verify_release(args.model_path, args.source_path, args.revision)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source_path / "src"))
    from autojev.model import DecisionModel

    model = DecisionModel(checkpoint=args.model_path, device="cuda:0", train=False)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_count != release["loaded_parameters"]:
        raise ValueError("Loaded parameters differ from frozen AutoJev package")
    stats = aggregate(selected, lambda state, item: _native_answer(model, state, item))
    payload = {
        "schema": "decision2-autojev-teacher-train-screen/1",
        "source": f"{MODEL_ID}@{MODEL_REVISION}",
        "source_revision": SOURCE_REVISION,
        **release,
        "train_sha256": args.train_sha256,
        "roster_sha256": identity,
        "script_sha256": file_sha256(__file__),
        "sampled_rows": len(selected),
        "sampled_groups": len({row["group_id"] for row in selected}),
        "by_type": stats,
    }
    write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
