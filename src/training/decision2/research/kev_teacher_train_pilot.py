"""Bounded TRAIN-only native Kev signal screen; aggregate output only.

This is a teacher-admission diagnostic, never a release evaluation or a
distillation-target generator. The 96 source-group roster is exactly the one
used by the preceding Eikos teacher screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference.kev import KEV_MODEL_REVISION, load_native, model_fingerprint
from inference.run import local_revision
from training.model.data import file_sha256, load_partition

from .eikos_teacher_train_pilot import (
    KINDS,
    PER_KIND,
    probabilities,
    question,
    roster,
    roster_sha256,
    write_once,
)


def aggregate(
    rows: list[dict[str, Any]], decide: Any
) -> dict[str, dict[str, float | int]]:
    """Count invalid native outputs as failures at the fixed denominator."""
    results: dict[str, dict[str, float | int]] = {
        kind: {
            "valid": 0,
            "invalid": 0,
            "ties": 0,
            "correct": 0,
            "brier_sum": 0.0,
            "gold_probability_sum": 0.0,
        }
        for kind in KINDS
    }
    for row in rows:
        kind = row["task_type"]
        stats = results[kind]
        response = decide(row["state"], {"decision": question(row)})
        if response.get("status") == "context_overflow":
            stats["invalid"] += 1
            continue
        if not isinstance(response.get("answers"), dict) or set(
            response["answers"]
        ) != {"decision"}:
            raise ValueError("Native teacher question roster differs")
        probs = probabilities(row, response["answers"]["decision"])
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
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--revision", default=KEV_MODEL_REVISION)
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
                    "per_type": PER_KIND,
                    "groups": len({row["group_id"] for row in selected}),
                    "roster_sha256": identity,
                },
                sort_keys=True,
            )
        )
        return
    if (
        args.revision != KEV_MODEL_REVISION
        or args.roster_sha256 != identity
        or args.model_path is None
        or args.source_path is None
        or args.output is None
    ):
        raise ValueError("Teacher source, roster, model path and output must be frozen")
    if not local_revision(args.model_path, args.revision):
        raise ValueError("Kev local download does not attest its pinned revision")
    fingerprint = model_fingerprint(args.model_path)
    model, decide, runtime = load_native(args.model_path, args.source_path, "cuda:0")
    assert model is not None
    stats = aggregate(selected, decide)
    payload = {
        "schema": "decision2-kev-teacher-train-screen/1",
        "source": f"jaredpalmer/kev-4b@{KEV_MODEL_REVISION}",
        "model_fingerprint": fingerprint,
        "runtime": runtime,
        "train_sha256": args.train_sha256,
        "roster_sha256": identity,
        "script_sha256": file_sha256(__file__),
        "sampled_rows": len(selected),
        "sampled_groups": len({row["group_id"] for row in selected}),
        "by_type": stats,
    }
    write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
