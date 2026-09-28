"""Own-Lux1 soft targets on rights-clean TRAIN prompts only, plus the teacher screen.

`prompts` writes one gold-free System One request per TRAIN row (never a panel
item). Lux1 answers them through the eval track's `inference.run --backend lux`
with a persisted Triton autotune cache. `build` maps its answers to the native
candidate order used by every 0.6B arm, writes the teacher file and scores the
preregistered screen against TRAIN gold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import (
    file_sha256,
    load_rights_clean,
    native_keys,
    native_records,
    original_probabilities,
    read_jsonl,
    select_record,
    write_json,
    write_jsonl,
)

SCREEN = {
    "choice_accuracy": 0.625,
    "noul_accuracy": 0.625,
    "score_accuracy": 0.375,
    "three_level_score_accuracy": 0.375,
    "half_brier_max": 0.25,
}


def request(row: dict[str, Any]) -> dict[str, Any]:
    from training.data.build_kai06b_native_v1 import convert

    captured: dict[str, Any] = {}

    def capture(req: dict[str, Any], targets: Any, **_: Any) -> list[dict[str, Any]]:
        captured.update(req)
        return [{}]

    convert(row, capture)
    return {
        "id": row["id"],
        "state": captured["state"],
        "questions": captured["questions"],
    }


def prompts(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rights_clean(args.data_parent)["train"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sha = write_jsonl(args.output, [request(r) for r in rows])
    return {"rows": len(rows), "prompts_sha256": sha}


def native_vector(record: dict[str, Any], answer: dict[str, Any]) -> list[float]:
    ids = native_keys(record)
    if answer.get("type") == "noul":
        p = float(answer["noul"])
        vector = [1 - p, p]
    else:
        probs = answer["probabilities"]
        if set(probs) != set(ids):
            raise ValueError("Teacher option keys differ from native candidates")
        vector = [float(probs[k]) for k in ids]
    total = sum(vector)
    if not all(math.isfinite(v) and v >= 0 for v in vector) or abs(total - 1) > 0.02:
        raise ValueError("Teacher vector is not a finite distribution")
    return [v / total for v in vector]


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rights_clean(args.data_parent)["train"]
    records = native_records(rows, args.bundle)
    answers = {r["id"]: r for r in read_jsonl(args.predictions)}
    if set(answers) != {r["id"] for r in rows}:
        raise ValueError("Teacher predictions do not cover exactly the TRAIN rows")
    out, outcomes, invalid = [], [], 0
    for row, record in zip(rows, records):
        answer = answers[row["id"]]["answers"].get("decision")
        if (
            not isinstance(answer, dict)
            or "error" in answer
            or answer.get("type") is None
        ):
            invalid += 1
            continue
        vector = native_vector(record, answer)
        out.append(
            {
                "source_row_id": row["id"],
                "native_ids": native_keys(record),
                "probabilities": vector,
            }
        )
        outcomes.append(
            {
                **select_record(
                    row, original_probabilities(row, native_keys(record), vector)
                ),
                "levels": len(row["options"]),
            }
        )
    by_type: dict[str, dict[str, float]] = {}
    for kind in ("choice", "noul", "score"):
        subset = [o for o in outcomes if o["task_type"] == kind]
        by_type[kind] = {
            "n": len(subset),
            "accuracy": sum(o["correct"] for o in subset) / len(subset),
            "half_brier": sum(o["brier"] for o in subset) / len(subset),
        }
    three = [o for o in outcomes if o["task_type"] == "score" and o["levels"] == 3]
    three_acc = sum(o["correct"] for o in three) / len(three)
    gates = {
        "coverage": invalid == 0,
        "choice": by_type["choice"]["accuracy"] >= SCREEN["choice_accuracy"],
        "noul": by_type["noul"]["accuracy"] >= SCREEN["noul_accuracy"],
        "score": by_type["score"]["accuracy"] >= SCREEN["score_accuracy"],
        "three_level_score": three_acc >= SCREEN["three_level_score_accuracy"],
        "half_brier": all(
            v["half_brier"] <= SCREEN["half_brier_max"] for v in by_type.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    teacher_sha = write_jsonl(args.output, out)
    report = {
        "status": "SCREEN_PASS" if all(gates.values()) else "SCREEN_FAIL",
        "gates": gates,
        "thresholds": SCREEN,
        "rows": len(rows),
        "invalid": invalid,
        "by_type": by_type,
        "three_level_score": {"n": len(three), "accuracy": three_acc},
        "teacher_sha256": teacher_sha,
        "predictions_sha256": file_sha256(args.predictions),
    }
    write_json(args.output.with_name(args.output.name + ".screen.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--data-parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--data-parent", type=Path, required=True)
    b.add_argument("--bundle", type=Path, required=True)
    b.add_argument("--predictions", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prompts(args) if args.command == "prompts" else build(args), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
