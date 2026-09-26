"""Retrospective macro-F1 re-selection from immutable native Kai SELECT traces.

This reads completed checkpoints' existing prediction files. It does not train,
inspect a pilot, or export raw examples. A selected step is a diagnostic only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

from transfer.score import macro_f1, read_jsonl
from transfer.build import sha_file


def summarize(
    select: dict[str, dict[str, Any]], prediction_path: Path
) -> dict[str, Any]:
    predictions = read_jsonl(prediction_path)
    if set(predictions) != set(select):
        raise ValueError("Checkpoint prediction IDs differ from SELECT")
    buckets: dict[str, list[tuple[str, str, dict[str, float], list[str]]]] = (
        defaultdict(list)
    )
    for item_id, row in select.items():
        pred = predictions[item_id]
        labels = [option["id"] for option in row["question"]["options"]]
        target, point = row["hard_target_id"], pred.get("choice_id")
        probs = pred.get("probabilities")
        if (
            row["question"]["type"] != "Choice"
            or target not in labels
            or point not in labels
            or pred.get("type") != "Choice"
            or pred.get("candidate_ids") != labels
            or not isinstance(probs, list)
            or len(probs) != len(labels)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in probs
            )
            or abs(sum(probs) - 1) > 0.02
        ):
            raise ValueError("Non-comparable native SELECT prediction")
        buckets[row["source_id"]].append(
            (target, point, dict(zip(labels, probs)), labels)
        )
    tasks: dict[str, Any] = {}
    for task, records in sorted(buckets.items()):
        labels = records[0][3]
        if any(record[3] != labels for record in records):
            raise ValueError("Task candidate order changed within SELECT")
        golds, points = [r[0] for r in records], [r[1] for r in records]
        tasks[task] = {
            "rows": len(records),
            "correct": sum(g == p for g, p in zip(golds, points)),
            "accuracy": sum(g == p for g, p in zip(golds, points)) / len(records),
            "macro_f1": macro_f1(golds, points, labels),
            "nll": statistics.mean(-math.log(max(r[2][r[0]], 1e-12)) for r in records),
            "label_support": {
                label: sum(g == label for g in golds) for label in labels
            },
            "label_predicted": {
                label: sum(p == label for p in points) for label in labels
            },
        }
    return {
        "prediction_sha256": sha_file(prediction_path),
        "tasks": tasks,
        "rows": sum(t["rows"] for t in tasks.values()),
        "row_accuracy": sum(t["correct"] for t in tasks.values())
        / sum(t["rows"] for t in tasks.values()),
        "median_task_macro_f1": statistics.median(
            t["macro_f1"] for t in tasks.values()
        ),
        "mean_task_macro_f1": statistics.mean(t["macro_f1"] for t in tasks.values()),
        "mean_task_nll": statistics.mean(t["nll"] for t in tasks.values()),
    }


def analyze(select_path: Path, run: Path, steps: list[int]) -> dict[str, Any]:
    select = read_jsonl(select_path)
    receipt = json.loads((run / "COMPLETE.json").read_text())
    run_receipt = json.loads((run / "RUN.json").read_text())
    if (
        receipt.get("status") != "COMPLETE_DECISION_FINETUNE"
        or receipt.get("identity") != run_receipt.get("identity")
        or receipt["identity"]["dev_sha256"] != sha_file(select_path)
    ):
        raise ValueError("SELECT differs from the completed native run")
    curves = {}
    for step in steps:
        path = run / "attempts" / "0001" / f"predictions-{step:06d}.jsonl"
        curve = json.loads(
            (run / "attempts" / "0001" / f"evaluation-{step:06d}.json").read_text()
        )
        item = summarize(select, path)
        if abs(item["row_accuracy"] - curve["metrics"]["row"]["hard_accuracy"]) > 1e-8:
            raise ValueError("Recomputed row accuracy differs from frozen evaluation")
        curves[step] = item
    best = sorted(
        steps,
        key=lambda step: (
            -curves[step]["median_task_macro_f1"],
            -curves[step]["mean_task_macro_f1"],
            -curves[step]["row_accuracy"],
            step,
        ),
    )[0]
    return {
        "schema_version": "kai06b-retrospective-macro-reselection/1",
        "selection_policy": "maximize median task macro-F1, then mean task macro-F1, then row accuracy, then earliest step",
        "select_sha256": sha_file(select_path),
        "run_complete_sha256": sha_file(run / "COMPLETE.json"),
        "run_identity_sha256": sha_file(run / "RUN.json"),
        "trained_selected_step": receipt["selected"]["step"],
        "macro_selected_step": best,
        "steps": {str(k): v for k, v in curves.items()},
        "scope": "retrospective development-only SELECT re-selection; no pilot/final labels used",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = [int(value) for value in args.steps.split(",")]
    if len(steps) != len(set(steps)) or steps != sorted(steps) or steps[0] != 0:
        raise ValueError("Steps must be unique, ordered, and start at zero")
    result = analyze(args.select, args.run, steps)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "macro_selected_step": result["macro_selected_step"],
                "trained_selected_step": result["trained_selected_step"],
            }
        )
    )


if __name__ == "__main__":
    main()
