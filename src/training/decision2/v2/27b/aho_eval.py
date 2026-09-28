"""Arm held-out (AHO) slice readout at a completed run's BEST checkpoint (GPU).

AHO slices are labeled, group-disjoint select-role rows from the data registry.
They are read once at the frozen BEST checkpoint with the pinned trainer's own
``evaluate`` (raw probabilities, argmax correctness) and never used for
selection. Rows over the training limit are counted as failures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(
    records: list[dict], rows: dict[str, dict], over_limit: list[str]
) -> dict:
    cells: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for record in records:
        row = rows[record["id"]]
        keys = [
            record["task_type"],
            f"family:{record['family']}",
            f"language:{row['language']}",
        ]
        if record["task_type"] == "score":
            keys.append(f"levels:{len(row['options'])}")
        for key in keys:
            cell = cells["all"][key]
            cell[0] += int(record["correct"])
            cell[1] += 1
    for row_id in over_limit:
        row = rows[row_id]
        for key in (
            row["task_type"],
            f"family:{row['family']}",
            f"language:{row['language']}",
        ):
            cells["all"][key][1] += 1
    total = len(records) + len(over_limit)
    families = {k: v for k, v in cells["all"].items() if k.startswith("family:")}
    return {
        "n": total,
        "over_limit": len(over_limit),
        "correct": sum(int(r["correct"]) for r in records),
        "micro_accuracy": sum(int(r["correct"]) for r in records) / total,
        "family_macro_accuracy": sum(c / n for c, n in families.values())
        / len(families),
        "cells": {
            k: {"correct": c, "n": n, "accuracy": c / n}
            for k, (c, n) in sorted(cells["all"].items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--slice", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, encode
    from training.model.train import evaluate

    if not (args.run_dir / "COMPLETE.json").is_file():
        raise SystemExit("Run is not complete")
    best = json.loads((args.run_dir / "BEST.json").read_text(encoding="utf-8"))[
        "checkpoint"
    ]
    model, tokenizer = DecisionModel.from_checkpoint(
        args.run_dir / best, source_path=args.source_path
    )
    device = torch.device("cuda:0")
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    summary = {
        "schema": "decision2-27b-aho-readout/1",
        "checkpoint": best,
        "slices": {},
    }
    for spec in args.slice:
        name, path = spec.split("=", 1)
        rows = load_partition(path, "select")
        items, over = [], []
        for row in rows:
            try:
                items.append(encode(row, tokenizer, args.max_length))
            except ValueError as exc:
                if "exceeds max_length" not in str(exc):
                    raise
                over.append(row["id"])
        tag = f"aho-{name}"
        evaluate(
            model,
            items,
            pad_id=pad_id,
            batch_size=1,
            device=device,
            output=args.out_dir,
            tag=tag,
        )
        records = [
            json.loads(line)
            for line in (args.out_dir / f"{tag}-predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        summary["slices"][name] = summarize(
            records, {row["id"]: row for row in rows}, over
        )
        print(
            json.dumps(
                {
                    "slice": name,
                    "micro_accuracy": summary["slices"][name]["micro_accuracy"],
                }
            ),
            flush=True,
        )
    (args.out_dir / "aho-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
