#!/usr/bin/env python3
"""Prepare a deterministic, manifest-ready MMLU-Pro question set.

The resulting JSON array contains the prompt and gold answer needed by
``collect_confidence_results.py``.  Dataset download is intentionally kept
outside the artifact builder so the artifact can record the prepared file's
digest and the exact split assignment.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_SPLIT_COUNT = 3


def _format_prompt(question: str, options: list[str]) -> str:
    labels = "ABCDEFGHIJ"
    rendered_options = "\n".join(
        f"{labels[index]}) {option}"
        for index, option in enumerate(options)
        if str(option).strip().lower() != "n/a"
    )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{rendered_options}\n\n"
        "Choose the correct option. Respond with exactly 'Answer: [letter]'."
    )


def _prepare_rows(
    rows: list[dict[str, Any]],
    samples_per_category: int,
    categories: set[str] | None,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or "").strip()
        question_id = str(row.get("question_id") or "").strip()
        options = row.get("options")
        if not category or not question_id or not isinstance(options, list):
            raise ValueError("MMLU-Pro rows require question_id, category, and options")
        if categories and category not in categories:
            continue
        grouped[category].append(row)

    if not grouped:
        raise ValueError("no MMLU-Pro rows matched the requested categories")

    sampler = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        category_rows = sorted(
            grouped[category], key=lambda row: str(row["question_id"])
        )
        if len(category_rows) > samples_per_category:
            category_rows = sampler.sample(category_rows, samples_per_category)
        selected.extend(category_rows)

    sampler.shuffle(selected)
    if len(selected) < MIN_SPLIT_COUNT:
        raise ValueError("at least three questions are required for three splits")

    train_count = max(1, int(len(selected) * 0.6))
    calibration_count = max(1, int(len(selected) * 0.2))
    if train_count + calibration_count >= len(selected):
        train_count = len(selected) - 2
        calibration_count = 1

    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        if index < train_count:
            split = "train"
        elif index < train_count + calibration_count:
            split = "calibration"
        else:
            split = "held_out"
        prepared.append(
            {
                "question_id": str(row["question_id"]),
                "category": str(row["category"]),
                "prompt": _format_prompt(str(row["question"]), row["options"]),
                "correct_answer": str(row["answer"]).strip().upper(),
                "split": split,
            }
        )
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic MMLU-Pro confidence-calibration inputs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.samples_per_category < 1:
        parser.error("--samples-per-category must be positive")

    try:
        from datasets import load_dataset  # noqa: PLC0415

        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        rows = [dict(row) for row in dataset]
        prepared = _prepare_rows(
            rows,
            samples_per_category=args.samples_per_category,
            categories=set(args.categories) if args.categories else None,
            seed=args.seed,
        )
    except ImportError:
        parser.error("the datasets package is required; install it in the active venv")
    except Exception as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = {
        split: sum(row["split"] == split for row in prepared)
        for split in (
            "train",
            "calibration",
            "held_out",
        )
    }
    print(f"Prepared {len(prepared)} questions: {counts}")
    print(f"Dataset written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
