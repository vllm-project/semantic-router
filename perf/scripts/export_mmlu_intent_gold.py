#!/usr/bin/env python3
"""
Export MMLU questions into JSONL for Go intent-accuracy benchmark.

Uses the same dataset loading flow as:
bench/reasoning/router_reason_bench_multi_dataset.py
  dataset = DatasetFactory.create_dataset("mmlu")
  questions, dataset_info = dataset.load_dataset(...)

Output format (JSONL):
  {"text": "...", "gold_intent": "..."}
where gold_intent is question.category.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable when running from perf/scripts
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.reasoning.dataset_factory import DatasetFactory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MMLU intent gold JSONL")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "perf" / "testdata" / "mmlu_intent_gold.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Optional MMLU categories to include",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=20,
        help="Number of samples per category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = DatasetFactory.create_dataset("mmlu")
    questions, dataset_info = dataset.load_dataset(
        categories=args.categories,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    if not questions:
        print("No MMLU questions loaded.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "# Generated from MMLU via bench/reasoning DatasetFactory.\n"
            "# Fields: text, gold_intent (category).\n"
        )
        for q in questions:
            f.write(
                json.dumps(
                    {
                        "text": q.question,
                        "gold_intent": q.category,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Exported {len(questions)} MMLU rows across {len(dataset_info.categories)} categories to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
