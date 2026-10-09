#!/usr/bin/env python3
"""Capability differentiation and saturation check for model evaluation.

Given two models' per-question results on the same sample, emits a
go/no-go verdict on three criteria:
  - Capability gap: if gap < redline (default 0.08), models lack
    sufficient differentiation for routing to be meaningful.
  - Saturation: both models >95% accuracy = no routing headroom.
  - Mutual exclusivity: high one-correct-other-wrong rate is a weak
    signal that the dataset can distinguish the models (low rate +
    high accuracy may indicate the dataset is too easy or leaked into
    training data, but this is NOT proof of contamination — public
    benchmarks aged 1-2 years may already be in model corpora).

Primary purpose: verify models have enough differentiation for routing,
and validate that the dataset itself can distinguish between models (if gap
is too small, the dataset may be too easy/hard/wrong format — not the models'
fault). Catching this early saves the entire downstream GPU cost.

Usage:
    python differentiation_check.py \\
        --small results/model_a/judged.jsonl \\
        --large results/model_b/judged.jsonl \\
        [--gap-redline 0.08] [--saturation-threshold 0.95] \\
        --output results/differentiation_verdict.json
"""
import argparse
import json
import sys
from pathlib import Path


def load_records(path: str) -> dict[str, dict]:
    """Load JSONL per-question records, keyed by question_id."""
    records = {}
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item.get("id") or item.get("question_id") or item.get("task_id")
            if qid:
                records[qid] = item
    return records


def classify_pair(a_correct: bool, b_correct: bool) -> str:
    """Classify a question pair into one of four categories."""
    if a_correct and b_correct:
        return "both_correct"
    if a_correct and not b_correct:
        return "a_only"
    if not a_correct and b_correct:
        return "b_only"
    return "both_wrong"


def check_differentiation(
    small: dict, large: dict, saturation_threshold: float = 0.95
) -> dict:
    """Run differentiation and saturation checks."""
    common_ids = set(small.keys()) & set(large.keys())
    if not common_ids:
        return {"verdict": "ERROR", "reason": "no common question IDs"}

    counts = {"both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0}
    for qid in common_ids:
        a_ok = small[qid].get("is_correct", False)
        b_ok = large[qid].get("is_correct", False)
        counts[classify_pair(a_ok, b_ok)] += 1

    n = len(common_ids)
    acc_a = (counts["both_correct"] + counts["a_only"]) / n
    acc_b = (counts["both_correct"] + counts["b_only"]) / n
    gap = abs(acc_a - acc_b)
    # Mutual exclusivity rate: questions where exactly one model is right
    excl = counts["a_only"] + counts["b_only"]
    excl_rate = excl / n

    # Saturation: both above the configured threshold → no routing headroom
    saturated = acc_a > saturation_threshold and acc_b > saturation_threshold

    return {
        "n_common": n,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "gap": round(gap, 4),
        "exclusive_rate": round(excl_rate, 4),
        "both_correct": counts["both_correct"],
        "a_only": counts["a_only"],
        "b_only": counts["b_only"],
        "both_wrong": counts["both_wrong"],
        "saturated": saturated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Capability differentiation and saturation pre-check"
    )
    parser.add_argument(
        "--small", required=True, help="JSONL: model A per-question results"
    )
    parser.add_argument(
        "--large", required=True, help="JSONL: model B per-question results"
    )
    parser.add_argument(
        "--gap-redline",
        type=float,
        default=0.08,
        help="Minimum accuracy gap for routing to be worthwhile (default: 0.08)",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.95,
        help="Saturation threshold (both models above this = no headroom)",
    )
    parser.add_argument("--output", default=None, help="Output JSON file for verdict")
    args = parser.parse_args()

    small = load_records(args.small)
    large = load_records(args.large)
    result = check_differentiation(small, large, args.saturation_threshold)

    if result.get("verdict") == "ERROR":
        print(f"ERROR: {result['reason']}")
        sys.exit(1)

    verdicts = []
    if result["gap"] < args.gap_redline:
        verdicts.append(
            f"NO-GO: gap {result['gap']:.4f} < redline {args.gap_redline} "
            f"(models too similar for routing)"
        )
    else:
        verdicts.append(f"GO: gap {result['gap']:.4f} >= redline {args.gap_redline}")

    if result["saturated"]:
        verdicts.append(
            f"NO-GO: both models >{args.saturation_threshold} "
            f"(saturated, no routing headroom)"
        )

    result["verdict"] = " | ".join(verdicts)
    result["gap_redline"] = args.gap_redline
    result["saturation_threshold"] = args.saturation_threshold

    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
