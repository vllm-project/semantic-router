"""Red-team evasion benchmark runner.

Usage:
    python3 -m bench.redteam.evaluate \
        --model models/mmbert32k-jailbreak-detector-merged \
        --dataset jailbreakbench --threshold 0.7

Exits non-zero when a gate is supplied and the model misses it, which is what
lets the benchmark run as a graduation gate in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attack import DEFAULT_WORD_POOL, greedy_suffix_attack, summarize
from .datasets import load_prompts
from .scorers import TransformerScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local classifier artifact")
    parser.add_argument("--dataset", default="jailbreakbench")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-words", type=int, default=10)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--word-pool", default=None, help="newline-delimited file")
    parser.add_argument("--output", default=None, help="write the JSON report here")
    parser.add_argument(
        "--min-baseline-recall",
        type=float,
        default=None,
        help="fail when baseline recall is below this",
    )
    parser.add_argument(
        "--max-flip-rate",
        type=float,
        default=None,
        help="fail when the evasion flip rate is above this",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompts = load_prompts(args.dataset)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    word_pool = (
        tuple(Path(args.word_pool).read_text().split())
        if args.word_pool
        else DEFAULT_WORD_POOL
    )

    scorer = TransformerScorer(
        args.model, device=args.device, batch_size=args.batch_size
    )
    print(f"scoring {len(prompts)} prompts on {scorer.device}", flush=True)

    results = [
        greedy_suffix_attack(
            prompt,
            scorer,
            threshold=args.threshold,
            word_pool=word_pool,
            max_words=args.max_words,
        )
        for prompt in prompts
    ]

    report = summarize(results).as_dict()
    report["model"] = args.model
    report["dataset"] = args.dataset
    report["threshold"] = args.threshold
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    failures: list[str] = []
    if args.min_baseline_recall is not None:
        recall = float(report["baseline_recall"])
        if recall < args.min_baseline_recall:
            failures.append(
                f"baseline_recall {recall} below {args.min_baseline_recall}"
            )
    if args.max_flip_rate is not None:
        flip_rate = float(report["flip_rate"])
        if flip_rate > args.max_flip_rate:
            failures.append(f"flip_rate {flip_rate} above {args.max_flip_rate}")

    for failure in failures:
        print(f"gate failed: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
