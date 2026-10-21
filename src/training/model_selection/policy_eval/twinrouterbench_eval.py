"""Score tier answers on the TwinRouterBench static track with the benchmark's own scorer.

TwinRouterBench (CommonstackAI, Apache-2.0) labels 970 router-visible prefixes,
each right before one model call in an agent run, with the cheapest tier
(low, mid, mid_high, high) whose call was verified to work. A router's tier per
row is scored with the benchmark's scores_v2, whose combined score the paper
reports in its Table 2.

The tier comes from the judge's answer to the tier question
(questions/twinrouterbench.json): the arg-max, or the cheapest tier whose
cumulative probability from the top tier stays under a cut (a lower cut routes
up more often). --constant scores one fixed tier on every row instead.

Usage: python twinrouterbench_eval.py --checkout DIR [--answers FILE] [--constant low]
DIR is a checkout of CommonstackAI/TwinRouterBench at the commit in d3_judge.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIERS = ["low", "mid", "mid_high", "high"]
CUTS = [None, 0.5, 0.3, 0.2, 0.1, 0.05]
DEFAULT_ANSWERS = HERE / "data" / "twinrouterbench.d3-turbo.answers.jsonl"


def tier(probs, cut):
    """Arg-max tier, or walk down from the top tier while the mass above stays under the cut."""
    if cut is None:
        return max(range(len(probs)), key=lambda i: probs[i])
    above = 0.0
    for t in range(len(TIERS) - 1, 0, -1):
        above += probs[t]
        if above >= cut:
            return t
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--answers", default=str(DEFAULT_ANSWERS))
    ap.add_argument("--constant", choices=TIERS, help="score one fixed tier instead")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.checkout)
    from main.dataset import QUESTION_BANK_PATH  # noqa: PLC0415
    from main.eval.predictors import FunctionPredictor  # noqa: PLC0415
    from main.eval.runner import (  # noqa: PLC0415
        build_eval_summary,
        evaluate_question_bank_rows,
    )

    rows = [
        json.loads(line) for line in Path(QUESTION_BANK_PATH).read_text().splitlines()
    ]
    for r in rows:
        r["target_tier_id"] = int(r["target_tier_id"])
    if a.constant:
        routers = {f"always {a.constant}": lambda row: TIERS.index(a.constant)}
    else:
        answers = {
            x["id"]: x["probs"]["tier"]
            for x in map(json.loads, Path(a.answers).read_text().splitlines())
        }
        routers = {
            ("argmax" if cut is None else f"cut {cut}"): (
                lambda row, cut=cut: tier(answers[row["id"]], cut)
            )
            for cut in CUTS
        }
    for name, route in routers.items():
        per_row, errors, correct = evaluate_question_bank_rows(
            FunctionPredictor(route), rows, predictor_label=name
        )
        s = build_eval_summary(
            per_row=per_row,
            errors=errors,
            correct=correct,
            predictor_label=name,
            shard=QUESTION_BANK_PATH,
            sample_mode="full",
            seed=0,
            proportional_quotas=None,
        )["scores_v2"]
        keys = [k for k in s if k.endswith("_percent")]
        print(
            f"{name:12s} "
            + "  ".join(f"{k.replace('_percent', '')} {s[k]:.2f}" for k in keys)
        )


if __name__ == "__main__":
    main()
