"""Summarize one development readout from pinned typed and transfer score reports.

Proxy `P = 100 * sqrt(T_dev * H_pilot)`: typed DEV four-family macro accuracy
times CSS pilot median task macro-F1, as in the official-Qwen control's gate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json


def summarize(
    typed: dict[str, Any], css: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    t = typed["macro_family_accuracy"]
    h = css["roles"]["pilot"]["median_task_macro_f1_all"]
    levels: Counter[str] = Counter()
    for row in predictions:
        for answer in row["answers"].values():
            if answer.get("type") == "score" and "probabilities" in answer:
                p = answer["probabilities"]
                levels[max(p, key=p.get)] += 1
    by_type = {
        kind: {"correct": v["correct_n"], "n": v["n"], "invalid": v["n"] - v["valid_n"]}
        for kind, v in typed["by_type"].items()
    }
    return {
        "T_dev": t,
        "H_pilot": h,
        "proxy": 100 * math.sqrt(t * h),
        "typed_correct": typed["overall"]["correct_n"],
        "typed_by_type": by_type,
        "typed_by_family": {
            k: v["accuracy_all"] for k, v in typed["by_family"].items()
        },
        "typed_invalid": typed["overall"]["n"] - typed["overall"]["valid_n"],
        "score_levels_predicted": dict(sorted(levels.items())),
        "css_pilot_tasks": {k: v["macro_f1_all"] for k, v in css["tasks"].items()},
        "css_pilot_correct": sum(v["correct_n"] for v in css["tasks"].values()),
        "css_pilot_invalid": sum(
            v["invalid_or_missing_n"] for v in css["tasks"].values()
        ),
        "typed_report_predictions_sha256": typed["predictions_sha256"],
        "css_report_predictions_sha256": css["predictions_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed", type=Path, required=True)
    parser.add_argument("--css", type=Path, required=True)
    parser.add_argument("--typed-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.typed.read_text()),
        json.loads(args.css.read_text()),
        read_jsonl(args.typed_predictions),
    )
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
