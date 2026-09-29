"""Score level usage of a formal run on typed FINAL (post-key; aggregates only).

    PYTHONPATH=<src>/training/decision2:<src>/training/decision2/v2/9b python3 -m \
        lux9b.score_levels --run-dir RUN --output OUT [--panel-root ROOT]

For the Score slots of typed FINAL (five ordered levels, 0-4) it reports the predicted
level histogram (``invalid`` = no valid answer), the gold level histogram, recall per gold
level and the largest single-answer share (share of all Score slots taken by the most
frequent predicted answer, invalid included), exactly as ``v2.eval.gates types`` counts
them (``type_summary``) from the sealed ``typed-final`` predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.gates import type_summary, verified
from v2.eval.same_panel import read_jsonl, sha_file, write_json

SCHEMA = "dev2-9b-score-levels/1"
LEVELS = 5


def score_levels(
    gold: list[dict[str, Any]], predictions: dict[str, Any]
) -> dict[str, Any]:
    entry = type_summary(gold, predictions)["score"]
    levels = [str(level) for level in range(LEVELS)]
    extra = set(entry["gold_distribution"]) - set(levels)
    if extra:
        raise ValueError(f"gold Score levels outside 0-{LEVELS - 1}: {sorted(extra)}")
    pred = entry["predicted_distribution"]
    top = next(iter(pred))  # most_common order, as predicted_top_share
    return {
        "slots": entry["slots"],
        "valid": entry["valid"],
        "accuracy": entry["accuracy"],
        "predicted_levels": {level: pred.get(level, 0) for level in levels},
        "invalid": pred.get("None", 0),
        "gold_levels": {
            level: entry["gold_distribution"].get(level, 0) for level in levels
        },
        "recall_by_level": {
            level: entry["recall_by_level"].get(level) for level in levels
        },
        "level0_recall": entry["recall_by_level"].get("0"),
        "levels_used": sum(pred.get(level, 0) > 0 for level in levels),
        "largest_answer": "invalid" if top == "None" else top,
        "largest_answer_share": entry["predicted_top_share"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    args = ap.parse_args(argv)
    from benchmark.score import load_jsonl

    panels.verify(args.panel_root, ["typed-final"])
    gold = load_jsonl(panels.path(args.panel_root, "typed-final", "gold"))
    gold = list(gold.values()) if isinstance(gold, dict) else gold
    path = verified(args.run_dir, "typed-final")
    predictions = {row["id"]: row for row in read_jsonl(path)}
    result = {
        "schema": SCHEMA,
        "label": "post-key typed FINAL Score level usage (aggregates only)",
        "run": str(args.run_dir),
        "predictions_sha256": sha_file(path),
        "gold_sha256": panels.FORMAL["typed-final"]["gold_sha256"],
        "score": score_levels(gold, predictions),
    }
    sha = write_json(args.output, result)
    s = result["score"]
    print(
        json.dumps(
            {
                "predicted": s["predicted_levels"],
                "invalid": s["invalid"],
                "gold": s["gold_levels"],
                "level0_recall": s["level0_recall"],
                "largest_answer_share": s["largest_answer_share"],
                "sha256": sha,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
