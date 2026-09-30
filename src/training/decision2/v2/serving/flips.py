"""Which answers a serving path changes, and how close to a tie they were when scored.

    python3 -m v2.serving.flips --stored DIR --panel NAME... --run LABEL=RUN_DIR...

For every panel and run (a plugin session directory with ``<panel>.predictions.jsonl``):
the changed slots against the stored scored predictions in DIR (release-parity
definition), their stored margins (Noul ``|p - 0.5|``; Choice and Score top-1
minus top-2 probability), how many changed slots all runs share, and how many of
the panel's slots have a stored margin at or below the largest changed margin.
Per-prompt answers stay where they are; only counts are printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2.release.examples import category, numbers

from .measure import quantile


def margin(answer: Any) -> float | None:
    if not isinstance(answer, dict) or "error" in answer:
        return None
    if isinstance(answer.get("noul"), (int, float)):
        return abs(float(answer["noul"]) - 0.5)
    probs = sorted(
        (float(v) for v in (answer.get("probabilities") or {}).values()), reverse=True
    )
    return probs[0] - probs[1] if len(probs) >= 2 else None


def changed(left: Any, right: Any) -> bool:
    return category(left) != category(right) or set(numbers(left)) != set(
        numbers(right)
    )


def read_answers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        return {row["id"]: row.get("answers") or {} for row in rows}


def panel_flips(
    stored: dict[str, dict[str, Any]], runs: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    margins = {
        (pid, qid): margin(answer)
        for pid, answers in stored.items()
        for qid, answer in answers.items()
    }
    out: dict[str, Any] = {"slots": len(margins), "runs": {}}
    flipped: dict[str, set[tuple[str, str]]] = {}
    for label, answers in runs.items():
        keys = {
            (pid, qid)
            for (pid, qid) in margins
            if pid in answers
            and qid in answers[pid]
            and changed(answers[pid][qid], stored[pid][qid])
        }
        flipped[label] = keys
        values = [margins[k] for k in keys if margins[k] is not None]
        out["runs"][label] = {
            "changed": len(keys),
            "stored_margin_max": max(values, default=None),
            "stored_margin_p50": quantile(values, 0.5),
        }
    if flipped:
        out["changed_in_every_run"] = len(set.intersection(*flipped.values()))
        top = max(
            (r["stored_margin_max"] or 0.0 for r in out["runs"].values()), default=0.0
        )
        out["slots_at_or_below_max_changed_margin"] = sum(
            1 for m in margins.values() if m is not None and m <= top
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    result = {}
    for panel in args.panel:
        stored = read_answers(args.stored / f"{panel}.predictions.jsonl")
        result[panel] = panel_flips(
            stored,
            {
                label: read_answers(Path(run) / f"{panel}.predictions.jsonl")
                for label, run in runs.items()
            },
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
