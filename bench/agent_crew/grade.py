#!/usr/bin/env python3
"""Grade a run's final review against defects.json, by hand.

    python grade.py runs/per-call/20260911T101500Z-1     # one run
    python grade.py --ungraded                            # every run without grades.json
    python grade.py --ungraded --blind                    # shuffled, arm names hidden

Hints only point at words the review uses. You decide whether the defect was found.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFECTS = json.loads((HERE / "defects.json").read_text())


def grade(run_dir: Path, blind: bool = False) -> None:
    review = (run_dir / "review.md").read_text()
    title = "review (arm hidden)" if blind else str(run_dir)
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n{review}\n{'-' * 72}")
    lowered = review.lower()
    found = {}
    for defect in DEFECTS:
        hits = [h for h in defect["hints"] if h in lowered]
        hint = f"  (mentions: {', '.join(hits)})" if hits else ""
        prompt = f"{defect['id']} {defect['file']}:{defect['lines']}  {defect['defect']}{hint}\n   found? [y/n] "
        answer = ""
        while answer not in ("y", "n"):
            answer = input(prompt).strip().lower()
        found[defect["id"]] = answer == "y"
    result = {
        "found": found,
        "defects_found": sum(found.values()),
        "defects_total": len(DEFECTS),
        "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "grades.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {result['defects_found']} of {result['defects_total']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--ungraded", action="store_true")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="shuffle runs and hide which arm produced each review",
    )
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    run_dirs = list(args.run_dirs)
    if args.ungraded:
        run_dirs += sorted(
            p.parent
            for p in args.runs.glob("*/*/review.md")
            if not (p.parent / "grades.json").exists()
        )
    if not run_dirs:
        parser.error("give a run directory or --ungraded")
    if args.blind:
        random.shuffle(run_dirs)
    for run_dir in run_dirs:
        grade(run_dir, args.blind)


if __name__ == "__main__":
    main()
