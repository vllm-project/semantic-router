"""Answer identity between two collections of the same gold-free panels (CPU, no gold).

Splits answer slots into valid in both, valid only in the new run (for example
inputs recovered by a longer limit), valid only in the reference, and invalid in
both. On slots valid in both it reports category changes (the runner's
``answer_category``) and max / p99 absolute drift of every numeric answer leaf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.eval.same_panel import answer_category, numeric_leaves, read_jsonl, sha_file


def valid(answer) -> bool:
    return isinstance(answer, dict) and "error" not in answer


def compare(new: Path, reference: Path) -> dict:
    a = {row["id"]: row for row in read_jsonl(new)}
    b = {row["id"]: row for row in read_jsonl(reference)}
    if set(a) != set(b):
        raise ValueError("the two collections cover different items")
    counts = {
        "both_valid": 0,
        "new_only_valid": 0,
        "reference_only_valid": 0,
        "both_invalid": 0,
    }
    changed, drifts = 0, []
    for item_id in sorted(a):
        left, right = a[item_id]["answers"], b[item_id]["answers"]
        if set(left) != set(right):
            raise ValueError(f"{item_id}: question keys differ")
        for key in sorted(left):
            x, y = left[key], right[key]
            if valid(x) and valid(y):
                counts["both_valid"] += 1
                changed += answer_category(x) != answer_category(y)
                nx, ny = numeric_leaves(x), numeric_leaves(y)
                drifts.extend(abs(nx[k] - ny[k]) for k in set(nx) & set(ny))
            elif valid(x):
                counts["new_only_valid"] += 1
            elif valid(y):
                counts["reference_only_valid"] += 1
            else:
                counts["both_invalid"] += 1
    drifts.sort()
    return {
        **counts,
        "category_changes_on_both_valid": changed,
        "max_abs_drift_on_both_valid": drifts[-1] if drifts else 0.0,
        "p99_abs_drift_on_both_valid": (
            drifts[int(0.99 * (len(drifts) - 1))] if drifts else 0.0
        ),
        "new_sha256": sha_file(new),
        "reference_sha256": sha_file(reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", required=True, help="PANEL=NEW,REFERENCE"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "decision2-27b-identity-check/1", "panels": {}}
    for spec in args.pair:
        panel, paths = spec.split("=", 1)
        new, reference = paths.split(",")
        result["panels"][panel] = compare(Path(new), Path(reference))
        print(json.dumps({panel: result["panels"][panel]}), flush=True)
    args.output.write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
