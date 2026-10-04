"""The registered interpolation choice: among a TF arm's points, the highest RP-DEV final macro accuracy whose SELECT
family-macro accuracy is at least the release's minus ``--floor``; ties go to the larger alpha.

usage: python3 -m v2.reasoning.select_alpha --release R.json --point 0.5=A.json --point 0.75=B.json ... [--floor 0.01]
(each JSON a v2.reasoning.devread output over rpdev-final, rpdev-nodes and SELECT)
"""

from __future__ import annotations

import argparse
import json


def macro(readout: dict, name: str) -> float:
    return readout["file_macro_accuracy"][name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--point", action="append", required=True, help="alpha=path")
    parser.add_argument("--floor", type=float, default=0.01)
    args = parser.parse_args()
    release = json.load(open(args.release))
    floor = macro(release, "select.jsonl") - args.floor
    rows = []
    for spec in args.point:
        alpha, _, path = spec.partition("=")
        readout = json.load(open(path))
        rows.append(
            {
                "alpha": float(alpha),
                "rpdev_final": macro(readout, "rpdev-final.jsonl"),
                "rpdev_nodes": macro(readout, "rpdev-nodes.jsonl"),
                "select": macro(readout, "select.jsonl"),
            }
        )
    eligible = [r for r in rows if r["select"] >= floor]
    chosen = (
        max(eligible, key=lambda r: (round(r["rpdev_final"], 6), r["alpha"]))
        if eligible
        else None
    )
    print(
        json.dumps(
            {
                "release": {
                    "rpdev_final": macro(release, "rpdev-final.jsonl"),
                    "select": macro(release, "select.jsonl"),
                },
                "select_floor": floor,
                "points": rows,
                "chosen_alpha": chosen["alpha"] if chosen else None,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
