"""Decoder M6b finalists (prereg dec-m6b-prereg-2026-09-30.md): the preregistered N6D points, written in the
m6_finalists.py schema so ops/m6/m6-formal.sh stages them unchanged (M6_SELECT = the output's directory).

Nothing is selected here: the points and their slots are fixed by the preregistration, in the order given. Each
point's M6 development values (typed DEV T, G, CSS-pilot H3, P and the M6 rule's reasons) are copied from the M6
4B rules output for the record; development readouts are never release scores.

usage: python3 m6b_finalists.py --rules /data/dev2/runs/dec/m6/select/4b-rules.json \
    --lines-root /data/dev2/runs/dec/m6/lines/4b --point 4b-N6D-b1 --point 4b-N6D-b2_3 \
    --output /data/dev2/runs/dec/m6b/select/4b-finalists.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dec-m6b-finalists/1"
_spec = importlib.util.spec_from_file_location(
    "m6_finalists", Path(__file__).resolve().parents[1] / "m6" / "m6_finalists.py"
)
m6f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m6f)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalists(
    rules: dict[str, Any], lines_root: Path, points: list[str]
) -> list[dict[str, Any]]:
    rows = {
        row["arm"]: (line, row)
        for line, entry in rules["lines"].items()
        for row in entry["rows"]
    }
    out = []
    for slot, point in enumerate(points, 1):
        if point not in rows:
            raise SystemExit(f"{point} has no development row in the rules output")
        line, row = rows[point]
        out.append(
            {
                "slot": slot,
                "line": line,
                "point": point,
                "step": row["alpha"],
                "T": row["T"],
                "G": row["G"],
                "H3": row["H3"],
                "proxy": row["proxy"],
                "m6_eligible": row["eligible"],
                "m6_reasons": row["reasons"],
                **m6f.point_info(lines_root, point),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--lines-root", type=Path, required=True)
    p.add_argument("--point", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if len(set(args.point)) != len(args.point):
        p.error("repeated --point")
    if not args.output.name.endswith("-finalists.json"):
        p.error("--output must end with -finalists.json")
    if args.output.exists():
        p.error(f"{args.output} exists")
    rules = json.loads(args.rules.read_text())
    result = {
        "schema": SCHEMA,
        "tier": args.output.name.split("-", 1)[0],
        "role": (
            "preregistered formal finalists (prereg dec-m6b-prereg-2026-09-30.md); no development selection; "
            "not a release or post-key score"
        ),
        "rules_output": str(args.rules),
        "rules_output_sha256": sha_file(args.rules),
        "finalists": finalists(rules, args.lines_root, args.point),
    }
    for f in result["finalists"]:
        for key in ("typed_dev_predictions", "css_pilot_predictions"):
            if not Path(f[key]).is_file():
                raise SystemExit(f"{f['point']}: missing {f[key]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            [
                (f["slot"], f["point"], f["files_sha256_list_sha256"])
                for f in result["finalists"]
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
