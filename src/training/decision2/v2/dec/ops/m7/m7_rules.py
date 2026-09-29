"""Decoder M7 finalists per tier (prereg dec-m7-prereg-2026-09-30.md, "Development gates and finalists").

For each arm line (readout/L-<LINE>.json from m7-lines.sh; points beta 1, 1/2, 1/3 toward the arm soup, reference
<tier>-I = the incumbent on the same node, image and limit), a point passes the development gates when the 9B rule
module's `eligibility` (v2/9b/lux9b/m4_rules.py, unchanged) accepts it against I:
  (i) every type keeps c_t >= c_t,I - 0.03 n_t; (ii) CSS-pilot three-task mean H3 >= H3_I (the human-transfer
  non-decrease); (iii) no typed-DEV family below F_f,I - 0.10.
There is no typed-gain requirement. The line's pick is its largest passing step (beta 1, then 1/2, then 1/3); a pick
whose proxy P is >= 8 below the tier's best pick is dropped. Slots are filled in the order H, P, C (L-<tier arm>);
a line without a pick leaves its slot empty (at most three finalists). Development only; never a release score.

usage: python3 m7_rules.py --tier 4b|2b --lines-root /data/dev2/runs/dec/m7/lines/<tier> \
    --rules-module <mirror>/src/training/decision2/v2/9b/lux9b/m4_rules.py --output <select>/<tier>-finalists.json \
    [--dropped L-X=reason ...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dec-m7-finalists/1"
LINES = {"4b": ("L-N7H", "L-N7P", "L-N7C"), "2b": ("L-S7H", "L-S7P", "L-S7C")}
ORDER = ("1", "1/2", "1/3")
PROXY_DROP = 8.0
_f = importlib.util.spec_from_file_location(
    "m6_finalists", Path(__file__).resolve().parents[1] / "m6" / "m6_finalists.py"
)
m6f = importlib.util.module_from_spec(_f)
_f.loader.exec_module(m6f)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m4_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_line(spec: str) -> tuple[str, dict[str, str]]:
    line, rest = spec.split(":", 1)
    return line, dict(item.split("=", 1) for item in rest.split(",") if item)


def line_rows(
    rules: Any, readout: dict[str, Any], ref: str, steps: dict[str, str]
) -> list[dict[str, Any]]:
    arms = readout["arms"]
    rows = []
    for step in ORDER:
        point = steps.get(step)
        if point is None:
            continue
        arm = arms[point]
        ok = rules.eligibility(arm, arms[ref], "H_mean")
        rows.append(
            {
                "step": step,
                "point": point,
                "T": arm["T"],
                "G": arm["T"] - arms[ref]["T"],
                "H3": arm["H_mean"],
                "H": arm["H"],
                "proxy": arm["proxy"],
                "by_type": {t: v["correct"] for t, v in arm["by_type"].items()},
                **ok,
            }
        )
    return rows


def select(
    tier: str, lines: dict[str, list[dict[str, Any]]], dropped: dict[str, str]
) -> dict[str, Any]:
    picks = {
        line: next((r for r in rows if r["eligible"]), None)
        for line, rows in lines.items()
    }
    live = [p["proxy"] for p in picks.values() if p]
    best = max(live, default=None)
    removed = {
        line: p["point"]
        for line, p in picks.items()
        if p and best is not None and p["proxy"] <= best - PROXY_DROP
    }
    finalists, not_finalists = [], []
    for line in LINES[tier]:
        if line in dropped:
            not_finalists.append({"line": line, "reason": f"dropped: {dropped[line]}"})
            continue
        pick = picks.get(line)
        if pick is None:
            reasons = sorted({x for r in lines.get(line, []) for x in r["reasons"]})
            not_finalists.append(
                {
                    "line": line,
                    "reason": "no point passes the gates",
                    "reasons": reasons,
                }
            )
        elif line in removed:
            not_finalists.append(
                {
                    "line": line,
                    "reason": "pick removed by the proxy drop rule",
                    "pick": pick["point"],
                }
            )
        else:
            finalists.append({"slot": len(finalists) + 1, "line": line, **pick})
    return {
        "picks": {line: (p["point"] if p else None) for line, p in picks.items()},
        "best_proxy": best,
        "proxy_dropped": removed,
        "finalists": finalists,
        "not_finalists": not_finalists,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=sorted(LINES), required=True)
    p.add_argument("--lines-root", type=Path, required=True)
    p.add_argument("--rules-module", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dropped",
        action="append",
        default=[],
        help="L-X=reason for a line without a readout",
    )
    a = p.parse_args(argv)
    if not a.output.name.endswith("-finalists.json"):
        p.error("--output must end with -finalists.json")
    rules = load_module(a.rules_module)
    dropped = dict(d.split("=", 1) for d in a.dropped)
    ref = f"{a.tier}-I"
    lines, readouts = {}, {}
    for line in LINES[a.tier]:
        out, spec = (
            a.lines_root / "readout" / f"{line}.json",
            a.lines_root / "readout" / f"{line}.line",
        )
        if not (out.is_file() and spec.is_file()):
            if line not in dropped:
                p.error(
                    f"{line} has no readout (run m7-lines.sh, or declare --dropped)"
                )
            continue
        name, steps = parse_line(spec.read_text().strip())
        if name != line:
            p.error(f"{spec}: names {name}, not {line}")
        doc = json.loads(out.read_text())
        lines[line] = line_rows(rules, doc, ref, steps)
        readouts[line] = {
            "file": str(out),
            "sha256": sha_file(out),
            "line": spec.read_text().strip(),
        }
    result = select(a.tier, lines, dropped)
    for f in result["finalists"]:
        f.update(m6f.point_info(a.lines_root, f["point"]))
    doc = {
        "schema": SCHEMA,
        "tier": a.tier,
        "role": "development-only selection (prereg dec-m7-prereg-2026-09-30.md); not a release or post-key score",
        "rules_module": str(a.rules_module),
        "rules_module_sha256": sha_file(a.rules_module),
        "reference": ref,
        "readouts": readouts,
        "lines": lines,
        **result,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        a.output.rename(a.output.with_name(a.output.name + ".prev"))
    a.output.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finalists": [
                    (f["slot"], f["line"], f["point"]) for f in doc["finalists"]
                ],
                "not_finalists": doc["not_finalists"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
