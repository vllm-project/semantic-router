"""Decoder M6 finalists per tier (prereg dec-m6-prereg-2026-09-29.md, "Selection rule" and the slot table).

Runs the 9B rule module `v2/9b/lux9b/m4_rules.py alpha` unchanged (reference `--lux <tier>-I`,
`--h-field H_mean`) over the tier's line readouts written by m6-lines.sh (`readout/L-<LINE>.json` +
`readout/L-<LINE>.line`), then fills the slots in priority order:

- 4B: L-N6D, L-N6A, then the better of the L-N5BN / L-Nox picks (higher G, then higher P), then the other; with
  the PN1 amendment an L-N6P line (present only if its readout exists) takes slot 3 ahead of them.
- 2B: L-S6X, L-S6D, L-Sol.
- 0.8B: L-E6K, L-Eos (two slots).

A line without a pick, or whose pick the proxy drop rule removed, passes its slot to the next line in that order;
at most three finalists (two for 0.8B). Every preregistered line must have a readout or be declared dropped
(`--dropped L-X=reason`, e.g. an arm stopped by a stop rule). Development only; never a release score.

usage: python3 m6_finalists.py --tier 4b|2b|08b --lines-root /data/dev2/runs/dec/m6/lines/<tier> \
    --rules <mirror>/src/training/decision2/v2/9b/lux9b/m4_rules.py --output <select>/<tier>-finalists.json \
    [--dropped L-X=reason ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dec-m6-finalists/1"
TIERS: dict[str, dict[str, Any]] = {
    "4b": {
        "lines": ["L-N6D", "L-N6A", "L-N5BN", "L-Nox"],
        "optional": ["L-N6P"],
        "slots": 3,
    },
    "2b": {"lines": ["L-S6X", "L-S6D", "L-Sol"], "optional": [], "slots": 3},
    "08b": {"lines": ["L-E6K", "L-Eos"], "optional": [], "slots": 2},
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_pick(rules: dict[str, Any], line: str) -> dict[str, Any] | None:
    entry = rules["lines"].get(line)
    if entry is None or entry["pick"] is None:
        return None
    if line in rules["proxy_drop"]["dropped"]:
        return None
    return entry["pick"]


def priority(tier: str, rules: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Line order in which slots are filled, and how the 4B slot-3 tie was broken."""
    if tier == "4b":
        pair = ["L-N5BN", "L-Nox"]
        picks = {line: live_pick(rules, line) for line in pair}
        live = [line for line in pair if picks[line] is not None]
        ordered = sorted(
            live, key=lambda l: (-picks[l]["G"], -picks[l]["proxy"], pair.index(l))
        )
        ordered += [line for line in pair if line not in ordered]
        order = ["L-N6D", "L-N6A"]
        if "L-N6P" in rules["lines"]:
            order.append("L-N6P")
        note = {
            "slot3_candidates": pair,
            "better": ordered[0] if live else None,
            "by": (
                "higher G, then higher P"
                if len(live) == 2
                else ("only live pick" if live else "no live pick")
            ),
            "pn1_line": "L-N6P" in rules["lines"],
        }
        return order + ordered, note
    return list(TIERS[tier]["lines"]), {}


def select(
    tier: str, rules: dict[str, Any], points: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    order, note = priority(tier, rules)
    finalists, passed = [], []
    for line in order:
        entry = rules["lines"].get(line)
        pick = live_pick(rules, line)
        if pick is None:
            if entry is None:
                why = "line not read out (dropped)"
            elif entry["pick"] is None:
                why = f"no pick: {entry['no_pick_reason']}"
            else:
                why = "pick removed by the proxy drop rule"
            passed.append({"line": line, "reason": why})
            continue
        if len(finalists) >= TIERS[tier]["slots"]:
            passed.append({"line": line, "reason": "slots full", "pick": pick["arm"]})
            continue
        info = points.get(pick["arm"], {})
        finalists.append(
            {
                "slot": len(finalists) + 1,
                "line": line,
                "point": pick["arm"],
                "step": pick["alpha"],
                "T": pick["T"],
                "G": pick["G"],
                "H3": pick["H3"],
                "proxy": pick["proxy"],
                **info,
            }
        )
    return {
        "order": order,
        "slot_note": note,
        "finalists": finalists,
        "not_finalists": passed,
    }


def point_info(lines_root: Path, point: str) -> dict[str, Any]:
    w = lines_root / point / "weights.json"
    if not w.is_file():
        raise SystemExit(f"{point}: no weights.json under {lines_root}")
    doc = json.loads(w.read_text())
    return {
        "checkpoint": doc["checkpoint"],
        "effective_weights": doc["effective_weights"],
        "weights_json": str(w),
        "weights_json_sha256": sha_file(w),
        "files_sha256_list": doc["files_sha256_list"],
        "files_sha256_list_sha256": doc["files_sha256_list_sha256"],
        "typed_dev_predictions": str(
            lines_root / point / "dev" / "dev.predictions.jsonl"
        ),
        "css_pilot_predictions": str(
            lines_root / point / "css-pilot" / "css-pilot.predictions.jsonl"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=sorted(TIERS), required=True)
    p.add_argument("--lines-root", type=Path, required=True)
    p.add_argument(
        "--rules",
        type=Path,
        required=True,
        help="v2/9b/lux9b/m4_rules.py of the mirror",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dropped",
        action="append",
        default=[],
        help="L-X=reason for a line with no readout",
    )
    args = p.parse_args(argv)
    cfg = TIERS[args.tier]
    readout = args.lines_root / "readout"
    dropped = dict(d.split("=", 1) for d in args.dropped)
    unknown = set(dropped) - set(cfg["lines"]) - set(cfg["optional"])
    if unknown:
        p.error(f"--dropped names lines outside the tier: {sorted(unknown)}")
    used, specs, missing = [], [], []
    for line in cfg["lines"] + cfg["optional"]:
        files = readout / f"{line}.json", readout / f"{line}.line"
        if all(f.is_file() for f in files):
            if line in dropped:
                p.error(f"{line} has a readout but is declared dropped")
            used.append(line)
            specs.append(files[1].read_text().strip())
        elif line in cfg["lines"] and line not in dropped:
            missing.append(line)
    if missing:
        p.error(
            f"lines without a readout (run m6-lines.sh, or declare --dropped): {missing}"
        )
    if not used:
        p.error("no line has a readout")
    rules_out = args.output.with_name(
        args.output.name.replace("-finalists.json", "-rules.json")
    )
    if rules_out == args.output:
        p.error("--output must end with -finalists.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-B",
        str(args.rules),
        "alpha",
        "--lux",
        f"{args.tier}-I",
        "--h-field",
        "H_mean",
        "--output",
        str(rules_out),
    ]
    for line, spec in zip(used, specs):
        if not spec.startswith(f"{line}:"):
            raise SystemExit(f"{line}: line spec {spec!r} does not name the line")
        cmd += ["--readout", str(readout / f"{line}.json"), "--line", spec]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    rules = json.loads(rules_out.read_text())
    points = {}
    for line in used:
        pick = rules["lines"][line]["pick"]
        if pick is not None:
            points[pick["arm"]] = point_info(args.lines_root, pick["arm"])
    result = {
        "schema": SCHEMA,
        "tier": args.tier,
        "role": "development-only selection (prereg dec-m6-prereg-2026-09-29.md); not a release or post-key score",
        "rules_module": str(args.rules),
        "rules_module_sha256": sha_file(args.rules),
        "rules_command": cmd,
        "rules_output": str(rules_out),
        "rules_output_sha256": sha_file(rules_out),
        "readouts": {
            line: {
                "file": str(readout / f"{line}.json"),
                "sha256": sha_file(readout / f"{line}.json"),
                "line": spec,
            }
            for line, spec in zip(used, specs)
        },
        "dropped_lines": dropped,
        "lines": {
            line: {
                "G_star": rules["lines"][line]["G_star"],
                "pick": rules["lines"][line]["pick"],
                "no_pick_reason": rules["lines"][line]["no_pick_reason"],
                "proxy_dropped": line in rules["proxy_drop"]["dropped"],
            }
            for line in used
        },
        "proxy_drop": rules["proxy_drop"],
        **select(args.tier, rules, points),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.rename(args.output.with_name(args.output.name + ".prev"))
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "tier": args.tier,
                "finalists": [
                    (f["slot"], f["line"], f["point"]) for f in result["finalists"]
                ],
                "not_finalists": result["not_finalists"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
