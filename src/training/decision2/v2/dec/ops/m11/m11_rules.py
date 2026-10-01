"""Decoder M11 development gates and finalists, per tier (prereg dec-m11-prereg-2026-10-01.md, "Development gates").

Each point X is gated against its tier's C0 (DEV2.0-<t>'s weights) read on the same node and cache:
  1-3. M10's gate (ops/m10/m10_rules.py): M8 eligibility (type / family / Noul floors, Score5-typed-DEV, HT-DEV v2 not
       FLAG) and the retention-probe macro delta CI upper bound >= 0. When C0's own Score5-typed-DEV check half is
       COLLAPSE (0.8B), the Score floor is M8s amendment 3's (ops/m8s/m8s_rules.py score_floor) instead;
  4.   yes-bias guard: the hs1-dev false-yes rate on unmet conditions not above C0's by more than 0.10;
  5.   0.8B only: HT-DEV v2 delta >= 0 (point estimate).
Finalists (<= 2 per tier): passing points with an HT-DEV v2 GAIN first, then the larger retention-macro delta.
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m11_rules.py --tier 2b|08b --lines-root L --readout R.json --point X=REF [...] [--contrast A:B ...] --output OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

SCHEMA = "dec-m11-finalists/1"
MAX_FINALISTS = 2
YES_BIAS_TOL = 0.10
YES_FAMILY = "hs1_unmet_condition"
OPS = Path(__file__).resolve().parents[1]


def load(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, OPS / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_opt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def false_yes(hs1: dict[str, Any] | None, name: str) -> float | None:
    if hs1 is None:
        return None
    cell = hs1.get("families", {}).get(YES_FAMILY, {})
    return cell.get("diagnostics", {}).get(name, {}).get("f3_false_yes_rate")


def gate(
    m10: Any,
    m8: Any,
    m8s: Any,
    arms: dict[str, Any],
    diag: Path,
    tier: str,
    point: str,
    ref: str,
) -> dict[str, Any]:
    row = m10.gate(m8, arms, diag, point, ref)
    reasons = list(row["reasons"])
    s5, s5_ref = load_opt(diag / f"{point}.score5t.json"), load_opt(
        diag / f"{ref}.score5t.json"
    )
    if s5 is not None and s5_ref is not None:
        if "COLLAPSE" in m8s.score5t_flags(s5_ref):
            reasons = [r for r in reasons if not r.startswith("Score floor")]
            reasons += [f"Score floor: {r}" for r in m8s.score_floor(s5, s5_ref)]
            row["score_floor_rule"] = "m8s amendment 3 (C0 check half COLLAPSE)"
        else:
            row["score_floor_rule"] = "m8 (no COLLAPSE; no WARN unless C0 has WARN)"
    hs1 = load_opt(diag / f"{point}.hs1.json")
    fy, fy_ref = false_yes(hs1, point), false_yes(hs1, ref)
    row["false_yes"] = {"point": fy, "reference": fy_ref}
    if fy is None or fy_ref is None:
        reasons.append("yes-bias guard: hs1-dev false-yes rate missing")
    elif fy > fy_ref + YES_BIAS_TOL:
        reasons.append(
            f"yes-bias guard: hs1-dev false-yes {fy:.3f} > C0 {fy_ref:.3f} + {YES_BIAS_TOL}"
        )
    if tier == "08b":
        ht = row["htdev2"]
        if ht is not None and ht["delta"] < 0:
            reasons.append(
                f"0.8B rule: HT-DEV v2 delta {ht['delta']:+.4f} < 0 (no development-only win)"
            )
    row["reasons"] = reasons
    row["eligible"] = not reasons
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("2b", "08b"), required=True)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True, help="X=REF")
    parser.add_argument("--contrast", action="append", default=[], help="A:B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; the rules run once")
    m10 = load("m10/m10_rules.py", "m10_rules")
    m8 = load("m8/m8_rules.py", "m8_rules")
    m8s = load("m8s/m8s_rules.py", "m8s_rules")
    arms = json.loads(args.readout.read_text())["arms"]
    diag = args.lines_root / "diag"
    rows = []
    for spec in args.point:
        point, ref = spec.split("=", 1)
        if not point.startswith(args.tier + "-") or not ref.startswith(args.tier + "-"):
            raise ValueError(f"{spec}: points and references must be tier {args.tier}")
        rows.append(gate(m10, m8, m8s, arms, diag, args.tier, point, ref))
    passing = [r for r in rows if r["eligible"]]
    passing.sort(
        key=lambda r: (
            (r["htdev2"] or {}).get("verdict") != "GAIN",
            -(r["retention"] or {}).get("macro_delta", 0.0),
        )
    )
    result = {
        "schema": SCHEMA,
        "tier": args.tier,
        "rule": "M10 gate (M8 eligibility, Score5t floor [m8s amendment 3 when C0 COLLAPSEs], HT-DEV v2 not FLAG, "
        "retention macro CI upper >= 0) + yes-bias guard (hs1-dev false-yes <= C0 + 0.10) + 0.8B HT-DEV v2 delta >= 0; "
        "pick: HT-DEV v2 GAIN first, then the larger retention-macro delta; at most two",
        "points": rows,
        "finalists": [r["point"] for r in passing[:MAX_FINALISTS]],
        "contrasts": [
            m10.contrast(arms, diag, *c.split(":", 1)) for c in args.contrast
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "finalists": result["finalists"],
                "points": {r["point"]: r["reasons"] for r in rows},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
