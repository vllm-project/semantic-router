"""Decoder M14 development gates and finalists, per tier (prereg dec-m14-prereg-2026-10-01.md, "Development gates").

Each point X is gated against its tier's reference read on the same node and cache (2b / 08b: C0 = DEV2.0-<t>'s
weights; 4b: the released LH):
  1-5. M11's gate (ops/m11/m11_rules.py gate, without its stage-2 breadth rule): M8 eligibility (type / family / Noul
       floors, Score5-typed-DEV with M8s amendment 3's floor when the reference's check half is COLLAPSE, HT-DEV v2 not
       FLAG), retention-probe macro delta CI upper bound >= 0, the hs1-dev yes-bias guard (false-yes <= reference +
       0.10) and, at 0.8B, HT-DEV v2 delta >= 0;
  6.   breadth (every tier): the transfer macro delta (the 9 IB DEV families outside the in-distribution ones) vs the
       reference >= 0 (m11_ibdev.py score, diag/<point>.ibdev.json). The 12-family IB DEV macro is reported.
Finalists (<= 2 per tier): passing points with an HT-DEV v2 GAIN first, then the larger transfer macro delta.
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m14_rules.py --tier 2b|08b|4b --lines-root L --readout R.json --point X=REF [...] [--contrast A:B ...] --output OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

SCHEMA = "dec-m14-finalists/1"
MAX_FINALISTS = 2
OPS = Path(__file__).resolve().parents[1]


def load(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, OPS / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def breadth(
    diag: Path, point: str, ref: str
) -> tuple[dict[str, Any] | None, list[str]]:
    path = diag / f"{point}.ibdev.json"
    ib = json.loads(path.read_text()) if path.is_file() else None
    if ib is None or ib.get("reference_name") != ref:
        return None, [f"breadth: no IB DEV score of {point} against {ref}"]
    row = {
        "macro": ib["ib_dev"]["family_macro"],
        "delta": ib["ib_dev"].get("delta_family_macro"),
        "ci95": ib["ib_dev"].get("paired", {}).get("ci95"),
        "transfer_macro": ib["transfer"]["family_macro"],
        "transfer_delta": ib["transfer"].get("delta_family_macro"),
        "transfer_ci95": ib["transfer"].get("paired", {}).get("ci95"),
    }
    if row["transfer_delta"] is None or row["transfer_delta"] < 0:
        return row, [
            f"breadth: transfer macro delta {row['transfer_delta']} < 0 vs {ref}"
        ]
    return row, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("2b", "08b", "4b"), required=True)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True, help="X=REF")
    parser.add_argument("--contrast", action="append", default=[], help="A:B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; the rules run once")
    m11 = load("m11/m11_rules.py", "m11_rules")
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
        # M11's 4b branch is its stage-2 breadth rule (12-family macro); M13 and M14 gate on the transfer macro instead
        row = m11.gate(
            m10, m8, m8s, arms, diag, "08b" if args.tier == "08b" else "-", point, ref
        )
        row["ib_dev"], extra = breadth(diag, point, ref)
        row["reasons"] = row["reasons"] + extra
        row["eligible"] = not row["reasons"]
        rows.append(row)
    passing = [r for r in rows if r["eligible"]]
    passing.sort(
        key=lambda r: (
            (r["htdev2"] or {}).get("verdict") != "GAIN",
            -((r["ib_dev"] or {}).get("transfer_delta") or 0.0),
        )
    )
    result = {
        "schema": SCHEMA,
        "tier": args.tier,
        "rule": "M11 gate (M8 eligibility, Score5t floor [m8s amendment 3 when the reference COLLAPSEs], HT-DEV v2 not "
        "FLAG, retention macro CI upper >= 0, yes-bias guard hs1-dev false-yes <= reference + 0.10, 0.8B HT-DEV v2 "
        "delta >= 0) + breadth (transfer macro delta >= 0); pick: HT-DEV v2 GAIN first, then the larger transfer "
        "macro delta; at most two",
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
