"""Decoder M16 development gates and finalists, per tier (prereg dec-m16-prereg-2026-10-01.md, "Development gates",
"Selection rule").

Each point <arm>-a<pct> is gated against its tier's reference read on the same node (2b / 08b: C0 = DEV2.0-<t>'s
weights; 4b: the released LH):
  1-6. M14's gates (ops/m14/m14_rules.py: M11's gate plus breadth, transfer macro delta >= 0);
  7.   the MLX-DEV guard: Noul-ML and Choice-ML 95% upper bounds >= 0 vs the reference (v2.dec.mlx_dev compare,
       <lines-root>/<point>/mlxcmp/<point>.mlxdev.json, A = the reference's readout); Score-ML and M_dev reported.
Selection: eligible = gates 1-7; rank by the transfer macro delta (rounded to 4 decimals, larger first), then an
HT-DEV v2 GAIN, then the smaller alpha. Finalist 1 = the top point; finalist 2 = the top point on a different arm line,
else the next point on the same line; at most two.
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m16_rules.py --tier 2b|08b|4b --lines-root L --readout R.json --point X=REF [...] --output OUT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "dec-m16-finalists/1"
MAX_FINALISTS = 2
GUARDED = ("noul_ml", "choice_ml")
REPORTED = ("score_ml", "m_dev")
POINT = re.compile(r"^(?P<line>.+)-a(?P<pct>25|50|75)$")
OPS = Path(__file__).resolve().parents[1]


def load(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, OPS / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_alpha(point: str) -> tuple[str, float]:
    match = POINT.match(point)
    if not match:
        raise ValueError(f"{point}: M16 points are named <arm>-a25 / -a50 / -a75")
    return match["line"], int(match["pct"]) / 100


def mlx_guard(
    lines: Path, point: str, ref: str
) -> tuple[dict[str, Any] | None, list[str]]:
    path = lines / point / "mlxcmp" / f"{point}.mlxdev.json"
    if not path.is_file():
        return None, [f"MLX-DEV: no compare of {point} against {ref}"]
    cmp = json.loads(path.read_text())
    ref_pred = lines / ref / "mlxdev" / "mlxdev-predictions.jsonl"
    if not ref_pred.is_file() or sha256(ref_pred) != cmp.get("a_sha256"):
        return None, [f"MLX-DEV: the compare of {point} is not against {ref}'s readout"]
    row = {
        k: {
            "a": cmp["metrics"][k]["a"],
            "b": cmp["metrics"][k]["b"],
            "diff": cmp["metrics"][k]["diff"],
            "ci95": cmp["metrics"][k]["ci95"],
        }
        for k in GUARDED + REPORTED
    }
    reasons = [
        f"MLX-DEV guard: {k} upper bound {row[k]['ci95'][1]:+.4f} < 0 vs {ref}"
        for k in GUARDED
        if row[k]["ci95"] is None or row[k]["ci95"][1] < 0
    ]
    return row, reasons


def rank_key(row: dict[str, Any]) -> tuple:
    delta = (row.get("ib_dev") or {}).get("transfer_delta")
    return (
        -round(delta if delta is not None else float("-inf"), 4),
        (row.get("htdev2") or {}).get("verdict") != "GAIN",
        row["alpha"],
    )


def pick(rows: list[dict[str, Any]]) -> list[str]:
    passing = sorted((r for r in rows if r["eligible"]), key=rank_key)
    if not passing:
        return []
    first = passing[0]
    rest = passing[1:]
    other = [r for r in rest if r["line"] != first["line"]]
    second = (other or rest)[:1]
    return [r["point"] for r in [first, *second]][:MAX_FINALISTS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("2b", "08b", "4b"), required=True)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True, help="X=REF")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; the rules run once")
    m14 = load("m14/m14_rules.py", "m14_rules")
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
        line, alpha = line_alpha(point)
        row = m11.gate(
            m10, m8, m8s, arms, diag, "08b" if args.tier == "08b" else "-", point, ref
        )
        row["line"], row["alpha"] = line, alpha
        row["ib_dev"], extra = m14.breadth(diag, point, ref)
        row["mlx_dev"], guard = mlx_guard(args.lines_root, point, ref)
        row["reasons"] = row["reasons"] + extra + guard
        row["eligible"] = not row["reasons"]
        rows.append(row)
    result = {
        "schema": SCHEMA,
        "tier": args.tier,
        "rule": "M14's gates (M8 eligibility, Score5t floor [m8s amendment 3 when the reference COLLAPSEs], HT-DEV v2 "
        "not FLAG, retention macro CI upper >= 0, yes-bias guard hs1-dev false-yes <= reference + 0.10, 0.8B HT-DEV v2 "
        "delta >= 0, transfer macro delta >= 0) + MLX-DEV guard (Noul-ML and Choice-ML 95% upper bounds >= 0); pick: "
        "the largest transfer macro delta (then HT-DEV v2 GAIN, then the smaller alpha); finalist 2 from a different "
        "arm line when one is eligible; at most two",
        "points": rows,
        "finalists": pick(rows),
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
