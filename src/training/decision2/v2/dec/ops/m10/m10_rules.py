"""Decoder M10 development gates and finalists (prereg dec-m10-prereg-2026-10-01.md, "Development gates").

Each arm soup X is gated against its reference R (C0 = DEV2.0-4B's weights read on the same M10 path):
  1-4. M8's eligibility (ops/m8/m8_rules.py): typed type / family floors, the Noul `rule_precedence` floor, the
       Score5-typed-DEV check-half floor, HT-DEV v2 not FLAG (delta > -0.02);
  5.   retention: the probe macro (MMLU, ARC, GSM8K) delta vs R with a paired 95% CI whose upper bound is >= 0.
Finalists (<= 2): passing points with an HT-DEV v2 GAIN first, then the larger retention-macro delta.
Contrasts (report only): any --contrast A:B pair's HT-DEV v2 / probe / typed differences (e.g. 4b-LT2:4b-LH).
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m10_rules.py --lines-root L --readout R.json --point X=REF [--point ...] [--contrast A:B ...] --output OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

SCHEMA = "dec-m10-finalists/1"
MAX_FINALISTS = 2


def load_m8_rules() -> Any:
    path = Path(__file__).resolve().parents[1] / "m8" / "m8_rules.py"
    spec = importlib.util.spec_from_file_location("m8_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_opt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def gate(
    m8: Any, arms: dict[str, Any], diag: Path, point: str, ref: str
) -> dict[str, Any]:
    htdev2 = load_opt(diag / f"{point}.htdev2.json")
    s5, s5_ref = load_opt(diag / f"{point}.score5t.json"), load_opt(
        diag / f"{ref}.score5t.json"
    )
    probes = load_opt(diag / f"{point}.probes.json")
    result = m8.eligibility(arms[point], arms[ref], htdev2, s5, s5_ref)
    reasons = list(result["reasons"])
    if probes is None or "macro_delta_ci95" not in probes:
        reasons.append("retention probes readout missing")
    elif probes["macro_delta_ci95"][1] < 0:
        reasons.append(
            f"retention: probe macro delta {probes['macro_delta']:+.4f} "
            f"CI {probes['macro_delta_ci95']} below 0"
        )
    return {
        "point": point,
        "reference": ref,
        "T": arms[point]["T"],
        "T_ref": arms[ref]["T"],
        "H3": arms[point]["H_mean"],
        "proxy": arms[point]["proxy"],
        "by_type": {t: v["correct"] for t, v in arms[point]["by_type"].items()},
        "rule_precedence": arms[point]["by_family"][m8.NOUL_FAMILY]["correct"],
        "htdev2": (
            None
            if htdev2 is None
            else {k: htdev2[k] for k in ("delta", "ci95", "verdict")}
        ),
        "score5t_check_flags": None if s5 is None else s5["check"]["flags"],
        "retention": (
            None
            if probes is None
            else {
                k: probes.get(k)
                for k in ("macro_mmlu_arc_gsm8k", "macro_delta", "macro_delta_ci95")
            }
            | {
                p: probes[p]["accuracy"]
                for p in ("mmlu", "arc", "gsm8k")
                if p in probes
            }
        ),
        "eligible": not reasons,
        "reasons": reasons,
    }


def contrast(arms: dict[str, Any], diag: Path, a: str, b: str) -> dict[str, Any]:
    htdev2 = load_opt(diag / f"{a}-vs-{b}.htdev2.json")
    probes = load_opt(diag / f"{a}-vs-{b}.probes.json")
    return {
        "left": a,
        "right": b,
        "typed_T_delta": arms[a]["T"] - arms[b]["T"],
        "by_type_delta": {
            t: arms[a]["by_type"][t]["correct"] - arms[b]["by_type"][t]["correct"]
            for t in arms[a]["by_type"]
        },
        "htdev2": (
            None
            if htdev2 is None
            else {k: htdev2[k] for k in ("delta", "ci95", "verdict")}
        ),
        "retention": (
            None
            if probes is None
            else {k: probes.get(k) for k in ("macro_delta", "macro_delta_ci95")}
            | {
                p: {k: probes[p].get(k) for k in ("delta", "delta_ci95")}
                for p in ("mmlu", "arc", "gsm8k")
                if p in probes
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True, help="X=REF")
    parser.add_argument("--contrast", action="append", default=[], help="A:B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; the rules run once")
    m8 = load_m8_rules()
    arms = json.loads(args.readout.read_text())["arms"]
    diag = args.lines_root / "diag"
    rows = [gate(m8, arms, diag, *spec.split("=", 1)) for spec in args.point]
    passing = [r for r in rows if r["eligible"]]
    passing.sort(
        key=lambda r: (
            (r["htdev2"] or {}).get("verdict") != "GAIN",
            -(r["retention"] or {}).get("macro_delta", 0.0),
        )
    )
    result = {
        "schema": SCHEMA,
        "rule": "M8 eligibility (type / family / Noul / Score5t / HT-DEV v2 not FLAG) + retention macro CI upper >= 0; "
        "pick: HT-DEV v2 GAIN first, then the larger retention-macro delta; at most two",
        "points": rows,
        "finalists": [r["point"] for r in passing[:MAX_FINALISTS]],
        "contrasts": [contrast(arms, diag, *c.split(":", 1)) for c in args.contrast],
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
