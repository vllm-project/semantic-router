"""9B M9 development gates and finalists (prereg lux9b-m9-prereg-2026-10-01.md, "Development gates").

Each arm soup X is gated against C0 (DEV2.0-9B's weights read on the same M9 path):
  1-4. the decoder M8 eligibility (v2/dec/ops/m8/m8_rules.py): typed type / family floors, the Noul
       `rule_precedence` floor, the Score5-typed-DEV check-half floor, HT-DEV v2 not FLAG (delta > -0.02);
  5.   yes-bias guard: Y1 PN1-dev clean gold-no yes-rate delta <= +0.02 and its paired 95% CI lower bound <= 0;
       Y2 PN1-dev hop yes-rate delta >= -0.03; Y3 hs1-dev unmet-condition false-yes rate delta <= +0.05;
  6.   multilingual: MLX-DEV-9B Noul-ML and Choice-ML paired 95% upper bounds >= 0;
  7.   retention: the probe macro (MMLU, ARC, GSM8K) delta with a paired 95% CI upper bound >= 0.
Finalists (<= 2): every passing point; beyond two, an HT-DEV v2 GAIN first, then the larger typed-DEV T.
Contrasts (report only): HT-DEV v2, probes, PN1 and typed differences of any --contrast A:B pair.
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m9_rules.py --lines-root L --readout R.json --point X=REF [--point ...] [--contrast A:B ...] --output OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "lux9b-m9-finalists/1"
MAX_FINALISTS = 2
Y1_MARGIN = Fraction(2, 100)
Y2_SLACK = Fraction(3, 100)
Y3_MARGIN = Fraction(5, 100)
HS1_FAMILY = "hs1_unmet_condition"


def load_m8_rules() -> Any:
    path = Path(__file__).resolve().parents[2] / "dec" / "ops" / "m8" / "m8_rules.py"
    spec = importlib.util.spec_from_file_location("m8_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_opt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def frac(value: float) -> Fraction:
    return Fraction(str(value))


def yes_bias(
    pn1: dict[str, Any] | None, hs1: dict[str, Any] | None, point: str, ref: str
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    out: dict[str, Any] = {}
    if pn1 is None:
        reasons.append("PN1 dev readout missing")
    else:
        d_no, d_hop = pn1["delta"]["clean_no"], pn1["delta"]["hop"]
        lo_no = pn1["delta_ci95"]["clean_no"][0]
        out["pn1"] = {
            "clean_no_delta": d_no,
            "clean_no_ci95": pn1["delta_ci95"]["clean_no"],
            "hop_delta": d_hop,
            "hop_ci95": pn1["delta_ci95"]["hop"],
            "pawsx6_delta": pn1["delta"]["pawsx6"],
            "all8_delta": pn1["delta"]["all8"],
            "clean_no": pn1["candidate"]["clean_no"]["yes"],
            "clean_no_ref": pn1["reference_summary"]["clean_no"]["yes"],
            "hop": pn1["candidate"]["hop"]["yes"],
            "hop_ref": pn1["reference_summary"]["hop"]["yes"],
        }
        if frac(d_no) > Y1_MARGIN:
            reasons.append(f"Y1: PN1 clean gold-no yes {d_no:+.4f} > +0.02")
        if lo_no > 0:
            reasons.append(f"Y1: PN1 clean gold-no yes CI lower {lo_no:+.4f} > 0")
        if frac(d_hop) < -Y2_SLACK:
            reasons.append(f"Y2: PN1 hop yes {d_hop:+.4f} < -0.03")
    fam = None if hs1 is None else hs1["families"].get(HS1_FAMILY)
    if fam is None:
        reasons.append("hs1-dev readout missing")
    else:
        mine = fam["diagnostics"][point].get("f3_false_yes_rate")
        theirs = fam["diagnostics"][ref].get("f3_false_yes_rate")
        out["hs1_false_yes"] = {"point": mine, "ref": theirs}
        if mine is None or theirs is None:
            reasons.append("hs1-dev false-yes rate missing")
        elif frac(mine) - frac(theirs) > Y3_MARGIN:
            reasons.append(f"Y3: hs1-dev false-yes {mine:.4f} - {theirs:.4f} > +0.05")
    return out, reasons


def multilingual(mlx: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if mlx is None:
        return {}, ["MLX-DEV-9B readout missing"]
    reasons = []
    out = {}
    for key in ("noul_ml", "choice_ml", "score_ml"):
        metric = mlx["metrics"].get(key)
        if metric is None:
            continue
        out[key] = {"diff": metric["diff"], "ci95": metric["ci95"]}
        if key != "score_ml" and metric["ci95"][1] < 0:
            reasons.append(f"MLX-DEV-9B {key} upper {metric['ci95'][1]:+.4f} < 0")
    return out, reasons


def gate(
    m8: Any, arms: dict[str, Any], diag: Path, point: str, ref: str
) -> dict[str, Any]:
    htdev2 = load_opt(diag / f"{point}.htdev2.json")
    s5, s5_ref = load_opt(diag / f"{point}.score5t.json"), load_opt(
        diag / f"{ref}.score5t.json"
    )
    probes = load_opt(diag / f"{point}.probes.json")
    result = m8.eligibility(arms[point], arms[ref], htdev2, s5, s5_ref)
    reasons = [r.replace("4b-I has none", "C0 has none") for r in result["reasons"]]
    bias, bias_reasons = yes_bias(
        load_opt(diag / f"{point}.pn1.json"),
        load_opt(diag / f"{point}.hs1.json"),
        point,
        ref,
    )
    mlx, mlx_reasons = multilingual(load_opt(diag / f"{point}.mlxdev.json"))
    reasons += bias_reasons + mlx_reasons
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
        "by_type": {t: v["correct"] for t, v in arms[point]["by_type"].items()},
        "by_type_ref": {t: v["correct"] for t, v in arms[ref]["by_type"].items()},
        "rule_precedence": arms[point]["by_family"][m8.NOUL_FAMILY]["correct"],
        "rule_precedence_ref": arms[ref]["by_family"][m8.NOUL_FAMILY]["correct"],
        "htdev2": (
            None
            if htdev2 is None
            else {k: htdev2[k] for k in ("delta", "ci95", "verdict")}
        ),
        "score5t_check_flags": None if s5 is None else s5["check"]["flags"],
        "yes_bias": bias,
        "mlxdev": mlx,
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
    pn1 = load_opt(diag / f"{a}-vs-{b}.pn1.json")
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
        ),
        "pn1": (
            None if pn1 is None else {"delta": pn1["delta"], "ci95": pn1["delta_ci95"]}
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
    passing.sort(key=lambda r: ((r["htdev2"] or {}).get("verdict") != "GAIN", -r["T"]))
    result = {
        "schema": SCHEMA,
        "rule": "decoder M8 eligibility (type / family / Noul / Score5t / HT-DEV v2 not FLAG) + yes-bias guard "
        "(Y1 PN1 clean gold-no delta <= +0.02 and CI lower <= 0; Y2 hop delta >= -0.03; Y3 hs1-dev false-yes "
        "delta <= +0.05) + MLX-DEV-9B Noul / Choice upper >= 0 + retention macro CI upper >= 0; at most two "
        "finalists, an HT-DEV v2 GAIN first, then the larger typed-DEV T",
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
