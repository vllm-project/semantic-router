"""Milestone 5 development-only decision rules (incumbent-anchored alpha rule).

Reads ``v2.dec.dev_readout`` JSON files like ``m4_rules`` (``arms.<name>`` with
``by_type`` / ``by_family`` correct / n counts, ``H_mean`` = H3, ``H`` and ``proxy`` =
P = 100 * sqrt(T * H)) and applies:

* ``seed``: the arm artifact is the uniform soup if P(soup) >= the seed mean, otherwise
  the median seed by P (three or more seeds) or the primary seed (two seeds), as in M4.
* ``alpha``: one interpolation line with points alpha in {1/3, 1/2, 2/3, 1} against the
  reference R (the M5 re-read of the incumbent K-a13). alpha is eligible if (i) every
  decision type keeps c_t >= c_t,R - 3/100 * n_t, (ii) H3 >= H3_R and (iii) every family
  keeps F_f >= F_f,R - 1/10 (equality is eligible). G(alpha) = T(alpha) - T_R with T the
  typed-DEV family macro. G* = the largest G over eligible alphas; no pick if nothing is
  eligible or G* < 1/100, otherwise alpha* = the smallest eligible alpha with
  G >= 3/4 * G*. Floors and gains are exact rationals from the counts. The pick is
  dropped (line-local) if P(alpha*) < P_R - 8. With ``--lux`` the M4 Lux-anchored rule
  (``m4_rules.alpha_line``) is reported alongside; it never changes the pick.
* ``finalists``: at most three finalists, the non-dropped line picks in priority order
  (the order of ``--line``, normally KD, KG, UM5).

Development readouts are never release scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from lux9b import m4_rules

ALPHAS = (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1))
TYPE_SLACK = Fraction(3, 100)
FAMILY_SLACK = Fraction(1, 10)
MIN_GAIN = Fraction(1, 100)
KEEP_SHARE = Fraction(3, 4)
PROXY_DROP = 8.0
MAX_FINALISTS = 3
ROLE = "development-only M5 rule; not a release or post-key score"


def family_value(entry: dict) -> Fraction | float:
    if "correct" in entry and "n" in entry:
        return Fraction(entry["correct"], entry["n"])
    return float(entry["accuracy"])


def eligibility(arm: dict, ref: dict) -> dict:
    reasons = []
    for t, r in ref["by_type"].items():
        got = arm["by_type"][t]
        if got["n"] != r["n"]:
            raise ValueError(f"type {t}: n differs from the reference")
        floor = r["correct"] - TYPE_SLACK * r["n"]
        if got["correct"] < floor:
            reasons.append(f"type {t} {got['correct']} < floor {float(floor):g}")
    if arm["H_mean"] < ref["H_mean"]:
        reasons.append(f"H3 {arm['H_mean']:.4f} < reference {ref['H_mean']:.4f}")
    for f, r in ref["by_family"].items():
        got = arm["by_family"][f]
        if "n" in got and "n" in r and got["n"] != r["n"]:
            raise ValueError(f"family {f}: n differs from the reference")
        if family_value(got) < family_value(r) - FAMILY_SLACK:
            reasons.append(
                f"family {f} {float(family_value(got)):.4f} < reference - 0.10"
            )
    return {"eligible": not reasons, "reasons": reasons}


def alpha_line(points: list[tuple[Fraction, str]], arms: dict, ref_name: str) -> dict:
    alphas = [a for a, _ in points]
    if not points or len(set(alphas)) != len(alphas) or set(alphas) - set(ALPHAS):
        raise ValueError(
            f"points must be distinct alphas from {[str(a) for a in ALPHAS]}"
        )
    ref = arms[ref_name]
    t_ref = m4_rules.family_macro(ref)
    rows = []
    for alpha, name in sorted(points):
        arm = arms[name]
        t = m4_rules.family_macro(arm)
        rows.append(
            {
                "alpha": str(alpha),
                "arm": name,
                "T": float(t),
                "G": float(t - t_ref),
                "_gain": t - t_ref,
                "H3": arm["H_mean"],
                "H": arm["H"],
                "proxy": arm["proxy"],
                "by_type": {k: v["correct"] for k, v in arm["by_type"].items()},
                **eligibility(arm, ref),
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    best = max((r["_gain"] for r in eligible), default=None)
    pick = None
    if best is not None and best >= MIN_GAIN:
        pick = next(r for r in eligible if r["_gain"] >= KEEP_SHARE * best)
    for r in rows:
        r.pop("_gain")
    proxy_floor = ref["proxy"] - PROXY_DROP
    dropped = pick is not None and pick["proxy"] < proxy_floor
    summary = (
        None
        if pick is None
        else {k: pick[k] for k in ("alpha", "arm", "T", "G", "H3", "proxy")}
    )
    return {
        "ref": {
            "arm": ref_name,
            "T": float(t_ref),
            "H3": ref["H_mean"],
            "H": ref["H"],
            "proxy": ref["proxy"],
            "by_type": {k: v["correct"] for k, v in ref["by_type"].items()},
        },
        "rows": rows,
        "G_star": None if best is None else float(best),
        "rule_pick": summary,
        "no_pick_reason": (
            None if pick else ("no eligible alpha" if best is None else "G* < 0.01")
        ),
        "proxy_floor": proxy_floor,
        "dropped": dropped,
        "pick": None if dropped else summary,
    }


def finalists(lines: list[tuple[str, dict]]) -> dict:
    out = []
    for name, line in lines:
        if line["pick"] is not None:
            out.append({"line": name, **line["pick"]})
    return {
        "priority": [name for name, _ in lines],
        "finalists": out[:MAX_FINALISTS],
        "no_finalist": [name for name, line in lines if line["pick"] is None],
        "role": ROLE,
    }


def parse_point(spec: str) -> tuple[Fraction, str]:
    alpha, sep, arm = spec.partition("=")
    if not sep or not arm:
        raise ValueError(f"--point {spec!r}: expected ALPHA=KEY")
    return Fraction(alpha), arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="soup vs seeds")
    s.add_argument("--readout", action="append", required=True)
    s.add_argument("--soup", required=True)
    s.add_argument("--seeds", required=True, help="comma-separated seed arm names")
    s.add_argument(
        "--primary", help="primary (-s1) seed; default: the first of --seeds"
    )
    a = sub.add_parser("alpha", help="incumbent-anchored alpha rule for one line")
    a.add_argument("--readout", action="append", required=True)
    a.add_argument("--name", required=True, help="line name, e.g. KD")
    a.add_argument("--ref", required=True, help="reference arm (M5 re-read of K-a13)")
    a.add_argument("--point", action="append", required=True, help="ALPHA=KEY")
    a.add_argument("--lux", help="Lux arm: also report the M4 Lux-anchored pick")
    f = sub.add_parser("finalists", help="non-dropped line picks in priority order")
    f.add_argument(
        "--line",
        action="append",
        required=True,
        help="NAME=ALPHA_JSON, in priority order",
    )
    for p in (s, a, f):
        p.add_argument("--output")
    args = ap.parse_args(argv)
    if args.cmd == "seed":
        arms = m4_rules.load_arms(args.readout)
        seeds = args.seeds.split(",")
        primary = args.primary or seeds[0]
        if primary not in seeds:
            ap.error("--primary must be one of --seeds")
        out = {**m4_rules.seed_rule(arms, args.soup, seeds, primary), "role": ROLE}
    elif args.cmd == "alpha":
        arms = m4_rules.load_arms(args.readout)
        points = [parse_point(p) for p in args.point]
        out = {"line": args.name, **alpha_line(points, arms, args.ref)}
        if args.lux:
            m4 = m4_rules.alpha_line(points, arms, args.lux, "H_mean")
            out["m4_lux_rule_report_only"] = {
                "lux": args.lux,
                "G_star": m4["G_star"],
                "pick": m4["pick"],
                "no_pick_reason": m4["no_pick_reason"],
                "eligible": [r["alpha"] for r in m4["rows"] if r["eligible"]],
            }
        out["role"] = ROLE
    else:
        lines = []
        for spec in args.line:
            name, sep, path = spec.partition("=")
            if not sep:
                ap.error(f"--line {spec!r}: expected NAME=ALPHA_JSON")
            lines.append((name, json.loads(Path(path).read_text())))
        out = finalists(lines)
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
