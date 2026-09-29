"""Milestone 6 development-only decision rules (incumbent-anchored, Noul-protected).

Reads ``v2.dec.dev_readout`` JSON files like ``m5_rules`` and applies:

* ``seed``: M5's seed rule (the soup if P(soup) >= the seed mean, else the primary seed).
* ``early``: the early stop at an arm's first full checkpoint. The arm's point is the
  alpha 1/3 interpolation of its first seed toward Lux 1.0; the control's point is the same
  interpolation of the matched control's same-seed run. The arm continues only if
  P(arm) - P(control) >= 1/2 and its protected screen is not below the control's (``H3``:
  the CSS-pilot three-task mean; ``RP``: typed-DEV ``rule_precedence`` correct count).
* ``alpha``: M5's incumbent-anchored alpha rule plus a Noul ``rule_precedence`` floor.
  alpha is eligible if (i) every type keeps c_t >= c_t,R - 3/100 * n_t, (ii) H3 >= H3_R,
  (iii) every family keeps F_f >= F_f,R - 1/10 and (iv) ``rule_precedence`` keeps
  c_RP >= c_RP,R - 1/100 * n_RP (equality is eligible). G = T - T_R; no pick if nothing is
  eligible or G* < 1/100, else alpha* = the smallest eligible alpha with G >= 3/4 * G*.
  The pick is dropped if P < P_R - 8. Floors and gains are exact rationals from the counts.
* ``finalists``: at most three, the non-dropped picks in the order of ``--line``.

Development readouts are never release scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from lux9b import m4_rules, m5_rules

ALPHAS = (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1))
RP_FAMILY = "rule_precedence"
RP_SLACK = Fraction(1, 100)
EARLY_MIN_GAIN = 0.5
PROTECTED = ("H3", "RP")
ROLE = "development-only M6 rule; not a release or post-key score"


def rp_count(arm: dict) -> tuple[int, int]:
    fam = arm["by_family"][RP_FAMILY]
    return fam["correct"], fam["n"]


def eligibility(arm: dict, ref: dict) -> dict:
    out = m5_rules.eligibility(arm, ref)
    got, n = rp_count(arm)
    want, n_ref = rp_count(ref)
    if n != n_ref:
        raise ValueError("rule_precedence: n differs from the reference")
    floor = want - RP_SLACK * n_ref
    if got < floor:
        out["reasons"].append(f"rule_precedence {got} < floor {float(floor):g}")
        out["eligible"] = False
    return out


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
                "rule_precedence": rp_count(arm)[0],
                "by_type": {k: v["correct"] for k, v in arm["by_type"].items()},
                **eligibility(arm, ref),
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    best = max((r["_gain"] for r in eligible), default=None)
    pick = None
    if best is not None and best >= m5_rules.MIN_GAIN:
        pick = next(r for r in eligible if r["_gain"] >= m5_rules.KEEP_SHARE * best)
    for r in rows:
        r.pop("_gain")
    floor = ref["proxy"] - m5_rules.PROXY_DROP
    dropped = pick is not None and pick["proxy"] < floor
    keys = ("alpha", "arm", "T", "G", "H3", "proxy", "rule_precedence")
    summary = None if pick is None else {k: pick[k] for k in keys}
    return {
        "ref": {
            "arm": ref_name,
            "T": float(t_ref),
            "H3": ref["H_mean"],
            "H": ref["H"],
            "proxy": ref["proxy"],
            "rule_precedence": rp_count(ref)[0],
            "by_type": {k: v["correct"] for k, v in ref["by_type"].items()},
        },
        "rows": rows,
        "G_star": None if best is None else float(best),
        "rule_pick": summary,
        "no_pick_reason": (
            None if pick else ("no eligible alpha" if best is None else "G* < 0.01")
        ),
        "proxy_floor": floor,
        "dropped": dropped,
        "pick": None if dropped else summary,
    }


def early(arms: dict, arm_name: str, control_name: str, protect: str) -> dict:
    if protect not in PROTECTED:
        raise ValueError(f"protect must be one of {PROTECTED}")
    arm, ctl = arms[arm_name], arms[control_name]
    gain = arm["proxy"] - ctl["proxy"]
    reasons = []
    if gain < EARLY_MIN_GAIN:
        reasons.append(f"P gain {gain:+.2f} < +{EARLY_MIN_GAIN}")
    if protect == "H3":
        screen = (arm["H_mean"], ctl["H_mean"])
    else:
        screen = (rp_count(arm)[0], rp_count(ctl)[0])
    if screen[0] < screen[1]:
        reasons.append(f"{protect} {screen[0]} < control {screen[1]}")
    return {
        "arm": arm_name,
        "control": control_name,
        "protect": protect,
        "P": arm["proxy"],
        "P_control": ctl["proxy"],
        "P_gain": gain,
        "screen": screen[0],
        "screen_control": screen[1],
        "T": float(m4_rules.family_macro(arm)),
        "T_control": float(m4_rules.family_macro(ctl)),
        "continue": not reasons,
        "reasons": reasons,
        "role": ROLE,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="soup vs seeds (M5 rule)")
    s.add_argument("--readout", action="append", required=True)
    s.add_argument("--soup", required=True)
    s.add_argument("--seeds", required=True)
    s.add_argument("--primary")
    e = sub.add_parser("early", help="early stop at the first full checkpoint")
    e.add_argument("--readout", action="append", required=True)
    e.add_argument("--arm", required=True)
    e.add_argument("--control", required=True)
    e.add_argument("--protect", required=True, choices=PROTECTED)
    a = sub.add_parser("alpha", help="incumbent-anchored, Noul-protected alpha rule")
    a.add_argument("--readout", action="append", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--ref", required=True)
    a.add_argument("--point", action="append", required=True, help="ALPHA=KEY")
    f = sub.add_parser("finalists", help="non-dropped line picks in priority order")
    f.add_argument("--line", action="append", required=True, help="NAME=ALPHA_JSON")
    for p in (s, e, a, f):
        p.add_argument("--output")
    args = ap.parse_args(argv)
    if args.cmd == "seed":
        arms = m4_rules.load_arms(args.readout)
        seeds = args.seeds.split(",")
        primary = args.primary or seeds[0]
        if primary not in seeds:
            ap.error("--primary must be one of --seeds")
        out = {**m4_rules.seed_rule(arms, args.soup, seeds, primary), "role": ROLE}
    elif args.cmd == "early":
        out = early(
            m4_rules.load_arms(args.readout), args.arm, args.control, args.protect
        )
    elif args.cmd == "alpha":
        arms = m4_rules.load_arms(args.readout)
        points = [m5_rules.parse_point(p) for p in args.point]
        out = {"line": args.name, **alpha_line(points, arms, args.ref), "role": ROLE}
    else:
        lines = []
        for spec in args.line:
            name, sep, path = spec.partition("=")
            if not sep:
                ap.error(f"--line {spec!r}: expected NAME=ALPHA_JSON")
            lines.append((name, json.loads(Path(path).read_text())))
        out = {**m5_rules.finalists(lines), "role": ROLE}
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
