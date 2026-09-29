"""Milestone 4 development-only decision rules (preregistration + amendments 2-3).

Reads ``v2.dec.dev_readout`` JSON files (``arms.<name>`` with ``by_type`` / ``by_family``
correct / n counts, ``T``, ``H_mean`` = CSS-pilot three-task mean macro-F1 (H3), ``H`` =
median CSS task macro-F1 and ``proxy`` = 100 * sqrt(T * H)) and applies:

* ``seed``: the coordinator seed rule. The arm artifact is the uniform soup if its proxy is
  >= the seed mean, otherwise the median seed by proxy (three or more seeds) or the primary
  seed (two seeds; amendment 3).
* ``alpha``: the alpha rule per interpolation line. alpha is eligible if (i) every decision
  type keeps c_t >= c_t,L - 0.03 * n_t, (ii) H3 >= H3_L and (iii) no family accuracy falls
  below F_f,L - 0.10. G(alpha) = T(alpha) - T_L. G* = the largest G over eligible alphas. A
  line has no pick if nothing is eligible or G* < 0.01. Otherwise alpha* = the smallest
  eligible alpha with G >= 0.75 * G*. Floors and gains are compared exactly, in rationals
  from the counts. The proxy drop rule removes a line pick whose proxy is >= 8 below the
  best pick's.

Development readouts are never release scores.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

TYPE_SLACK = Fraction(3, 100)
FAMILY_SLACK = Fraction(1, 10)
MIN_GAIN = Fraction(1, 100)
KEEP_SHARE = Fraction(3, 4)
PROXY_DROP = 8.0


def load_arms(paths: list[str]) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for name, arm in data["arms"].items():
            if name in arms and arms[name] != arm:
                raise ValueError(f"arm {name} differs between readout files")
            arms[name] = arm
    return arms


def family_macro(arm: dict) -> Fraction:
    fams = arm["by_family"].values()
    return sum(Fraction(f["correct"], f["n"]) for f in fams) / len(arm["by_family"])


def eligibility(arm: dict, lux: dict, h_field: str) -> dict:
    reasons = []
    for t, ref in lux["by_type"].items():
        got = arm["by_type"][t]
        if got["n"] != ref["n"]:
            raise ValueError(f"type {t}: n differs from the reference")
        floor = ref["correct"] - TYPE_SLACK * ref["n"]
        if got["correct"] < floor:
            reasons.append(f"type {t} {got['correct']} < floor {float(floor):g}")
    if arm[h_field] < lux[h_field]:
        reasons.append(f"{h_field} {arm[h_field]:.4f} < reference {lux[h_field]:.4f}")
    for f, ref in lux["by_family"].items():
        got = arm["by_family"][f]
        if got["n"] != ref["n"]:
            raise ValueError(f"family {f}: n differs from the reference")
        if (
            Fraction(got["correct"], got["n"])
            < Fraction(ref["correct"], ref["n"]) - FAMILY_SLACK
        ):
            reasons.append(
                f"family {f} {got['correct']}/{got['n']} below reference - 0.10"
            )
    return {"eligible": not reasons, "reasons": reasons}


def alpha_line(
    points: list[tuple[Fraction, str]], arms: dict, lux_name: str, h_field: str
) -> dict:
    lux = arms[lux_name]
    t_ref = family_macro(lux)
    rows = []
    for alpha, name in sorted(points):
        arm = arms[name]
        gain = family_macro(arm) - t_ref
        rows.append(
            {
                "alpha": str(alpha),
                "arm": name,
                "T": float(family_macro(arm)),
                "G": float(gain),
                "_gain": gain,
                "H3": arm["H_mean"],
                "H": arm["H"],
                "proxy": arm["proxy"],
                "by_type": {t: v["correct"] for t, v in arm["by_type"].items()},
                **eligibility(arm, lux, h_field),
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    best = max((r["_gain"] for r in eligible), default=None)
    pick = None
    if best is not None and best >= MIN_GAIN:
        pick = next(r for r in eligible if r["_gain"] >= KEEP_SHARE * best)
    for r in rows:
        r.pop("_gain")
    return {
        "rows": rows,
        "G_star": None if best is None else float(best),
        "pick": (
            None
            if pick is None
            else {k: pick[k] for k in ("alpha", "arm", "T", "G", "H3", "proxy")}
        ),
        "no_pick_reason": (
            None if pick else ("no eligible alpha" if best is None else "G* < 0.01")
        ),
    }


def proxy_drop(picks: dict[str, dict | None]) -> dict:
    live = {line: p for line, p in picks.items() if p}
    if not live:
        return {"best_proxy": None, "kept": [], "dropped": []}
    best = max(p["proxy"] for p in live.values())
    dropped = sorted(
        line for line, p in live.items() if best - p["proxy"] >= PROXY_DROP
    )
    return {
        "best_proxy": best,
        "kept": [line for line in live if line not in dropped],
        "dropped": dropped,
    }


def seed_rule(arms: dict, soup: str, seeds: list[str], primary: str) -> dict:
    proxies = {s: arms[s]["proxy"] for s in seeds}
    mean = statistics.fmean(proxies.values())
    if arms[soup]["proxy"] >= mean:
        choice, why = soup, "soup proxy >= seed mean"
    elif len(seeds) >= 3:
        ordered = sorted(seeds, key=lambda s: (proxies[s], s))
        choice, why = (
            ordered[(len(ordered) - 1) // 2],
            "median seed (soup proxy < seed mean)",
        )
    else:
        choice, why = primary, "primary seed (two seeds, soup proxy < seed mean)"
    return {
        "soup": soup,
        "soup_proxy": arms[soup]["proxy"],
        "seed_proxies": proxies,
        "seed_mean": mean,
        "artifact": choice,
        "reason": why,
    }


def parse_line(spec: str) -> tuple[str, list[tuple[Fraction, str]]]:
    name, _, rest = spec.partition(":")
    points = []
    for item in rest.split(","):
        alpha, _, arm = item.partition("=")
        points.append((Fraction(alpha), arm))
    return name, points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="soup vs seeds")
    s.add_argument("--readout", action="append", required=True)
    s.add_argument("--soup", required=True)
    s.add_argument("--seeds", required=True, help="comma-separated seed arm names")
    s.add_argument("--primary", required=True)
    a = sub.add_parser("alpha", help="alpha rule per line + proxy drop rule")
    a.add_argument("--readout", action="append", required=True)
    a.add_argument("--lux", required=True, help="Lux reference arm name")
    a.add_argument(
        "--line", action="append", required=True, help="NAME:1/4=arm,1/2=arm,..."
    )
    a.add_argument(
        "--h-field", default="H_mean", help="human-transfer floor field (default H3)"
    )
    for p in (s, a):
        p.add_argument("--output")
    args = ap.parse_args(argv)
    arms = load_arms(args.readout)
    if args.cmd == "seed":
        seeds = args.seeds.split(",")
        if args.primary not in seeds:
            ap.error("--primary must be one of --seeds")
        out = seed_rule(arms, args.soup, seeds, args.primary)
    else:
        lines = {}
        for spec in args.line:
            name, points = parse_line(spec)
            lines[name] = alpha_line(points, arms, args.lux, args.h_field)
        out = {
            "lux": args.lux,
            "h_field": args.h_field,
            "lines": lines,
            "proxy_drop": proxy_drop({n: v["pick"] for n, v in lines.items()}),
            "role": "development-only alpha rule; not a release or post-key score",
        }
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
