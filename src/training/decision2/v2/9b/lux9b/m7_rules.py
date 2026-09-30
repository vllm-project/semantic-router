"""Milestone 7 development-only decision rules (PN1-r2 continuation of the K seeds).

* ``pn1``: the PN1 dev readout of one checkpoint against a reference (yes = P(true) > 0.5):
  yes-rates by family and language, the true-paraphrase (``pn-hop``, gold yes) yes-rate over
  all eight languages, and the **clean gold-no yes-rate**: gold-no rows of ``pn-near`` /
  ``pn-name`` / ``pn-twin`` outside the constructions PN1-r2 dropped for label noise
  (``pn-near`` in es / fr / ar / ru / ko, Russian ``pn-name``), with a paired group bootstrap.
* ``early``: the early stop after the first member's continuations (P_1 vs its matched control
  C_1, alpha 1): P continues only if (a) its clean gold-no yes-rate is at least 0.02 below
  C_1's, (b) its hop yes-rate is not more than 0.03 below C_1's and (c) its final SELECT700
  family-macro accuracy is not more than 0.02 below C_1's.
* ``alpha``: M6's incumbent-anchored, Noul-protected alpha rule (type floors, family floors,
  ``rule_precedence`` floor, G* >= 1/100, alpha* = the smallest eligible alpha with
  G >= 3/4 G*, proxy drop) with the CSS-pilot H3 condition replaced by three development
  screens per point: HT-DEV v2 is not FLAG against the 9B reference; PN1 dev hop yes-rate
  >= R's - 0.03 and clean gold-no yes-rate <= R's; MLX-DEV-9B Noul-ML and Choice-ML paired
  95% upper bounds vs R >= 0. H3 is reported only.
* ``finalists``: at most three, the non-dropped picks in the order of ``--line``.

Development readouts are never release scores.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from lux9b import m4_rules, m5_rules, m6_rules

ROLE = "development-only M7 rule; not a release or post-key score"
PN1_NOISY = {("pn-near", lang) for lang in ("es", "fr", "ar", "ru", "ko")} | {
    ("pn-name", "ru")
}
PN1_NO_FAMILIES = ("pn-near", "pn-name", "pn-twin")
PAWSX = ("de", "es", "fr", "ja", "ko", "zh")
HOP_SLACK = 0.03
EARLY_NO_GAIN = 0.02
EARLY_SELECT_SLACK = 0.02
PN1_REPS = 2000
PN1_SEED = 20260930


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pn1_rows(gold_path: Path) -> list[dict[str, Any]]:
    rows = []
    for g in read_jsonl(gold_path):
        rows.append(
            {
                "id": g["id"],
                "group": g["group_id"],
                "language": g["language"],
                "family": g["task"].split("/", 1)[1],
                "gold": bool(g["gold"]["decision"]["value"]),
            }
        )
    return rows


def pn1_yes(path: Path) -> dict[str, bool]:
    return {
        r["id"]: float(r["answers"]["decision"]["noul"]) > 0.5 for r in read_jsonl(path)
    }


def clean_no(row: dict[str, Any]) -> bool:
    return (
        not row["gold"]
        and row["family"] in PN1_NO_FAMILIES
        and (row["family"], row["language"]) not in PN1_NOISY
    )


def pn1_summary(rows: list[dict[str, Any]], yes: dict[str, bool]) -> dict[str, Any]:
    missing = [r["id"] for r in rows if r["id"] not in yes]
    if missing:
        raise ValueError(f"{len(missing)} PN1 dev rows have no prediction")

    def rate(sel):
        picked = [r for r in rows if sel(r)]
        return {
            "n": len(picked),
            "yes": sum(yes[r["id"]] for r in picked) / len(picked),
        }

    out = {
        "all8": rate(lambda r: True),
        "pawsx6": rate(lambda r: r["language"] in PAWSX),
        "hop": rate(lambda r: r["family"] == "pn-hop"),
        "clean_no": rate(clean_no),
        "accuracy": sum(yes[r["id"]] == r["gold"] for r in rows) / len(rows),
        "by_family": {},
        "by_language": {},
    }
    for fam in sorted({r["family"] for r in rows}):
        out["by_family"][fam] = rate(lambda r, f=fam: r["family"] == f)
    for lang in sorted({r["language"] for r in rows}):
        out["by_language"][lang] = rate(lambda r, lg=lang: r["language"] == lg)
    return out


def pn1_compare(rows, yes_a, yes_b, reps=PN1_REPS, seed=PN1_SEED) -> dict[str, Any]:
    """B - A for the hop and clean gold-no yes-rates, groups resampled with replacement."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    names = sorted(by_group)

    def counts(members, yes):
        hop = [yes[r["id"]] for r in members if r["family"] == "pn-hop"]
        no = [yes[r["id"]] for r in members if clean_no(r)]
        return (sum(hop), len(hop), sum(no), len(no))

    stats = {g: (counts(by_group[g], yes_a), counts(by_group[g], yes_b)) for g in names}
    rng = random.Random(seed)
    draws = {"hop": [], "clean_no": []}
    for _ in range(reps):
        tot = [[0, 0, 0, 0], [0, 0, 0, 0]]
        for g in rng.choices(names, k=len(names)):
            for side in (0, 1):
                for i in range(4):
                    tot[side][i] += stats[g][side][i]
        a, b = tot
        if a[1] and a[3]:
            draws["hop"].append(b[0] / b[1] - a[0] / a[1])
            draws["clean_no"].append(b[2] / b[3] - a[2] / a[3])
    out = {}
    for k, values in draws.items():
        values.sort()
        n = len(values)
        out[k] = [values[int(0.025 * (n - 1))], values[int(round(0.975 * (n - 1)))]]
    return out


def pn1_command(args) -> dict[str, Any]:
    rows = pn1_rows(args.gold)
    yes, ref = pn1_yes(args.predictions), pn1_yes(args.reference_predictions)
    a, b = pn1_summary(rows, ref), pn1_summary(rows, yes)
    ci = pn1_compare(rows, ref, yes)
    return {
        "label": args.label,
        "reference": args.reference_label,
        "candidate": b,
        "reference_summary": a,
        "delta": {
            "hop": b["hop"]["yes"] - a["hop"]["yes"],
            "clean_no": b["clean_no"]["yes"] - a["clean_no"]["yes"],
            "all8": b["all8"]["yes"] - a["all8"]["yes"],
            "pawsx6": b["pawsx6"]["yes"] - a["pawsx6"]["yes"],
        },
        "delta_ci95": ci,
        "noisy_constructions_excluded": sorted(f"{f}/{lang}" for f, lang in PN1_NOISY),
        "role": ROLE,
    }


def select_accuracy(path: Path) -> float:
    return float(json.loads(path.read_text())["family_macro_accuracy"])


def early(p_pn1: dict, c_pn1: dict, p_select: float, c_select: float) -> dict[str, Any]:
    p, c = p_pn1["candidate"], c_pn1["candidate"]
    if p_pn1["reference"] != c_pn1["reference"]:
        raise ValueError("P and C PN1 readouts use different references")
    reasons = []
    if p["clean_no"]["yes"] > c["clean_no"]["yes"] - EARLY_NO_GAIN:
        reasons.append(
            f"clean gold-no yes {p['clean_no']['yes']:.4f} > control {c['clean_no']['yes']:.4f} - {EARLY_NO_GAIN}"
        )
    if p["hop"]["yes"] < c["hop"]["yes"] - HOP_SLACK:
        reasons.append(
            f"hop yes {p['hop']['yes']:.4f} < control {c['hop']['yes']:.4f} - {HOP_SLACK}"
        )
    if p_select < c_select - EARLY_SELECT_SLACK:
        reasons.append(
            f"SELECT700 {p_select:.4f} < control {c_select:.4f} - {EARLY_SELECT_SLACK}"
        )
    return {
        "P": {
            "clean_no_yes": p["clean_no"]["yes"],
            "hop_yes": p["hop"]["yes"],
            "select": p_select,
        },
        "C": {
            "clean_no_yes": c["clean_no"]["yes"],
            "hop_yes": c["hop"]["yes"],
            "select": c_select,
        },
        "continue": not reasons,
        "reasons": reasons,
        "role": ROLE,
    }


def screens(htdev2: dict, pn1: dict, mlx: dict) -> dict[str, Any]:
    reasons = []
    verdict = htdev2["htdev2"]["vs_reference"]["verdict"]
    if verdict == "FLAG":
        reasons.append("HT-DEV v2 FLAG")
    if pn1["delta"]["hop"] < -HOP_SLACK:
        reasons.append(f"PN1 hop yes {pn1['delta']['hop']:+.4f} < -{HOP_SLACK}")
    if pn1["delta"]["clean_no"] > 0:
        reasons.append(f"PN1 clean gold-no yes {pn1['delta']['clean_no']:+.4f} > 0")
    for key in ("noul_ml", "choice_ml"):
        high = mlx["metrics"][key]["ci95"][1]
        if high < 0:
            reasons.append(f"MLX-DEV-9B {key} upper {high:+.4f} < 0")
    return {
        "htdev2": {
            k: htdev2["htdev2"]["vs_reference"][k] for k in ("delta", "ci95", "verdict")
        },
        "pn1": {"delta": pn1["delta"], "ci95": pn1["delta_ci95"]},
        "mlxdev": {
            k: {"diff": mlx["metrics"][k]["diff"], "ci95": mlx["metrics"][k]["ci95"]}
            for k in ("noul_ml", "choice_ml", "score_ml", "noul_pred_yes_rate_macro")
        },
        "reasons": reasons,
    }


def eligibility(arm: dict, ref: dict, screen: dict | None) -> dict:
    base = m6_rules.eligibility(arm, ref)
    reasons = [r for r in base["reasons"] if not r.startswith("H3 ")]
    if screen is None:
        reasons.append("development screens missing")
    else:
        reasons += screen["reasons"]
    return {"eligible": not reasons, "reasons": reasons}


def alpha_line(points, arms: dict, ref_name: str, point_screens: dict) -> dict:
    alphas = [a for a, _ in points]
    if (
        not points
        or len(set(alphas)) != len(alphas)
        or set(alphas) - set(m6_rules.ALPHAS)
    ):
        raise ValueError("points must be distinct alphas from 1/3, 1/2, 2/3, 1")
    ref = arms[ref_name]
    t_ref = m4_rules.family_macro(ref)
    rows = []
    for alpha, name in sorted(points):
        arm = arms[name]
        t = m4_rules.family_macro(arm)
        screen = point_screens.get(name)
        rows.append(
            {
                "alpha": str(alpha),
                "arm": name,
                "T": float(t),
                "G": float(t - t_ref),
                "_gain": t - t_ref,
                "H3_report_only": arm["H_mean"],
                "H": arm["H"],
                "proxy": arm["proxy"],
                "rule_precedence": m6_rules.rp_count(arm)[0],
                "by_type": {k: v["correct"] for k, v in arm["by_type"].items()},
                "screens": screen,
                **eligibility(arm, ref, screen),
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
    keys = ("alpha", "arm", "T", "G", "H3_report_only", "proxy", "rule_precedence")
    summary = None if pick is None else {k: pick[k] for k in keys}
    return {
        "ref": {
            "arm": ref_name,
            "T": float(t_ref),
            "H3": ref["H_mean"],
            "proxy": ref["proxy"],
            "rule_precedence": m6_rules.rp_count(ref)[0],
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


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pn1", help="PN1 dev readout vs a reference")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference-predictions", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--reference-label", required=True)
    e = sub.add_parser("early", help="member-1 early stop of the P arm")
    e.add_argument("--p-pn1", required=True)
    e.add_argument("--c-pn1", required=True)
    e.add_argument(
        "--p-select", required=True, help="P_1 final select-step metrics JSON"
    )
    e.add_argument(
        "--c-select", required=True, help="C_1 final select-step metrics JSON"
    )
    a = sub.add_parser("alpha", help="the M7 alpha rule with development screens")
    a.add_argument("--readout", action="append", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--ref", required=True)
    a.add_argument("--point", action="append", required=True, help="ALPHA=KEY")
    a.add_argument(
        "--screen",
        action="append",
        default=[],
        help="KEY=HTDEV2_JSON,PN1_JSON,MLX_JSON",
    )
    f = sub.add_parser("finalists", help="non-dropped line picks in priority order")
    f.add_argument("--line", action="append", required=True, help="NAME=ALPHA_JSON")
    for sp in (p, e, a, f):
        sp.add_argument("--output")
    args = ap.parse_args(argv)
    if args.cmd == "pn1":
        out = pn1_command(args)
    elif args.cmd == "early":
        out = early(
            load_json(args.p_pn1),
            load_json(args.c_pn1),
            select_accuracy(Path(args.p_select)),
            select_accuracy(Path(args.c_select)),
        )
    elif args.cmd == "alpha":
        arms = m4_rules.load_arms(args.readout)
        points = [m5_rules.parse_point(x) for x in args.point]
        point_screens = {}
        for spec in args.screen:
            key, sep, paths = spec.partition("=")
            parts = paths.split(",")
            if not sep or len(parts) != 3:
                ap.error(f"--screen {spec!r}: expected KEY=HTDEV2,PN1,MLX")
            point_screens[key] = screens(*(load_json(x) for x in parts))
        out = {
            "line": args.name,
            **alpha_line(points, arms, args.ref, point_screens),
            "role": ROLE,
        }
    else:
        lines = []
        for spec in args.line:
            name, sep, path = spec.partition("=")
            if not sep:
                ap.error(f"--line {spec!r}: expected NAME=ALPHA_JSON")
            lines.append((name, load_json(path)))
        out = {**m5_rules.finalists(lines), "role": ROLE}
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(
        text
        if args.cmd != "pn1"
        else json.dumps({"label": out["label"], "delta": out["delta"]})
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
