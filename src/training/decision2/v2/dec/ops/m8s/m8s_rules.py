"""Decoder M8-small development rules (prereg dec-m8s-prereg-2026-09-30.md), host Python on node B.

  early      after seed 1: a D arm continues iff HT-DEV v2 of D-s1 vs C-s1 is not FLAG (delta > -0.02) and its final
             SELECT700 family-macro accuracy is >= C-s1's - 0.03. Writes <status>/early-<tier>-<arm>.json and an
             empty marker early-<tier>-<arm>.PASS or .STOP.
  finalists  per line L-<ARM> (points alpha 1, 1/2, 1/3 toward the arm soup; reference <tier>-I read the same way):
             a point passes when every typed-DEV type keeps c_t >= c_t,I - 0.03 n_t, no typed-DEV family falls below
             F_f,I - 0.10, and HT-DEV v2 vs I is not FLAG. Pick = a passing HT-DEV v2 GAIN point if any, else a TIE,
             the larger alpha first within a class; a pick whose proxy P is >= 8 below the tier's best pick is
             dropped; slots D1, D2, C; at most three finalists. Development only; never a release score.

             Amendment 2 adds the 4B M8 floors: Noul `rule_precedence` c >= c_I - 0.01 n, and Score5-typed-DEV check
             half without COLLAPSE, and without WARN unless I's check half has WARN.
  htdev2     one point vs a reference on HT-DEV v2 (paired, 04:10 verdict) -> the JSON the finalists rule reads
  score5t    node A (gold): the eval's Score5-typed-DEV block of one prediction file -> the JSON the finalists rule reads

usage: python3 m8s_rules.py early --root M --tier 2b|08b --arm D1|D2 --gold G
       python3 m8s_rules.py finalists --tier 2b|08b --lines-root L --output <select>/<tier>-finalists.json
       python3 m8s_rules.py htdev2 --gold G --left P --left-name A --right P --right-name B --output OUT
       python3 m8s_rules.py score5t --panel-root /data/dev2/private/panels --predictions P --output OUT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

SCHEMA = "dec-m8s-rules/1"
LINES = ("D1", "D2", "C")
ORDER = ("1", "1/2", "1/3")
TYPE_SLACK = Fraction(3, 100)
FAMILY_SLACK = Fraction(1, 10)
NOUL_FAMILY = "rule_precedence"
NOUL_SLACK = Fraction(1, 100)
TOP_SHARE = 0.90
HT_TIE = 0.02
EARLY_SELECT_SLACK = 0.03
PROXY_DROP = 8.0
RANK = {"GAIN": 0, "TIE": 1}


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def htdev2_verdict(delta: float) -> str:
    return "FLAG" if delta <= -HT_TIE else "GAIN" if delta >= HT_TIE else "TIE"


def floors(point: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    reasons = []
    for t, r in ref["by_type"].items():
        got = point["by_type"][t]
        if got["n"] != r["n"]:
            raise ValueError(f"type {t}: n differs from the reference")
        floor = r["correct"] - TYPE_SLACK * r["n"]
        if got["correct"] < floor:
            reasons.append(f"type {t} {got['correct']} < floor {float(floor):g}")
    for f, r in ref["by_family"].items():
        got = point["by_family"][f]
        if got["n"] != r["n"]:
            raise ValueError(f"family {f}: n differs from the reference")
        if (
            Fraction(got["correct"], got["n"])
            < Fraction(r["correct"], r["n"]) - FAMILY_SLACK
        ):
            reasons.append(
                f"family {f} {got['correct']}/{got['n']} below reference - 0.10"
            )
    noul_ref = ref["by_family"].get(NOUL_FAMILY) or ref["by_type"]["noul"]
    noul = point["by_family"].get(NOUL_FAMILY) or point["by_type"]["noul"]
    noul_floor = noul_ref["correct"] - NOUL_SLACK * noul_ref["n"]
    if noul["correct"] < noul_floor:
        reasons.append(
            f"Noul {NOUL_FAMILY} {noul['correct']} < floor {float(noul_floor):g}"
        )
    return reasons


def score5t_flags(block: dict[str, Any]) -> set[str]:
    flags = block["check"]["flags"]
    return set(flags if isinstance(flags, list) else [flags] if flags else [])


def score_floor(point_s5: dict[str, Any], ref_s5: dict[str, Any]) -> list[str]:
    flags, ref_flags = score5t_flags(point_s5), score5t_flags(ref_s5)
    reasons = []
    if "COLLAPSE" in ref_flags:
        # Amendment 3: the incumbent's check half is itself flagged, so only a new level concentration counts
        # (the eval's top-share thresholds for COLLAPSE / WARN).
        top, upper = (
            point_s5["check"]["top_share"],
            point_s5["check"]["top_share_wilson95"][1],
        )
        if top >= TOP_SHARE:
            reasons.append(f"Score5-typed-DEV check top share {top:.3f} >= {TOP_SHARE}")
        elif upper >= TOP_SHARE > ref_s5["check"]["top_share_wilson95"][1]:
            reasons.append(
                f"Score5-typed-DEV check top-share upper bound {upper:.3f} >= {TOP_SHARE}"
            )
        return reasons
    if "COLLAPSE" in flags:
        reasons.append("Score5-typed-DEV check half COLLAPSE")
    if "WARN" in flags and "WARN" not in ref_flags:
        reasons.append("Score5-typed-DEV check half WARN (reference has none)")
    return reasons


def gate(
    point: dict[str, Any],
    ref: dict[str, Any],
    ht: dict[str, Any],
    s5: dict[str, Any] | None = None,
    s5_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons = floors(point, ref)
    if s5 is not None:
        reasons += score_floor(s5, s5_ref)
    verdict = htdev2_verdict(ht["delta"])
    if verdict == "FLAG":
        reasons.append(f"HT-DEV v2 FLAG ({ht['delta']:+.4f})")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "htdev2": verdict,
        "htdev2_delta": ht["delta"],
        "score5t_check_flags": sorted(score5t_flags(s5)) if s5 is not None else None,
    }


def pick(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [r for r in rows if r["eligible"]]
    if not passing:
        return None
    return min(passing, key=lambda r: (RANK[r["htdev2"]], ORDER.index(r["step"])))


def select(
    lines: dict[str, list[dict[str, Any]]], dropped: dict[str, str]
) -> dict[str, Any]:
    picks = {line: pick(rows) for line, rows in lines.items()}
    live = [p["proxy"] for p in picks.values() if p]
    best = max(live, default=None)
    removed = {
        line: p["point"]
        for line, p in picks.items()
        if p and best is not None and p["proxy"] <= best - PROXY_DROP
    }
    finalists, not_finalists = [], []
    for line in LINES:
        if line in dropped:
            not_finalists.append({"line": line, "reason": f"dropped: {dropped[line]}"})
            continue
        p = picks.get(line)
        if p is None:
            reasons = sorted({x for r in lines.get(line, []) for x in r["reasons"]})
            not_finalists.append(
                {
                    "line": line,
                    "reason": "no point passes the gates",
                    "reasons": reasons,
                }
            )
        elif line in removed:
            not_finalists.append(
                {
                    "line": line,
                    "reason": "pick removed by the proxy drop rule",
                    "pick": p["point"],
                }
            )
        else:
            finalists.append({"slot": len(finalists) + 1, "line": line, **p})
    return {
        "picks": {line: (p["point"] if p else None) for line, p in picks.items()},
        "best_proxy": best,
        "proxy_dropped": removed,
        "finalists": finalists,
        "not_finalists": not_finalists,
    }


def final_select(run: Path) -> float:
    best = json.loads((run / "BEST.json").read_text())["checkpoint"]
    return float(
        json.loads((run / best / "checkpoint.json").read_text())["dev_metrics"][
            "family_macro_accuracy"
        ]
    )


def early(a: argparse.Namespace) -> int:
    htdev2 = _module("m7_htdev2", CODE / "v2/dec/ops/m7/m7_htdev2.py")
    from v2.eval.htdev2 import score as scorer

    def preds(arm: str) -> Path:
        return (
            a.root
            / "early"
            / f"{a.tier}-{arm}-s1"
            / "ht-dev2"
            / "ht-dev2.predictions.jsonl"
        )

    def run(arm: str) -> Path:
        return a.root / "arms" / f"{a.tier}-{arm}-s1" / "full"

    ht = htdev2.evaluate(
        scorer,
        a.gold,
        preds(a.arm),
        f"{a.tier}-{a.arm}-s1",
        preds("C"),
        f"{a.tier}-C-s1",
    )
    sel_d, sel_c = final_select(run(a.arm)), final_select(run("C"))
    reasons = []
    if ht["delta"] <= -HT_TIE:
        reasons.append(f"HT-DEV v2 vs C-s1 FLAG ({ht['delta']:+.4f})")
    if sel_d < sel_c - EARLY_SELECT_SLACK:
        reasons.append(f"SELECT700 {sel_d:.4f} < C-s1 {sel_c:.4f} - 0.03")
    status = "STOP" if reasons else "PASS"
    doc = {
        "schema": SCHEMA + ":early",
        "tier": a.tier,
        "arm": a.arm,
        "status": status,
        "reasons": reasons,
        "htdev2": {
            k: ht[k]
            for k in ("H_dev2", "delta", "ci95", "p_le_0", "verdict", "tasks", "files")
        },
        "select700_family_macro": {"D": sel_d, "C": sel_c},
    }
    st = a.root / "status"
    st.mkdir(parents=True, exist_ok=True)
    (st / f"early-{a.tier}-{a.arm}.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n"
    )
    (st / f"early-{a.tier}-{a.arm}.{status}").touch()
    print(
        json.dumps(
            {
                "tier": a.tier,
                "arm": a.arm,
                "status": status,
                "delta": round(ht["delta"], 4),
                "select": [round(sel_d, 4), round(sel_c, 4)],
                "reasons": reasons,
            }
        )
    )
    return 0


def htdev2_pair(a: argparse.Namespace) -> int:
    """One point against a reference on the 1,944 HT-DEV v2 items (the eval scorer through m7_htdev2.evaluate)."""
    htdev2 = _module("m7_htdev2", CODE / "v2/dec/ops/m7/m7_htdev2.py")
    from v2.eval.htdev2 import score as scorer

    out = htdev2.evaluate(scorer, a.gold, a.left, a.left_name, a.right, a.right_name)
    out["schema"] = SCHEMA + ":htdev2"
    out["role"] = (
        "M8-small development gate (prereg dec-m8s-prereg-2026-09-30.md: not FLAG, prefer GAIN); "
        "development readout, never a release score"
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "left": a.left_name,
                "right": a.right_name,
                "delta": round(out["delta"], 4),
                "ci95": [round(x, 4) for x in out["ci95"]],
                "verdict": out["verdict"],
            }
        )
    )
    return 0


def score5t_one(a: argparse.Namespace) -> int:
    """The eval's Score5-typed-DEV block (full / fit / check, flags) of one prediction file (gold on node A)."""
    from v2.eval.dev_readout import score5t_block

    out = score5t_block(a.panel_root, a.predictions)
    out["role"] = (
        "M8-small Score floor (amendment 2): check half without COLLAPSE / new WARN"
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "check_flags": sorted(score5t_flags(out)),
                "check_top_share": out["check"].get("top_share"),
            }
        )
    )
    return 0


def parse_line(spec: str) -> tuple[str, dict[str, str]]:
    line, rest = spec.split(":", 1)
    return line, dict(item.split("=", 1) for item in rest.split(",") if item)


def finalists(a: argparse.Namespace) -> int:
    ref = f"{a.tier}-I"
    dropped = dict(d.split("=", 1) for d in a.dropped)
    lines, readouts = {}, {}
    for line in LINES:
        out = a.lines_root / "readout" / f"L-{line}.json"
        spec = a.lines_root / "readout" / f"L-{line}.line"
        if not (out.is_file() and spec.is_file()):
            if line not in dropped:
                raise SystemExit(
                    f"L-{line} has no readout (m8s-lines.sh) and is not declared --dropped"
                )
            continue
        name, steps = parse_line(spec.read_text().strip())
        if name != f"L-{line}":
            raise SystemExit(f"{spec} names {name}")
        arms = json.loads(out.read_text())["arms"]
        s5_ref = json.loads((a.lines_root / "diag" / f"{ref}.score5t.json").read_text())
        rows = []
        for step in ORDER:
            point = steps.get(step)
            if point is None:
                continue
            ht_path = a.lines_root / "diag" / f"{point}.htdev2.json"
            ht = json.loads(ht_path.read_text())
            s5_path = a.lines_root / "diag" / f"{point}.score5t.json"
            g = gate(
                arms[point], arms[ref], ht, json.loads(s5_path.read_text()), s5_ref
            )
            rows.append(
                {
                    "step": step,
                    "point": point,
                    "T": arms[point]["T"],
                    "G": arms[point]["T"] - arms[ref]["T"],
                    "H3": arms[point]["H_mean"],
                    "H": arms[point]["H"],
                    "proxy": arms[point]["proxy"],
                    "by_type": {
                        t: v["correct"] for t, v in arms[point]["by_type"].items()
                    },
                    "htdev2_file": str(ht_path),
                    "htdev2_sha256": sha_file(ht_path),
                    "score5t_file": str(s5_path),
                    "score5t_sha256": sha_file(s5_path),
                    **g,
                }
            )
        lines[line] = rows
        readouts[line] = {
            "file": str(out),
            "sha256": sha_file(out),
            "line": spec.read_text().strip(),
        }
    doc = {
        "schema": SCHEMA + ":finalists",
        "tier": a.tier,
        "role": "development-only selection (prereg dec-m8s-prereg-2026-09-30.md); not a release or post-key score",
        "reference": ref,
        "readouts": readouts,
        "lines": lines,
        **select(lines, dropped),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        a.output.rename(a.output.with_name(a.output.name + ".prev"))
    a.output.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finalists": [
                    (f["slot"], f["line"], f["point"], f["htdev2"])
                    for f in doc["finalists"]
                ],
                "not_finalists": doc["not_finalists"],
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("early")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--tier", choices=("2b", "08b"), required=True)
    e.add_argument("--arm", choices=("D1", "D2"), required=True)
    e.add_argument("--gold", type=Path, required=True)
    f = sub.add_parser("finalists")
    f.add_argument("--tier", choices=("2b", "08b"), required=True)
    f.add_argument("--lines-root", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument(
        "--dropped",
        action="append",
        default=[],
        help="LINE=reason for a line without a readout",
    )
    h = sub.add_parser("htdev2")
    h.add_argument("--gold", type=Path, required=True)
    h.add_argument("--left", type=Path, required=True)
    h.add_argument("--left-name", required=True)
    h.add_argument("--right", type=Path, required=True)
    h.add_argument("--right-name", required=True)
    h.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score5t")
    s.add_argument("--panel-root", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    return {
        "early": early,
        "finalists": finalists,
        "htdev2": htdev2_pair,
        "score5t": score5t_one,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
