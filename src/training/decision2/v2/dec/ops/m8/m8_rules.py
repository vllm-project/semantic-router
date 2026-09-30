"""Decoder M8 development rules (prereg dec-m8-prereg-2026-09-30.md, "Early stop", "Lines and the development rule").

  early    node B, after X-m1 and C-m1: X continues iff X-m1's final SELECT700 family-macro accuracy
           >= C-m1's - 0.02 -> m8/early/<X>.json
  score5t  node A: one point's Score5-typed-DEV blocks (v2.eval.score5t.blocks; check-half flags) -> diag/<point>.score5t.json
  alpha    node A: every point of the lines L-D1, L-D2, L-C against the reference 4b-I:
             1. typed floors: c_t >= c_t,I - 0.03 n_t for every type, F_f >= F_f,I - 0.10 for every typed-DEV family;
             2. Noul floor: rule_precedence c >= c_I - 0.01 * 400;
             3. Score floor: Score5-typed-DEV check half without COLLAPSE, and without WARN unless 4b-I's has WARN;
             4. HT-DEV v2 not FLAG (m7_htdev2 verdict vs 4b-I, delta > -0.02).
           Pick per line: eligible points with an HT-DEV v2 GAIN first, then the largest alpha (1, 2/3, 1/3); a pick whose
           proxy P is >= 8 below the best pick's is dropped. Finalists in line order D1, D2, C (at most three).
Development only; never a release or post-key score; never v3, C1, mlx-diag or public 231.

usage: m8_rules.py early --arm D1|D2 --x-run RUN --c-run RUN --output OUT
       m8_rules.py score5t --panel-root /data/dev2/private/panels --predictions P --output OUT
       m8_rules.py alpha --lines-root /data/dev2/runs/dec/m8/lines/4b --output <select>/4b-finalists.json [--dropped L-X=why]
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

SCHEMA = "dec-m8-finalists/1"
LINES = ("L-D1", "L-D2", "L-C")
ORDER = ("1", "2/3", "1/3")
REF = "4b-I"
SELECT_MARGIN = Fraction(2, 100)
TYPE_TOL = Fraction(3, 100)
FAMILY_TOL = Fraction(1, 10)
NOUL_FAMILY, NOUL_TOL = "rule_precedence", Fraction(1, 100)
PROXY_DROP = 8.0


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_select(run: Path) -> dict[str, Any]:
    latest = json.loads((run / "LATEST.json").read_text())["checkpoint"]
    complete = json.loads((run / "COMPLETE.json").read_text())
    metrics = json.loads((run / latest / "checkpoint.json").read_text())["dev_metrics"]
    return {
        "run": str(run),
        "checkpoint": latest,
        "step": complete["step"],
        "select_family_macro": metrics["family_macro_accuracy"],
    }


def early(arm: str, x_run: Path, c_run: Path) -> dict[str, Any]:
    x, c = final_select(x_run), final_select(c_run)
    ok = (
        Fraction(x["select_family_macro"])
        >= Fraction(c["select_family_macro"]) - SELECT_MARGIN
    )
    return {
        "schema": "dec-m8-early/1",
        "arm": arm,
        "rule": "X-m1 final SELECT700 family-macro accuracy >= C-m1's - 0.02",
        "x": x,
        "c": c,
        "difference": x["select_family_macro"] - c["select_family_macro"],
        "decision": "continue" if ok else "stop",
    }


def eligibility(
    point: dict[str, Any],
    ref: dict[str, Any],
    htdev2: dict[str, Any] | None,
    s5: dict[str, Any] | None,
    s5_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons = []
    for kind, value in ref["by_type"].items():
        n, c, c_ref = value["n"], point["by_type"][kind]["correct"], value["correct"]
        if point["by_type"][kind].get("invalid"):
            reasons.append(
                f"{kind}: {point['by_type'][kind]['invalid']} invalid answers"
            )
        if c < c_ref - TYPE_TOL * n:
            reasons.append(f"type floor {kind}: {c} < {c_ref} - 0.03*{n}")
    for family, value in ref["by_family"].items():
        mine = point["by_family"][family]
        if (
            Fraction(mine["correct"], mine["n"])
            < Fraction(value["correct"], value["n"]) - FAMILY_TOL
        ):
            reasons.append(
                f"family floor {family}: {mine['correct']}/{mine['n']} < {value['correct']}/{value['n']} - 0.10"
            )
    rp, rp_ref = point["by_family"][NOUL_FAMILY], ref["by_family"][NOUL_FAMILY]
    if rp["correct"] < rp_ref["correct"] - NOUL_TOL * rp_ref["n"]:
        reasons.append(
            f"Noul floor: {NOUL_FAMILY} {rp['correct']} < {rp_ref['correct']} - 0.01*{rp_ref['n']}"
        )
    if s5 is None or s5_ref is None:
        reasons.append("Score5-typed-DEV readout missing")
    else:
        flags, ref_flags = set(s5["check"]["flags"]), set(s5_ref["check"]["flags"])
        if "COLLAPSE" in flags:
            reasons.append("Score floor: Score5-typed-DEV check half COLLAPSE")
        if "WARN" in flags and "WARN" not in ref_flags:
            reasons.append(
                "Score floor: Score5-typed-DEV check half WARN (4b-I has none)"
            )
    if htdev2 is None:
        reasons.append("HT-DEV v2 readout missing")
    elif htdev2["verdict"] == "FLAG":
        reasons.append(f"HT-DEV v2 FLAG ({htdev2['delta']:+.4f})")
    return {"eligible": not reasons, "reasons": reasons}


def line_rows(
    readout: dict[str, Any], steps: dict[str, str], diag: Path
) -> list[dict[str, Any]]:
    arms = readout["arms"]
    s5_ref = load_opt(diag / f"{REF}.score5t.json")
    rows = []
    for step in ORDER:
        point = steps.get(step)
        if point is None:
            continue
        arm = arms[point]
        htdev2 = load_opt(diag / f"{point}.htdev2.json")
        s5 = load_opt(diag / f"{point}.score5t.json")
        rows.append(
            {
                "step": step,
                "point": point,
                "T": arm["T"],
                "G": arm["T"] - arms[REF]["T"],
                "H3": arm["H_mean"],
                "H": arm["H"],
                "proxy": arm["proxy"],
                "by_type": {t: v["correct"] for t, v in arm["by_type"].items()},
                "rule_precedence": arm["by_family"][NOUL_FAMILY]["correct"],
                "htdev2": (
                    None
                    if htdev2 is None
                    else {k: htdev2[k] for k in ("delta", "ci95", "verdict")}
                ),
                "score5t_check_flags": None if s5 is None else s5["check"]["flags"],
                **eligibility(arm, arms[REF], htdev2, s5, s5_ref),
            }
        )
    return rows


def load_opt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def pick(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if r["eligible"]]
    gain = [r for r in ok if r["htdev2"] and r["htdev2"]["verdict"] == "GAIN"]
    for pool in (gain, ok):
        if pool:
            return min(pool, key=lambda r: ORDER.index(r["step"]))
    return None


def select(
    lines: dict[str, list[dict[str, Any]]], dropped: dict[str, str]
) -> dict[str, Any]:
    picks = {line: pick(rows) for line, rows in lines.items()}
    live = [p["proxy"] for line, p in picks.items() if p and line not in dropped]
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


def parse_line(spec: str) -> tuple[str, dict[str, str]]:
    line, rest = spec.split(":", 1)
    return line, dict(item.split("=", 1) for item in rest.split(",") if item)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("early")
    e.add_argument("--arm", choices=("D1", "D2"), required=True)
    e.add_argument("--x-run", type=Path, required=True)
    e.add_argument("--c-run", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score5t")
    s.add_argument("--panel-root", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    a_ = sub.add_parser("alpha")
    a_.add_argument("--lines-root", type=Path, required=True)
    a_.add_argument("--output", type=Path, required=True)
    a_.add_argument(
        "--dropped",
        action="append",
        default=[],
        help="L-X=reason for a line without a readout",
    )
    a = p.parse_args(argv)
    if a.cmd == "early":
        out = early(a.arm, a.x_run, a.c_run)
        with open(a.output, "x") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(
            json.dumps(
                {
                    "arm": a.arm,
                    "decision": out["decision"],
                    "difference": round(out["difference"], 4),
                }
            )
        )
        return 0
    if a.cmd == "score5t":
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from v2.eval import panels, score5t

        panels.verify(a.panel_root, [score5t.PANEL])
        gold = [
            json.loads(x)
            for x in panels.path(a.panel_root, score5t.PANEL, "gold")
            .read_text()
            .splitlines()
            if x.strip()
        ]
        preds = {
            r["id"]: r
            for r in (
                json.loads(x)
                for x in a.predictions.read_text().splitlines()
                if x.strip()
            )
        }
        out = {
            "schema": "dec-m8-score5t/1",
            "role": "development screen (M8 Score floor); never a release score",
            "predictions_sha256": sha_file(a.predictions),
            **score5t.blocks(gold, preds),
        }
        with open(a.output, "x") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(
            json.dumps(
                {
                    "check_flags": out["check"]["flags"],
                    "top_share": out["check"].get("top_share"),
                }
            )
        )
        return 0
    if not a.output.name.endswith("-finalists.json"):
        p.error("--output must end with -finalists.json")
    dropped = dict(d.split("=", 1) for d in a.dropped)
    _f = importlib.util.spec_from_file_location(
        "m6_finalists", Path(__file__).resolve().parents[1] / "m6" / "m6_finalists.py"
    )
    m6f = importlib.util.module_from_spec(_f)
    _f.loader.exec_module(m6f)
    lines, readouts = {}, {}
    for line in LINES:
        out, spec = (
            a.lines_root / "readout" / f"{line}.json",
            a.lines_root / "readout" / f"{line}.line",
        )
        if not (out.is_file() and spec.is_file()):
            if line not in dropped:
                p.error(
                    f"{line} has no readout (run m8-lines.sh, or declare --dropped)"
                )
            continue
        name, steps = parse_line(spec.read_text().strip())
        if name != line:
            p.error(f"{spec}: names {name}, not {line}")
        lines[line] = line_rows(
            json.loads(out.read_text()), steps, a.lines_root / "diag"
        )
        readouts[line] = {
            "file": str(out),
            "sha256": sha_file(out),
            "line": spec.read_text().strip(),
        }
    result = select(lines, dropped)
    for f in result["finalists"]:
        f.update(m6f.point_info(a.lines_root, f["point"]))
    doc = {
        "schema": SCHEMA,
        "tier": "4b",
        "role": "development-only selection (prereg dec-m8-prereg-2026-09-30.md); not a release or post-key score",
        "rules_module_sha256": sha_file(Path(__file__)),
        "reference": REF,
        "readouts": readouts,
        "lines": lines,
        **result,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        a.output.rename(a.output.with_name(a.output.name + ".prev"))
    a.output.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finalists": [
                    (f["slot"], f["line"], f["point"]) for f in doc["finalists"]
                ],
                "not_finalists": doc["not_finalists"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
