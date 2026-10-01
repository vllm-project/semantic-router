"""Decoder M9 development rules (prereg dec-m9-prereg-2026-10-01.md, "Early stop", "Lines and the development gates").

  htdev2   one paired HT-DEV v2 comparison (the eval track's scorer through ops/m7/m7_htdev2.evaluate: task macro-F1,
           H_dev2 = mean of the nine tasks, item bootstrap within tasks, 2,000 draws; FLAG <= -0.02 < TIE < +0.02 <= GAIN)
  score5t  one point's Score5-typed-DEV blocks (v2.eval.score5t.blocks; check-half flags)
  early    after H9-s1 and C9-s1 (node A): H9 stops iff
             E1-typed: H9-s1's BEST SELECT700 family-macro accuracy < C9-s1's - 0.03, or
             E1-human: dH_dev2(H9-s1 - C9-s1) <= -0.03 and dH_dev2(H9-s1 - 4b-I) <= -0.02 (FLAG);
           if H9 stops, C9 stops too
  gates    every point of L-H9 and the control line (L-N7C after amendment 1; L-C9 as preregistered) (alpha 1,
           1/2) against 4b-I with the M8 gates (ops/m8/m8_rules.eligibility:
           typed type / family floors, Noul rule_precedence floor, Score5-typed-DEV check-half floor, HT-DEV v2 not
           FLAG); the formal candidate is L-H9's pick (eligible points with a GAIN first, then the larger alpha); the
           control line is gated for the report only. Adds the matched contrasts (H9 - C9 at each alpha) when present.
Development only; never a release or post-key score; never v3, C1, mlx-diag or public 231.

usage: m9_rules.py htdev2 --gold G --left P --left-name A --right P --right-name B --output OUT
       m9_rules.py score5t --panel-root /data/dev2/private/panels --predictions P --output OUT
       m9_rules.py early --h-run RUN --c-run RUN --gold G --h-ht P --c-ht P --i-ht P --output OUT
       m9_rules.py gates --lines-root /data/dev2/runs/dec/m9/lines/4b --output <select>/4b-pick.json
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

OPS = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parents[4]
SCHEMA = "dec-m9-pick/1"
CONTROLS = ("C9", "N7C")
ORDER = ("1", "1/2")
REF = "4b-I"
CANDIDATE_LINE = "L-H9"
E1_TYPED = Fraction(3, 100)
E1_VS_C = -0.03
E1_VS_I = -0.02
ROLE = "development readout (decoder M9, HR2 efficacy pilot); never a release or post-key score"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_opt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def write_new(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")


def htdev2_pair(
    gold: Path, left: Path, left_name: str, right: Path, right_name: str
) -> dict[str, Any]:
    sys.path.insert(0, str(CODE))
    from v2.eval.htdev2 import score as htdev2

    m7 = _load("m7_htdev2", OPS / "m7" / "m7_htdev2.py")
    out = m7.evaluate(htdev2, gold, left, left_name, right, right_name)
    out["schema"] = "dec-m9-htdev2/1"
    out["role"] = ROLE + "; HT-DEV v2 screen (COORDINATION 2026-09-30 04:10)"
    return out


def best_select(run: Path) -> dict[str, Any]:
    best = json.loads((run / "BEST.json").read_text())["checkpoint"]
    metrics = json.loads((run / best / "checkpoint.json").read_text())["dev_metrics"]
    return {
        "run": str(run),
        "checkpoint": best,
        "select_family_macro": metrics["family_macro_accuracy"],
    }


def early(
    h_run: Path, c_run: Path, h_vs_c: dict[str, Any], h_vs_i: dict[str, Any]
) -> dict[str, Any]:
    h, c = best_select(h_run), best_select(c_run)
    typed_stop = (
        Fraction(h["select_family_macro"])
        < Fraction(c["select_family_macro"]) - E1_TYPED
    )
    human_stop = h_vs_c["delta"] <= E1_VS_C and h_vs_i["delta"] <= E1_VS_I
    stop = typed_stop or human_stop
    return {
        "schema": "dec-m9-early/1",
        "role": ROLE,
        "rules": {
            "E1-typed": "H9-s1 BEST SELECT700 family-macro < C9-s1's - 0.03 -> stop",
            "E1-human": "dH_dev2(H9-s1 - C9-s1) <= -0.03 and dH_dev2(H9-s1 - 4b-I) <= -0.02 -> stop",
            "both_arms": "if H9 stops, C9 stops too",
        },
        "h9_s1": h,
        "c9_s1": c,
        "select_difference": h["select_family_macro"] - c["select_family_macro"],
        "htdev2_h9_minus_c9": {
            k: h_vs_c[k] for k in ("delta", "ci95", "verdict", "H_dev2")
        },
        "htdev2_h9_minus_i": {
            k: h_vs_i[k] for k in ("delta", "ci95", "verdict", "H_dev2")
        },
        "E1_typed_stop": typed_stop,
        "E1_human_stop": human_stop,
        "decision": "stop" if stop else "continue",
    }


def line_rows(
    readout: dict[str, Any], steps: dict[str, str], diag: Path, m8
) -> list[dict[str, Any]]:
    arms = readout["arms"]
    s5_ref = load_opt(diag / f"{REF}.score5t.json")
    rows = []
    for step in ORDER:
        point = steps.get(step)
        if point is None:
            continue
        arm = arms[point]
        ht = load_opt(diag / f"{point}.htdev2.json")
        s5 = load_opt(diag / f"{point}.score5t.json")
        hr2 = load_opt(diag / f"{point}.hr2dev.json")
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
                "rule_precedence": arm["by_family"][m8.NOUL_FAMILY]["correct"],
                "htdev2": (
                    None
                    if ht is None
                    else {k: ht[k] for k in ("delta", "ci95", "verdict")}
                ),
                "score5t_check_flags": None if s5 is None else s5["check"]["flags"],
                "score5t_check_top_share": (
                    None if s5 is None else s5["check"].get("top_share")
                ),
                "hr2dev_family_macro": None if hr2 is None else hr2["family_macro"],
                "hr2dev_delta_vs_I": (
                    None if hr2 is None else hr2.get("delta_family_macro")
                ),
                **m8.eligibility(arm, arms[REF], ht, s5, s5_ref),
            }
        )
    return rows


def pick(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if r["eligible"]]
    gain = [r for r in ok if r["htdev2"] and r["htdev2"]["verdict"] == "GAIN"]
    for pool in (gain, ok):
        if pool:
            return min(pool, key=lambda r: ORDER.index(r["step"]))
    return None


def parse_line(spec: str) -> tuple[str, dict[str, str]]:
    line, rest = spec.split(":", 1)
    return line, dict(item.split("=", 1) for item in rest.split(",") if item)


def gates(lines_root: Path, control: str = "C9") -> dict[str, Any]:
    m8 = _load("m8_rules", OPS / "m8" / "m8_rules.py")
    m6f = _load("m6_finalists", OPS / "m6" / "m6_finalists.py")
    lines, readouts = {}, {}
    for line in (CANDIDATE_LINE, f"L-{control}"):
        out = lines_root / "readout" / f"{line}.json"
        spec = lines_root / "readout" / f"{line}.line"
        if not (out.is_file() and spec.is_file()):
            continue
        name, steps = parse_line(spec.read_text().strip())
        if name != line:
            raise SystemExit(f"{spec}: names {name}, not {line}")
        lines[line] = line_rows(
            json.loads(out.read_text()), steps, lines_root / "diag", m8
        )
        readouts[line] = {
            "file": str(out),
            "sha256": sha_file(out),
            "line": spec.read_text().strip(),
        }
    if CANDIDATE_LINE not in lines:
        raise SystemExit(f"{CANDIDATE_LINE} has no readout")
    chosen = pick(lines[CANDIDATE_LINE])
    contrasts = {}
    for step, tag in (("1", "a1"), ("1/2", "a1_2")):
        doc = load_opt(lines_root / "diag" / f"pair-H9-{control}-{tag}.htdev2.json")
        if doc is not None:
            contrasts[step] = {
                k: doc[k] for k in ("delta", "ci95", "p_le_0", "verdict", "H_dev2")
            }
    result = {
        "schema": SCHEMA,
        "tier": "4b",
        "role": ROLE
        + "; selects at most one formal run (pilot, not a release candidate)",
        "rules_module_sha256": sha_file(Path(__file__)),
        "reference": REF,
        "readouts": readouts,
        "lines": lines,
        "control": control,
        "control_note": (
            "amendment 1: C9 stopped by its seed-1 preflight; the control is M7's N7C "
            "(same base, recipe and seeds; 43.5M vs 46.2M tokens)"
            if control == "N7C"
            else "preregistered matched-token control"
        ),
        "matched_contrasts_htdev2_h9_minus_control": contrasts,
        "pick": (
            None
            if chosen is None
            else {**chosen, **m6f.point_info(lines_root, chosen["point"])}
        ),
        "no_pick_reasons": (
            None
            if chosen is not None
            else sorted({x for r in lines[CANDIDATE_LINE] for x in r["reasons"]})
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("htdev2")
    for name in ("gold", "left", "right", "output"):
        h.add_argument(f"--{name}", type=Path, required=True)
    h.add_argument("--left-name", required=True)
    h.add_argument("--right-name", required=True)
    s = sub.add_parser("score5t")
    s.add_argument("--panel-root", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("early")
    for name in ("h-run", "c-run", "gold", "h-ht", "c-ht", "i-ht", "output"):
        e.add_argument(f"--{name}", type=Path, required=True)
    g = sub.add_parser("gates")
    g.add_argument("--lines-root", type=Path, required=True)
    g.add_argument("--control", choices=CONTROLS, default="N7C")
    g.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "htdev2":
        out = htdev2_pair(a.gold, a.left, a.left_name, a.right, a.right_name)
        write_new(a.output, out)
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
    if a.cmd == "score5t":
        sys.path.insert(0, str(CODE))
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
            "schema": "dec-m9-score5t/1",
            "role": ROLE + "; Score floor screen",
            "predictions_sha256": sha_file(a.predictions),
            **score5t.blocks(gold, preds),
        }
        write_new(a.output, out)
        print(
            json.dumps(
                {
                    "check_flags": out["check"]["flags"],
                    "top_share": out["check"].get("top_share"),
                }
            )
        )
        return 0
    if a.cmd == "early":
        h_vs_c = htdev2_pair(a.gold, a.h_ht, "4b-H9-s1", a.c_ht, "4b-C9-s1")
        h_vs_i = htdev2_pair(a.gold, a.h_ht, "4b-H9-s1", a.i_ht, REF)
        out = early(a.h_run, a.c_run, h_vs_c, h_vs_i)
        out["files"] = {
            "gold": sha_file(a.gold),
            "h9_s1": sha_file(a.h_ht),
            "c9_s1": sha_file(a.c_ht),
            "4b-I": sha_file(a.i_ht),
        }
        write_new(a.output, out)
        print(
            json.dumps(
                {
                    "decision": out["decision"],
                    "select_difference": round(out["select_difference"], 4),
                    "h9_minus_c9": round(h_vs_c["delta"], 4),
                    "h9_minus_i": round(h_vs_i["delta"], 4),
                }
            )
        )
        return 0
    if not a.output.name.endswith("-pick.json"):
        p.error("--output must end with -pick.json")
    doc = gates(a.lines_root, a.control)
    if a.output.exists():
        a.output.rename(a.output.with_name(a.output.name + ".prev"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "pick": None if doc["pick"] is None else doc["pick"]["point"],
                "contrasts": doc["matched_contrasts_htdev2_h9_minus_control"],
                "no_pick_reasons": doc["no_pick_reasons"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
