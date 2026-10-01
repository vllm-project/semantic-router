"""Decoder 0.8B fast track: successor items 1-8 for M12's 08b-RA against two bars (record
dec-08bfast-formal-lock-2026-10-01.md).

`bar-t1` is the stored formal run of DEV2.0-0.8B (node A, v3 50.236), the bar the coordinator named; `bar-e` is node
E's formal-path collection of the same weights (`f08-08b-C0`), the M12 preregistration's same-path parity run. The
rule is ops/m6/m6_successor.py's, evaluated once per bar; the bar-dependent items (1 v3, 2 H, 4 mlx-diag card-eligible,
6(b) reduced panels, 7 public 231) must pass against both, and the bar-free items (3 types, 5 tier gates, 6(a) the new
TRAIN file's exposure receipt, 8 C1 post-key) are read once.

    python3 f08_successor.py --run RUN --types TYPES.json --mlx-paired MLX-PAIRED.json \
        --mlx-paired-e MLX-PAIRED-bar-e.json --overlap overlap-effects.json --exposure RECEIPT \
        --public231 GATE-bar-t1.json --public231-e GATE-bar-e.json [--c1 SUMMARY.json] --output PREFIX
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "m6_successor", Path(__file__).resolve().parents[1] / "m6" / "m6_successor.py"
)
m6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m6)

SCHEMA = "dec-f08-successor/1"
TIER = "08b"
BARS = ("bar-t1", "bar-e")
BAR_ITEMS = (
    "1_v3_vs_bar",
    "2_H_vs_bar",
    "4_mlx_card_eligible",
    "6b_reduced_panels",
    "7_jevbench_public231",
)
ITEMS = (*m6.RULE_1_7, "8_c1_postkey")


def evaluate_bar(bar: str, run: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """m6_successor.evaluate with `bar` as the paired bar (PAIRED-vs-<bar>.json, the guard's right name, the
    overlap pair `<run> - <bar>`)."""
    saved = m6.BAR
    m6.BAR = bar
    try:
        return m6.evaluate(
            TIER,
            run,
            inputs["types"],
            inputs["mlx"][bar],
            inputs["overlap"],
            inputs["exposures"],
            inputs["public"][bar],
            inputs["c1"],
        )
    finally:
        m6.BAR = saved


def two_bar(run: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    views = {bar: evaluate_bar(bar, run, inputs) for bar in BARS}
    first = views[BARS[0]]
    items: dict[str, Any] = {}
    for key in ITEMS:
        if key in BAR_ITEMS:
            items[key] = {
                "pass": m6.all_true([views[b]["items"][key]["pass"] for b in BARS]),
                **{b: views[b]["items"][key] for b in BARS},
            }
        else:
            items[key] = first["items"][key]
    status_1_7 = m6.verdict([items[k]["pass"] for k in m6.RULE_1_7])
    status = m6.verdict([items[k]["pass"] for k in ITEMS])
    return {
        "schema": SCHEMA,
        "tier": TIER,
        "run": str(run),
        "name": first["name"],
        "point": first["point"],
        "revision": first["revision"],
        "bars": list(BARS),
        "rule": "m6_successor items per bar; items 1, 2, 4, 6(b), 7 must pass against both bars",
        "missing": sorted({m for b in BARS for m in views[b]["missing"]}),
        "v3": first["v3"],
        "T": first["T"],
        "H": first["H"],
        "items": items,
        "status_1_7": status_1_7,
        "status": status,
        "per_bar": {
            b: {"status_1_7": views[b]["status_1_7"], "status": views[b]["status"]}
            for b in BARS
        },
        "report_only": first["report_only"],
        "inputs": {b: views[b]["inputs"] for b in BARS},
    }


def fmt_ci(c: dict[str, float] | None, digits: int = 2) -> str:
    return "—" if not c else f"[{c['low']:+.{digits}f}, {c['high']:+.{digits}f}]"


def render(r: dict[str, Any]) -> str:
    it = r["items"]
    lines = [
        f"# 0.8B fast track successor rule — {r['name']} (two bars)",
        "",
        f"Status: **{r['status']}** (items 1–7: {r['status_1_7']}). v3 {r['v3']}, T {r['T']}, H {r['H']}. "
        "Post-key same-panel evidence. bar-t1 = the stored DEV2.0-0.8B formal run; bar-e = its node-E collection.",
        "",
        "| Item | Pass | bar-t1 | bar-e |",
        "| --- | --- | --- | --- |",
    ]
    for key in ITEMS:
        if key not in BAR_ITEMS:
            continue
        cells = []
        for b in BARS:
            v = it[key][b]
            if key == "1_v3_vs_bar":
                cells.append(
                    f"{v['pass']} {v.get('delta', 0):+.3f} {fmt_ci(v.get('ci95'))}"
                )
            elif key == "2_H_vs_bar":
                cells.append(f"{v['pass']} {fmt_ci(v.get('H_ci95'), 4)}")
            elif key == "4_mlx_card_eligible":
                cells.append(
                    f"{v['pass']} {v.get('delta', 0):+.4f} {fmt_ci(v.get('ci95'), 4)}"
                )
            elif key == "6b_reduced_panels":
                cells.append(
                    f"{v['pass']} {fmt_ci((v.get('rule1_v3_vs_bar') or {}).get('ci95'))}"
                )
            else:
                cells.append(
                    f"{v['pass']} {v.get('correct')} vs {v.get('bar_correct')} {v.get('verdict')}"
                    if "verdict" in v
                    else f"{v['pass']} {v.get('reason')}"
                )
        lines.append(f"| {key} | {it[key]['pass']} | {cells[0]} | {cells[1]} |")
    for key in ITEMS:
        if key in BAR_ITEMS:
            continue
        lines.append(f"| {key} | {it[key]['pass']} | (bar-free) | |")
    if r["missing"]:
        lines += ["", f"Missing inputs: {r['missing']}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--types", type=Path, required=True)
    p.add_argument("--mlx-paired", type=Path, required=True)
    p.add_argument("--mlx-paired-e", type=Path, required=True)
    p.add_argument("--overlap", type=Path, required=True)
    p.add_argument("--exposure", action="append", default=[])
    p.add_argument("--public231", type=Path, required=True)
    p.add_argument("--public231-e", type=Path, required=True)
    p.add_argument("--c1", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    inputs = {
        "types": m6.load(args.types),
        "mlx": {
            "bar-t1": m6.load(args.mlx_paired),
            "bar-e": m6.load(args.mlx_paired_e),
        },
        "overlap": m6.load(args.overlap),
        "exposures": [(x, m6.load(Path(x))) for x in args.exposure],
        "public": {
            "bar-t1": m6.load(args.public231),
            "bar-e": m6.load(args.public231_e),
        },
        "c1": m6.load(args.c1),
    }
    out = two_bar(args.run, inputs)
    out["input_files"] = {
        k: (m6.sha(v) if v is not None and Path(v).is_file() else None)
        for k, v in (
            ("types", args.types),
            ("mlx_paired", args.mlx_paired),
            ("mlx_paired_e", args.mlx_paired_e),
            ("overlap", args.overlap),
            ("public231", args.public231),
            ("public231_e", args.public231_e),
            ("c1", args.c1),
        )
    }
    Path(f"{args.output}.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n"
    )
    Path(f"{args.output}.md").write_text(render(out))
    print(
        json.dumps(
            {
                "name": out["name"],
                "status_1_7": out["status_1_7"],
                "status": out["status"],
                "items": {k: v["pass"] for k, v in out["items"].items()},
                "per_bar": out["per_bar"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
