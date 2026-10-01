"""Decoder M14: successor items 1-8 for a formal finalist against two bars (amendment 1,
dec-m14-amendment-1-2026-10-01.md; M13 amendment 2's two-bar rule, ops/m13/m13_successor.py, per tier).

The first bar is the tier's stored formal run of its current release (4B: the released LH, `bar-lh`; 0.8B:
DEV2.0-0.8B, `bar-t1`); the second is the same weights collected on the M14 node-B formal path (`bar-b`). The rule
is ops/m6/m6_successor.py's, evaluated once per bar; the bar-dependent items (1 v3, 2 H, 4 mlx-diag card-eligible,
6(b) reduced panels, 7 public 231) must pass against both, and the bar-free items (3 types, 5 tier gates, 6(a) the
TRAIN file's exposure receipt, 8 C1 post-key) are read once.

    python3 m14_successor.py --tier 4b --run RUN --types TYPES.json \
        --bar bar-lh=MLX-PAIRED-bar-lh.json,GATE-bar-lh.json --bar bar-b=MLX-PAIRED-bar-b.json,GATE-bar-b.json \
        --overlap overlap-effects.json --exposure RECEIPT [--c1 SUMMARY.json] --output PREFIX
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

SCHEMA = "dec-m14-successor/1"
BAR_ITEMS = (
    "1_v3_vs_bar",
    "2_H_vs_bar",
    "4_mlx_card_eligible",
    "6b_reduced_panels",
    "7_jevbench_public231",
)
ITEMS = (*m6.RULE_1_7, "8_c1_postkey")
BAR_NOTES = {
    "4b": "bar-lh = the released LH's stored formal run (T = 1); bar-b = its node-B M14 collection",
    "08b": "bar-t1 = the stored DEV2.0-0.8B formal run; bar-b = its node-B M14 collection",
}


def evaluate_bar(
    tier: str, bar: str, run: Path, inputs: dict[str, Any]
) -> dict[str, Any]:
    saved = m6.BAR
    m6.BAR = bar
    try:
        return m6.evaluate(
            tier,
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


def two_bar(
    tier: str, bars: list[str], run: Path, inputs: dict[str, Any]
) -> dict[str, Any]:
    views = {bar: evaluate_bar(tier, bar, run, inputs) for bar in bars}
    first = views[bars[0]]
    items: dict[str, Any] = {}
    for key in ITEMS:
        if key in BAR_ITEMS:
            items[key] = {
                "pass": m6.all_true([views[b]["items"][key]["pass"] for b in bars]),
                **{b: views[b]["items"][key] for b in bars},
            }
        else:
            items[key] = first["items"][key]
    return {
        "schema": SCHEMA,
        "tier": tier,
        "run": str(run),
        "name": first["name"],
        "point": first["point"],
        "revision": first["revision"],
        "bars": bars,
        "bar_note": BAR_NOTES[tier],
        "rule": "m6_successor items per bar; items 1, 2, 4, 6(b), 7 must pass against both bars",
        "missing": sorted({m for b in bars for m in views[b]["missing"]}),
        "v3": first["v3"],
        "T": first["T"],
        "H": first["H"],
        "items": items,
        "status_1_7": m6.verdict([items[k]["pass"] for k in m6.RULE_1_7]),
        "status": m6.verdict([items[k]["pass"] for k in ITEMS]),
        "per_bar": {
            b: {"status_1_7": views[b]["status_1_7"], "status": views[b]["status"]}
            for b in bars
        },
        "report_only": first["report_only"],
        "inputs": {b: views[b]["inputs"] for b in bars},
    }


def fmt_ci(c: dict[str, float] | None, digits: int = 2) -> str:
    return "—" if not c else f"[{c['low']:+.{digits}f}, {c['high']:+.{digits}f}]"


def cell(key: str, v: dict[str, Any]) -> str:
    if key == "1_v3_vs_bar":
        return f"{v['pass']} {v.get('delta', 0):+.3f} {fmt_ci(v.get('ci95'))}"
    if key == "2_H_vs_bar":
        return f"{v['pass']} {fmt_ci(v.get('H_ci95'), 4)}"
    if key == "4_mlx_card_eligible":
        return f"{v['pass']} {v.get('delta', 0):+.4f} {fmt_ci(v.get('ci95'), 4)}"
    if key == "6b_reduced_panels":
        return f"{v['pass']} {fmt_ci((v.get('rule1_v3_vs_bar') or {}).get('ci95'))}"
    if "verdict" in v:
        return f"{v['pass']} {v.get('correct')} vs {v.get('bar_correct')} {v.get('verdict')}"
    return f"{v['pass']} {v.get('reason')}"


def render(r: dict[str, Any]) -> str:
    it, bars = r["items"], r["bars"]
    lines = [
        f"# Decoder M14 successor rule — {r['name']} (two bars)",
        "",
        f"Status: **{r['status']}** (items 1–7: {r['status_1_7']}). v3 {r['v3']}, T {r['T']}, H {r['H']}. "
        f"Post-key same-panel evidence. {r['bar_note']}.",
        "",
        "| Item | Pass | " + " | ".join(bars) + " |",
        "| --- | --- | " + " | ".join("---" for _ in bars) + " |",
    ]
    for key in ITEMS:
        if key in BAR_ITEMS:
            lines.append(
                f"| {key} | {it[key]['pass']} | "
                + " | ".join(cell(key, it[key][b]) for b in bars)
                + " |"
            )
    for key in ITEMS:
        if key not in BAR_ITEMS:
            lines.append(
                f"| {key} | {it[key]['pass']} | (bar-free) |" + " |" * (len(bars) - 1)
            )
    if r["missing"]:
        lines += ["", f"Missing inputs: {r['missing']}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=sorted(BAR_NOTES), required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--types", type=Path, required=True)
    p.add_argument(
        "--bar",
        action="append",
        required=True,
        help="NAME=MLX-PAIRED.json,PUBLIC231-GATE.json",
    )
    p.add_argument("--overlap", type=Path, required=True)
    p.add_argument("--exposure", action="append", default=[])
    p.add_argument("--c1", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    bars, mlx, public, files = [], {}, {}, {}
    for spec in args.bar:
        name, _, rest = spec.partition("=")
        mlx_path, _, pub_path = rest.partition(",")
        if not name or not mlx_path or not pub_path:
            p.error(
                f"--bar {spec!r}: expected NAME=MLX-PAIRED.json,PUBLIC231-GATE.json"
            )
        bars.append(name)
        mlx[name], public[name] = m6.load(Path(mlx_path)), m6.load(Path(pub_path))
        files[f"mlx_paired_{name}"], files[f"public231_{name}"] = Path(mlx_path), Path(
            pub_path
        )
    if len(bars) != 2 or len(set(bars)) != 2:
        p.error("exactly two distinct --bar entries")
    inputs = {
        "types": m6.load(args.types),
        "mlx": mlx,
        "overlap": m6.load(args.overlap),
        "exposures": [(x, m6.load(Path(x))) for x in args.exposure],
        "public": public,
        "c1": m6.load(args.c1),
    }
    out = two_bar(args.tier, bars, args.run, inputs)
    files.update(types=args.types, overlap=args.overlap, c1=args.c1)
    out["input_files"] = {
        k: (m6.sha(v) if v is not None and Path(v).is_file() else None)
        for k, v in sorted(files.items())
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
