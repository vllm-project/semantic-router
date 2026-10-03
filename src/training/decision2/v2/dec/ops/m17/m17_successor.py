"""Decoder M17: successor items for a formal finalist against two bars (prereg dec-m17-prereg-2026-10-02.md, "Formal and
both successor paths"): M14's two-bar evaluation of the classic items 1-8 (ops/m14/m14_successor.py, unchanged rule;
bar-lh = the released LH's stored formal run, bar-f = its node-F M17 collection), plus the Index path's public items:

  1'(a)  post-key v3 not significantly below LH: item 1's paired v3 delta 95% CI upper bound > 0 against both bars;
  6(b)'  the reduced-panel v3 delta's 95% CI upper bound > 0 against both bars;
  candidate  fails item 1 but passes 1'(a), 6(b)' and items 2-5, 6(a), 7: only then is the private Index run made, and
             its paired bootstrap decides 1'(b) (recorded here as pass / fail only, never a value).

    python3 m17_successor.py --tier 4b --run RUN --types TYPES.json \
        --bar bar-lh=MLX-PAIRED.json,GATE.json --bar bar-f=MLX-PAIRED.json,GATE.json \
        --overlap overlap-effects.json --exposure RECEIPT [--exposure RECEIPT] [--c1 SUMMARY.json] \
        [--index-1b PASS|FAIL] --output PREFIX
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "m14_successor", Path(__file__).resolve().parents[1] / "m14" / "m14_successor.py"
)
m14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m14)

m14.SCHEMA = "dec-m17-successor/1"
m14.BAR_NOTES.clear()
m14.BAR_NOTES.update(
    {
        "4b": "bar-lh = the released LH's stored formal run (T = 1); bar-f = its node-F M17 collection (m17-4b-LH)",
    }
)
REST = (
    "2_H_vs_bar",
    "3_types",
    "4_mlx_card_eligible",
    "5_tier_gates",
    "6a_exposure_new_files",
    "7_jevbench_public231",
)


def high(ci: dict[str, float] | None) -> float | None:
    return None if not ci else ci.get("high")


def index_path(out: dict[str, Any], index_1b: str | None) -> dict[str, Any]:
    items, bars = out["items"], out["bars"]
    v3_highs = {b: high(items["1_v3_vs_bar"][b].get("ci95")) for b in bars}
    red_highs = {
        b: high(
            (items["6b_reduced_panels"][b].get("rule1_v3_vs_bar") or {}).get("ci95")
        )
        for b in bars
    }
    a = all(h is not None and h > 0 for h in v3_highs.values())
    b6 = all(h is not None and h > 0 for h in red_highs.values())
    rest = all(items[k]["pass"] is True for k in REST)
    item1 = items["1_v3_vs_bar"]["pass"] is True
    candidate = (not item1) and a and b6 and rest
    b = None if index_1b is None else index_1b == "PASS"
    return {
        "rule": "item 1' = (a) paired v3 CI upper bound > 0 vs both bars and (b) the private Index delta's paired "
        "bootstrap 95% CI lower bound > 0; 6(b)' = reduced-panel v3 CI upper bound > 0 vs both bars; candidate = "
        "fails item 1, passes 1'(a), 6(b)' and items 2-5, 6(a), 7",
        "1p_a_v3_not_below": a,
        "1p_a_ci95_high": v3_highs,
        "6bp_reduced_not_below": b6,
        "6bp_ci95_high": red_highs,
        "items_2_5_6a_7": rest,
        "candidate": candidate,
        "1p_b_index_significantly_positive": b,
        "items_1p_7_pass": bool(candidate and b),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--index-1b", choices=("PASS", "FAIL"))
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    if "--index-1b" in argv:
        i = argv.index("--index-1b")
        del argv[i : i + 2]
    rc = m14.main(argv)
    if rc:
        return rc
    path = Path(f"{known.output}.json")
    out = json.loads(path.read_text())
    out["index_path"] = index_path(out, known.index_1b)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    ip = out["index_path"]
    md = Path(f"{known.output}.md")
    md.write_text(
        md.read_text()
        + "\n## Index path (public items)\n\n"
        + f"- 1'(a) v3 not significantly below LH (CI upper > 0, both bars): {ip['1p_a_v3_not_below']}\n"
        + f"- 6(b)' reduced panels not significantly below (both bars): {ip['6bp_reduced_not_below']}\n"
        + f"- items 2–5, 6(a), 7: {ip['items_2_5_6a_7']}\n"
        + f"- Index-path candidate: {ip['candidate']}; 1'(b) (private Index run): {ip['1p_b_index_significantly_positive']}\n"
    )
    print(
        json.dumps(
            {
                "index_path": {
                    k: ip[k]
                    for k in (
                        "1p_a_v3_not_below",
                        "6bp_reduced_not_below",
                        "candidate",
                        "items_1p_7_pass",
                    )
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
