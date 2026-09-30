"""Milestone 5 development gates for the ~27B track (host CPU; never a release or formal score).

Every readout directory holds ``READOUT.json`` from ``v2.eval.dev_readout`` and the three panels' native
predictions in ``output/`` (``typed-dev``, ``css-pilot``, ``ht-dev2``), all collected on node B at 32,768 tokens on
the kernel path from a fresh copy of DEV2.0-27B's scored cache. The reference is M4-A20r's readout of the same three
panels (``m5-htdev2.sh`` with ``PANELS=typed-dev,css-pilot,ht-dev2``). Gates of the preregistration
(``records/m5-prereg-2026-09-30.md``, "Candidates and finalists"):

1. collapse: ``m4_guard.collapse`` (M4 amendment 1) with A20r as the incumbent;
2. HT-DEV v2 not FLAG against A20r: ``v2.eval.dev_readout.htdev2_block`` with A20r's predictions as the paired
   reference (FLAG at delta <= -0.02);
3. typed guard: T_dev >= T_dev(A20r) - 0.03;
4. proxy guard: P_dev = 100*sqrt(T_dev*H_pilot) >= the best P_dev of A20r and the candidates - 8.

Per-type counts, T_dev, H_pilot and P_dev are recomputed from the predictions and must equal ``READOUT.json``.
Development results never rank candidates; every candidate passing all four goes formal (at most three).

    python3 -m v2.27b.m5.m5_devgates --reference A20r=DIR --candidate M5-FF20=DIR [...] --output GATES.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from v2.eval import dev_readout
from v2.eval import panels as panel_registry

guard = importlib.import_module("v2.27b.m4_guard")
contrast = importlib.import_module("v2.27b.contrast")

SCHEMA = "decision2-27b-m5-devgates/1"
TYPED_GUARD = 0.03
PROXY_DROP = guard.PROXY_DROP
TYPES = guard.TYPES


def values_of(
    name: str, directory: Path, panels: Any, panel_root: Path
) -> dict[str, Any]:
    record = guard.read_json(directory / "READOUT.json")
    typed = directory / "output" / "typed-dev.predictions.jsonl"
    css = directory / "output" / "css-pilot.predictions.jsonl"
    for key, path in (("typed_dev", typed), ("css_pilot", css)):
        if record[key]["predictions_sha256"] != panel_registry.sha_file(path):
            raise ValueError(f"{name}: READOUT.json {key} scored other predictions")
    outcome = guard.typed_cells(panels.dev, typed)
    cells, families = outcome["cells"], outcome["families"]
    by_type = {}
    for kind in TYPES:
        cell = cells[kind]
        recorded = record["typed_dev"]["by_type"][kind]
        if (cell["correct"], cell["n"]) != (recorded["correct"], recorded["n"]):
            raise ValueError(f"{name}: typed-DEV {kind} differs from READOUT.json")
        categories = dict(Counter(cell["categories"]).most_common())
        by_type[kind] = {
            "n": cell["n"],
            "correct": cell["correct"],
            "accuracy": cell["correct"] / cell["n"],
            "invalid": cell["n"] - cell["valid"],
            "answer_categories": categories,
            "distinct_categories": len(categories),
        }
    slots = sum(c["n"] for c in by_type.values())
    typed_invalid = sum(c["invalid"] for c in by_type.values())
    t_dev = statistics.fmean(c / n for c, n in families.values())
    guard.agree(f"{name} T_dev", t_dev, record["typed_dev"]["T_dev"])
    choices = panels.css_choices(css)
    h_pilot = contrast.css_macro(panels, choices, panels.task_items)["H_pilot"]
    guard.agree(f"{name} H_pilot", h_pilot, record["css_pilot"]["H_pilot"])
    p_dev = 100 * math.sqrt(t_dev * h_pilot)
    guard.agree(f"{name} P_dev", p_dev, record["development_proxy"])
    css_invalid = sum(c is None for c in choices)
    return {
        "readout": str(directory / "READOUT.json"),
        "label": record.get("label"),
        "P_dev": p_dev,
        "T_dev": t_dev,
        "H_pilot": h_pilot,
        "typed_dev": {
            "slots": slots,
            "invalid": typed_invalid,
            "invalid_rate": typed_invalid / slots,
            "by_type": by_type,
            "by_family": {f: c / n for f, (c, n) in sorted(families.items())},
        },
        "css_pilot": {
            "items": len(choices),
            "invalid": css_invalid,
            "invalid_rate": css_invalid / len(choices),
        },
        "htdev2_predictions": str(directory / "output" / "ht-dev2.predictions.jsonl"),
    }


def decide(
    reference: tuple[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    panel_root: Path,
) -> dict:
    ref_name, ref = reference
    pool = {ref_name: ref["P_dev"], **{n: v["P_dev"] for n, v in candidates.items()}}
    best = max(pool, key=lambda n: pool[n])
    out = {}
    for name, value in candidates.items():
        flags = guard.collapse(value, ref)
        ht = dev_readout.htdev2_block(
            panel_root,
            Path(value["htdev2_predictions"]),
            Path(ref["htdev2_predictions"]),
        )
        typed_ok = value["T_dev"] >= ref["T_dev"] - TYPED_GUARD - guard.EPSILON
        gap = pool[best] - value["P_dev"]
        gates = {
            "1_collapse": {"flags": flags, "pass": not flags},
            "2_htdev2_vs_reference": {
                **ht["vs_reference"],
                "H_dev2": ht["H_dev2"],
                "pass": ht["vs_reference"]["verdict"] != "FLAG",
            },
            "3_typed_guard": {
                "T_dev": value["T_dev"],
                "floor": ref["T_dev"] - TYPED_GUARD,
                "pass": typed_ok,
            },
            "4_proxy_guard": {
                "P_dev": value["P_dev"],
                "best": best,
                "gap": gap,
                "pass": gap < PROXY_DROP,
            },
        }
        out[name] = {
            **value,
            "gates": gates,
            "finalist": all(g["pass"] for g in gates.values()),
        }
    return {
        "proxy_pool": {"P_dev": pool, "best": best},
        "candidates": out,
        "finalists": [n for n, c in out.items() if c["finalist"]],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel-root", type=Path, default=panel_registry.DEFAULT_ROOT)
    parser.add_argument("--reference", required=True, help="NAME=READOUT_DIR (A20r)")
    parser.add_argument(
        "--candidate", action="append", required=True, help="NAME=READOUT_DIR"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot", "ht-dev2"])
    panels = contrast.Panels(
        panel_registry.path(args.panel_root, "typed-dev", "gold"),
        panel_registry.path(args.panel_root, "css-pilot", "gold"),
    )
    pairs = guard.pairs([args.reference, *args.candidate])
    ref_name = args.reference.partition("=")[0]
    values = {n: values_of(n, p, panels, args.panel_root) for n, p in pairs.items()}
    reference = (ref_name, values.pop(ref_name))
    result = {
        "schema": SCHEMA,
        "label": "development gates (kernel path, 32K); never a release or formal score",
        "rules": {
            "collapse": "m4_guard.collapse (M4 amendment 1) against the reference",
            "htdev2": "not FLAG (delta <= -0.02) against the reference's HT-DEV v2 predictions",
            "typed_guard": f"T_dev >= reference T_dev - {TYPED_GUARD}",
            "proxy_guard": f"P_dev < {PROXY_DROP} below the best of the reference and the candidates",
        },
        "reference": {"name": ref_name, **reference[1]},
        **decide(reference, values, args.panel_root),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    summary = {
        n: {"finalist": c["finalist"], **{g: v["pass"] for g, v in c["gates"].items()}}
        for n, c in result["candidates"].items()
    }
    print(
        json.dumps({"finalists": result["finalists"], "gates": summary}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
