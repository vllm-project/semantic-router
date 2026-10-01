"""Stage A screen rules of the 27B MoE milestone (host CPU; development readouts only).

Inputs: each cell's T = 1 readout at checkpoint 892 (``moe-readout.sh``: typed DEV, CSS
pilot and HT-DEV v2 against the dense matched reference) and the dense matched reference's
own readout (M4-A20r-s1 checkpoint 892). Rules (preregistration, Stage A; amendment 3):

1. collapse: a typed-DEV type with one answer category for the whole type or >= 95% of its
   answers; Choice or Score at or below chance where the reference is >= 10 points above it
   (M4's amended guard); invalid answers above 1% on typed DEV or the CSS pilot;
2. HT-DEV v2 FLAG (delta <= -0.02) against the dense matched reference;
3. proxy: P_dev at least 8 below the best of the cells and the reference;
4. at most one variant per family (higher P_dev; within |dP| < 2 the higher H_dev2), at most
   two cells; the best survivor (same ordering) gets seed 2.
Writes SCREEN.json once. Never a release or formal score.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

guard = importlib.import_module("v2.27b.m4_guard")
FAMILY = {"gemma-4-26B-A4B-it": "gemma", "gemma-4-26B-A4B": "gemma"}
PROXY_DROP = 8.0
TIE = 2.0


def family(base: str) -> str:
    return FAMILY.get(base, "qwen")


def values(name: str, readout: Path, panels: Any) -> dict[str, Any]:
    record = json.loads(readout.read_text(encoding="utf-8"))
    directory = readout.parent
    cells = guard.typed_cells(
        panels.dev, directory / "output" / "typed-dev.predictions.jsonl"
    )["cells"]
    by_type = {}
    for kind in guard.TYPES:
        cell = cells.get(kind)
        recorded = record["typed_dev"]["by_type"].get(kind, {"correct": 0, "n": 0})
        if cell is None or (cell["correct"], cell["n"]) != (
            recorded["correct"],
            recorded["n"],
        ):
            raise ValueError(f"{name}: typed-DEV {kind} differs from READOUT.json")
        categories = dict(cell["categories"].most_common())
        answered = sum(categories.values())
        by_type[kind] = {
            "n": cell["n"],
            "accuracy": cell["correct"] / cell["n"],
            "answer_categories": categories,
            "distinct_categories": len(categories),
            "top_share": (
                max(categories.values(), default=0) / answered if answered else 1.0
            ),
        }
    choices = panels.css_choices(directory / "output" / "css-pilot.predictions.jsonl")
    slots = sum(c["n"] for c in by_type.values())
    htdev2 = record.get("htdev2") or {}
    return {
        "readout": str(readout),
        "P_dev": record["development_proxy"],
        "T_dev": record["typed_dev"]["T_dev"],
        "H_pilot": record["css_pilot"]["H_pilot"],
        "H_dev2": htdev2.get("H_dev2"),
        "htdev2_vs_reference": htdev2.get("vs_reference"),
        "by_type": by_type,
        "typed_invalid_rate": record["typed_dev"]["invalid_or_missing"] / slots,
        "css_invalid_rate": record["css_pilot"]["invalid_or_missing"] / len(choices),
    }


def collapse(cell: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    flags = []
    for kind in guard.TYPES:
        own, ref = cell["by_type"][kind], reference["by_type"][kind]
        chance = guard.CHANCE[kind]
        if (
            ref["accuracy"] >= chance + guard.CHANCE_MARGIN
            and own["accuracy"] <= chance
        ):
            flags.append(
                f"typed-DEV {kind} accuracy {own['accuracy']:.4f} at or below chance {chance}"
            )
        if own["distinct_categories"] <= 1 or own["top_share"] >= guard.MODAL_MAX:
            flags.append(f"typed-DEV {kind} top answer share {own['top_share']:.4f}")
    for key in ("typed_invalid_rate", "css_invalid_rate"):
        if cell[key] > guard.INVALID_MAX:
            flags.append(f"{key} {cell[key]:.4f} above {guard.INVALID_MAX}")
    return flags


def order(cells: dict[str, dict[str, Any]], names: list[str]) -> list[str]:
    """Best first: higher P_dev; within |dP| < TIE of the best, the higher H_dev2."""
    ranked = sorted(names, key=lambda n: -cells[n]["P_dev"])
    if len(ranked) > 1 and cells[ranked[0]]["P_dev"] - cells[ranked[1]]["P_dev"] < TIE:
        top = [n for n in ranked if cells[ranked[0]]["P_dev"] - cells[n]["P_dev"] < TIE]
        top.sort(key=lambda n: -(cells[n]["H_dev2"] or 0.0))
        ranked = top + [n for n in ranked if n not in top]
    return ranked


def decide(
    reference: dict[str, Any], cells: dict[str, dict[str, Any]], bases: dict[str, str]
) -> dict[str, Any]:
    best_pool = max([reference["P_dev"], *(c["P_dev"] for c in cells.values())])
    verdicts = {}
    for name, cell in cells.items():
        reasons = [f"collapse: {flag}" for flag in collapse(cell, reference)]
        verdict = (cell["htdev2_vs_reference"] or {}).get("verdict")
        if verdict is None:
            reasons.append("HT-DEV v2 comparison with the dense reference is missing")
        elif verdict == "FLAG":
            reasons.append(
                f"HT-DEV v2 FLAG vs the dense reference ({cell['htdev2_vs_reference']['delta']:+.4f})"
            )
        gap = best_pool - cell["P_dev"]
        if gap >= PROXY_DROP:
            reasons.append(f"proxy gap {gap:.2f} >= {PROXY_DROP}")
        verdicts[name] = {
            "family": family(bases[name]),
            "proxy_gap": gap,
            "reasons": reasons,
        }
    alive = [n for n, v in verdicts.items() if not v["reasons"]]
    kept: list[str] = []
    for name in order(cells, alive):
        if (
            any(verdicts[k]["family"] == verdicts[name]["family"] for k in kept)
            or len(kept) >= 2
        ):
            verdicts[name]["reasons"].append(
                "rule 4: another variant of its family ranks higher (or two cells continue)"
            )
            continue
        kept.append(name)
    for name, verdict in verdicts.items():
        verdict["continues"] = name in kept
    return {
        "best_pool_P_dev": best_pool,
        "cells": {n: {**cells[n], **verdicts[n], "base": bases[n]} for n in cells},
        "continue": kept,
        "seed2": kept[0] if kept else None,
        "stop": sorted(n for n in cells if n not in kept),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--panel-root", type=Path, default=guard.panel_registry.DEFAULT_ROOT
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="dense matched reference READOUT.json",
    )
    parser.add_argument(
        "--cell", action="append", required=True, help="NAME=BASE=READOUT.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    guard.panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot"])
    panels = guard.contrast.Panels(
        guard.panel_registry.path(args.panel_root, "typed-dev", "gold"),
        guard.panel_registry.path(args.panel_root, "css-pilot", "gold"),
    )
    bases, cells = {}, {}
    for spec in args.cell:
        name, base, path = spec.split("=", 2)
        bases[name] = base
        cells[name] = values(name, Path(path), panels)
    reference = values("dense-reference", args.reference, panels)
    result = {
        "schema": "decision2-27b-moe-screen/1",
        "scope": "Stage A development screen at checkpoint 892; never a release or formal score",
        "reference": reference,
        **decide(reference, cells, bases),
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("continue", "seed2", "stop")}))


if __name__ == "__main__":
    main()
