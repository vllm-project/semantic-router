"""Milestone 6 development gates for the ~27B track (host CPU; never a release, formal or Index score).

Readouts (``readouts/<name>``: ``READOUT.json`` and typed DEV, CSS pilot, HT-DEV v2 predictions, node B, 32,768 tokens,
kernel path, fresh copies of DEV2.0-27B's scored cache) are read with M5's ``m5_devgates.values_of``. The references
are M5's collections of A20r (the successor reference) and M5-L128 (the base recipe). Slices (``slices/<name>/probs``)
come from ``kernel_readout slices``. Gates of ``records/m6-prereg-2026-10-01.md`` ("Candidates and development gates"):

1. G1 collapse: ``m4_guard.collapse`` (M4 amendment 1) against A20r;
2. G2 human transfer: HT-DEV v2 vs A20r not FLAG (GAIN or TIE; ``dev_readout.htdev2_block``);
3. G3 typed floor: T_dev >= T_dev(M5-L128) - 0.03, and Choice / Score typed-DEV accuracy each >= M5-L128's - 0.03;
4. G4 Noul floor: typed-DEV Noul (``rule_precedence``) accuracy >= M5-L128's - 0.03;
5. G5 PN1 guard vs A20r (``m6_slices.pn1_report``): a gate only when step 0 validated it (``--pn1-validation``: the
   M5-L128 - A20r report, validated when its clean gold-no delta is > 0), otherwise reported;
6. G6 breadth: IB DEV B_dev vs A20r paired lower bound > 0 (``m6_slices.breadth_report``).

P_dev and M5's proxy guard (P_dev >= best - 8) are reported only. Every candidate passing all gates is a finalist, at
most two per stage; with more passers, HT-DEV v2 GAIN first, then the larger B_dev delta.

    python3 -m v2.27b.m6.m6_devgates --root /data/dev2/runs/27b/m6 --pn1-rows PN1 --ib-rows IB \
        [--in-distribution w2c ...] --pn1-validation SLICES/M5-L128/pn1-vs-A20r.json NAME... --output GATES.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from v2.eval import dev_readout
from v2.eval import panels as panel_registry

guard = importlib.import_module("v2.27b.m4_guard")
contrast = importlib.import_module("v2.27b.contrast")
m5_devgates = importlib.import_module("v2.27b.m5.m5_devgates")
m6_slices = importlib.import_module("v2.27b.m6.m6_slices")

SCHEMA = "decision2-27b-m6-devgates/1"
FLOOR = 0.03
MAX_FINALISTS = 2
A20R_READOUT = Path("/data/dev2/runs/27b/m5/readouts/m4-a20r-soup")
L128_READOUT = Path("/data/dev2/runs/27b/m5/readouts/M5-L128")


def pn1_validated(report: dict[str, Any]) -> bool:
    if (report["candidate"], report["reference"]) != ("M5-L128", "A20r"):
        raise ValueError("the step-0 report must be M5-L128 against A20r")
    return report["delta"]["clean_no"] > 0


def floors(value: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    by, ref = value["typed_dev"]["by_type"], base["typed_dev"]["by_type"]
    eps = guard.EPSILON
    typed = {
        "T_dev": value["T_dev"],
        "T_dev_floor": base["T_dev"] - FLOOR,
        **{f"{k}_accuracy": by[k]["accuracy"] for k in ("choice", "score")},
        **{f"{k}_floor": ref[k]["accuracy"] - FLOOR for k in ("choice", "score")},
    }
    typed["pass"] = value["T_dev"] >= typed["T_dev_floor"] - eps and all(
        by[k]["accuracy"] >= typed[f"{k}_floor"] - eps for k in ("choice", "score")
    )
    noul = {
        "noul_accuracy": by["noul"]["accuracy"],
        "floor": ref["noul"]["accuracy"] - FLOOR,
        "answer_categories": by["noul"]["answer_categories"],
    }
    noul["pass"] = by["noul"]["accuracy"] >= noul["floor"] - eps
    return {"typed": typed, "noul": noul}


def decide(
    a20r: dict[str, Any],
    l128: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    slices: dict[str, dict[str, Any]],
    validated: bool,
    panel_root: Path,
) -> dict[str, Any]:
    pool = {"A20r": a20r["P_dev"], **{n: v["P_dev"] for n, v in candidates.items()}}
    best = max(pool, key=lambda n: pool[n])
    out = {}
    for name, value in candidates.items():
        flags = guard.collapse(value, a20r)
        ht = dev_readout.htdev2_block(
            panel_root,
            Path(value["htdev2_predictions"]),
            Path(a20r["htdev2_predictions"]),
        )
        floor = floors(value, l128)
        pn1, breadth = slices[name]["pn1"], slices[name]["breadth"]
        gates = {
            "G1_collapse": {"flags": flags, "pass": not flags},
            "G2_htdev2_vs_A20r": {
                **ht["vs_reference"],
                "H_dev2": ht["H_dev2"],
                "pass": ht["vs_reference"]["verdict"] != "FLAG",
            },
            "G3_typed_floor_vs_L128": floor["typed"],
            "G4_noul_floor_vs_L128": floor["noul"],
            "G5_pn1_guard_vs_A20r": {
                "validated": validated,
                "delta": pn1["delta"],
                "delta_ci95": pn1["delta_ci95"],
                "reasons": pn1["reasons"],
                "pass": pn1["pass"] or not validated,
            },
            "G6_breadth_vs_A20r": {
                "B_dev": breadth["B_dev"],
                "delta": breadth["delta"],
                "delta_ci95": breadth["delta_ci95"],
                "pass": breadth["pass"],
            },
        }
        out[name] = {
            **value,
            "gates": gates,
            "report_only": {
                "P_dev": value["P_dev"],
                "proxy_best": best,
                "proxy_gap": pool[best] - value["P_dev"],
                "proxy_guard": pool[best] - value["P_dev"] < guard.PROXY_DROP,
                "pn1_gate_applied": validated,
            },
            "pass": all(g["pass"] for g in gates.values()),
        }
    passers = [n for n, c in out.items() if c["pass"]]
    passers.sort(
        key=lambda n: (
            out[n]["gates"]["G2_htdev2_vs_A20r"]["verdict"] != "GAIN",
            -out[n]["gates"]["G6_breadth_vs_A20r"]["delta"],
            n,
        )
    )
    return {
        "proxy_pool": {"P_dev": pool, "best": best},
        "candidates": out,
        "passers": passers,
        "finalists": passers[:MAX_FINALISTS],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, default=panel_registry.DEFAULT_ROOT)
    parser.add_argument("--a20r-readout", type=Path, default=A20R_READOUT)
    parser.add_argument("--l128-readout", type=Path, default=L128_READOUT)
    parser.add_argument("--pn1-rows", type=Path, required=True)
    parser.add_argument("--ib-rows", type=Path, required=True)
    parser.add_argument("--in-distribution", action="append", default=[])
    parser.add_argument("--pn1-validation", type=Path, required=True)
    parser.add_argument(
        "--ref-pn1", default="A20r", help="slices NAME of A20r's PN1 dev"
    )
    parser.add_argument(
        "--ref-ib", default="A20r-ib1", help="slices NAME of A20r's IB DEV"
    )
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot", "ht-dev2"])
    panels = contrast.Panels(
        panel_registry.path(args.panel_root, "typed-dev", "gold"),
        panel_registry.path(args.panel_root, "css-pilot", "gold"),
    )

    def values(name: str, directory: Path) -> dict[str, Any]:
        return m5_devgates.values_of(name, directory, panels, args.panel_root)

    a20r = values("A20r", args.a20r_readout)
    l128 = values("M5-L128", args.l128_readout)
    candidates = {n: values(n, args.root / "readouts" / n) for n in args.names}
    pn1_rows = m6_slices.read_jsonl(args.pn1_rows)
    ib_rows = m6_slices.read_jsonl(args.ib_rows)
    sdir = args.root / "slices"

    def probs(name: str, slice_name: str, rows: list[dict[str, Any]], label: str = ""):
        path = sdir / name / "probs" / f"{slice_name}.probs.jsonl"
        return label or name, m6_slices.read_probs(path, rows)

    ref_pn1 = probs(args.ref_pn1, "pn1", pn1_rows, "A20r")
    ref_ib = probs(args.ref_ib, "ib", ib_rows, "A20r")
    slices = {
        n: {
            "pn1": m6_slices.pn1_report(pn1_rows, probs(n, "pn1", pn1_rows), ref_pn1),
            "breadth": m6_slices.breadth_report(
                ib_rows, probs(n, "ib", ib_rows), ref_ib, args.in_distribution
            ),
        }
        for n in args.names
    }
    validation = json.loads(args.pn1_validation.read_text(encoding="utf-8"))
    validated = pn1_validated(validation)
    result = {
        "schema": SCHEMA,
        "label": "M6 development gates (kernel path, 32K); never a release, formal or Index score",
        "rules": {
            "G1": "m4_guard.collapse (M4 amendment 1) against A20r",
            "G2": "HT-DEV v2 vs A20r not FLAG (delta > -0.02)",
            "G3": f"T_dev >= T_dev(M5-L128) - {FLOOR}; Choice / Score accuracy >= M5-L128's - {FLOOR}",
            "G4": f"typed-DEV Noul accuracy >= M5-L128's - {FLOOR}",
            "G5": "PN1 dev vs A20r: hop delta >= -0.03 and clean gold-no delta <= 0 (gate only if step 0 validated it)",
            "G6": "IB DEV B_dev vs A20r: paired group-bootstrap lower bound > 0",
            "finalists": f"all passers, at most {MAX_FINALISTS}: HT-DEV v2 GAIN first, then the larger B_dev delta",
        },
        "references": {
            "A20r": {
                "readout": str(args.a20r_readout),
                "T_dev": a20r["T_dev"],
                "P_dev": a20r["P_dev"],
            },
            "M5-L128": {
                "readout": str(args.l128_readout),
                "T_dev": l128["T_dev"],
                "P_dev": l128["P_dev"],
            },
        },
        "pn1_validation": {
            "report": str(args.pn1_validation),
            "delta": validation["delta"],
            "delta_ci95": validation["delta_ci95"],
            "validated": validated,
        },
        "inputs_sha256": {
            "pn1_rows": m6_slices.sha_file(args.pn1_rows),
            "ib_rows": m6_slices.sha_file(args.ib_rows),
        },
        **decide(a20r, l128, candidates, slices, validated, args.panel_root),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    summary = {
        n: {"pass": c["pass"], **{g: v["pass"] for g, v in c["gates"].items()}}
        for n, c in result["candidates"].items()
    }
    print(
        json.dumps({"finalists": result["finalists"], "gates": summary}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
