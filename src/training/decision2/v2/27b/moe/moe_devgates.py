"""Stage B development gates of the 27B MoE milestone (host CPU; never a release or formal score).

Preregistration ``records/moe-prereg-2026-10-01.md``, "Development gates on each soup": a soup
goes to the formal runner only if its T = 1 readout (``moe-readout.sh``: typed DEV, CSS pilot and
HT-DEV v2 at 32,768 tokens on node B) passes

1. collapse: screen rule 1 (``screen_rules.collapse``) with A20r's readout as the reference;
2. HT-DEV v2 not FLAG against A20r: ``v2.eval.dev_readout.htdev2_block`` paired with A20r's own
   predictions (M5's ``A20r-ref``, H_dev2 .5655); FLAG at delta <= -0.02;
3. proxy: P_dev not 8 or more below A20r's (78.99).

Report only: M5's typed guard (T_dev >= T_dev(A20r) - 0.03) and per-type / per-family values.
HT-DEV v2 is recomputed and must equal the readout's own comparison when it used the same
reference. At most three soups go formal.

    python3 -m v2.27b.moe.moe_devgates --reference A20r=READOUT_DIR --candidate NAME=READOUT_DIR \
        [--candidate ...] --output DEVGATES.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

from v2.eval import dev_readout
from v2.eval import panels as panel_registry

screen = importlib.import_module("v2.27b.moe.screen_rules")
guard = screen.guard

SCHEMA = "decision2-27b-moe-devgates/1"
PROXY_DROP = 8.0
TYPED_GUARD_REPORT = 0.03
MAX_FINALISTS = 3


def readout(name: str, directory: Path, panels: Any) -> dict[str, Any]:
    value = screen.values(name, directory / "READOUT.json", panels)
    record = json.loads((directory / "READOUT.json").read_text(encoding="utf-8"))
    guard.agree(
        f"{name} P_dev",
        100 * math.sqrt(value["T_dev"] * value["H_pilot"]),
        value["P_dev"],
    )
    return {
        **value,
        "label": record.get("label"),
        "by_family": record["typed_dev"].get("by_family"),
        "htdev2_predictions": str(directory / "output" / "ht-dev2.predictions.jsonl"),
    }


def decide(
    reference: dict[str, Any], candidates: dict[str, dict[str, Any]], panel_root: Path
) -> dict[str, Any]:
    out = {}
    for name, value in candidates.items():
        flags = screen.collapse(value, reference)
        ht = dev_readout.htdev2_block(
            panel_root,
            Path(value["htdev2_predictions"]),
            Path(reference["htdev2_predictions"]),
        )
        vs = ht["vs_reference"]
        own = value.get("htdev2_vs_reference") or {}
        if own.get("reference_sha256") == vs["reference_sha256"]:
            guard.agree(f"{name} HT-DEV v2 delta", vs["delta"], own["delta"])
        gap = reference["P_dev"] - value["P_dev"]
        gates = {
            "1_collapse": {"flags": flags, "pass": not flags},
            "2_htdev2_not_flag_vs_A20r": {
                **vs,
                "H_dev2": ht["H_dev2"],
                "pass": vs["verdict"] != "FLAG",
            },
            "3_proxy_vs_A20r": {
                "P_dev": value["P_dev"],
                "reference_P_dev": reference["P_dev"],
                "gap": gap,
                "pass": gap < PROXY_DROP,
            },
        }
        out[name] = {
            **{k: v for k, v in value.items() if k != "htdev2_vs_reference"},
            "gates": gates,
            "report_only": {
                "typed_guard_m5": {
                    "T_dev": value["T_dev"],
                    "floor": reference["T_dev"] - TYPED_GUARD_REPORT,
                    "pass": value["T_dev"]
                    >= reference["T_dev"] - TYPED_GUARD_REPORT - guard.EPSILON,
                },
                "delta_vs_A20r": {
                    "P_dev": value["P_dev"] - reference["P_dev"],
                    "T_dev": value["T_dev"] - reference["T_dev"],
                    "H_pilot": value["H_pilot"] - reference["H_pilot"],
                    "by_type_accuracy": {
                        kind: value["by_type"][kind]["accuracy"]
                        - reference["by_type"][kind]["accuracy"]
                        for kind in guard.TYPES
                    },
                },
            },
            "passes": all(g["pass"] for g in gates.values()),
        }
    passing = [n for n, c in out.items() if c["passes"]]
    return {
        "candidates": out,
        "passing": passing,
        "finalists": passing[:MAX_FINALISTS],
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
    panels = guard.contrast.Panels(
        panel_registry.path(args.panel_root, "typed-dev", "gold"),
        panel_registry.path(args.panel_root, "css-pilot", "gold"),
    )
    ref_name, _, ref_dir = args.reference.partition("=")
    reference = readout(ref_name, Path(ref_dir), panels)
    candidates = {}
    for spec in args.candidate:
        name, _, directory = spec.partition("=")
        candidates[name] = readout(name, Path(directory), panels)
    result = {
        "schema": SCHEMA,
        "label": "Stage B development gates (T = 1, 32K, node B); never a release or formal score",
        "rules": {
            "collapse": "screen rule 1 (screen_rules.collapse) against the reference",
            "htdev2": "not FLAG (delta <= -0.02) against the reference's HT-DEV v2 predictions",
            "proxy": f"P_dev not {PROXY_DROP} or more below the reference's",
        },
        "reference": {"name": ref_name, **reference},
        **decide(reference, candidates, args.panel_root),
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "finalists": result["finalists"],
                "gates": {
                    n: {g: v["pass"] for g, v in c["gates"].items()}
                    for n, c in result["candidates"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
