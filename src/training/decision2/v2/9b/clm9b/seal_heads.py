"""Freeze every selected head before any development panel feature exists.

Collects, per source and arm, the primary-seed run at the chosen layer and the
two extra seeds, with head, calibration and BEST digests plus SELECT and CAL
summaries. The resulting seal is committed before readout extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        action="append",
        required=True,
        help="grid output directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    seal = {
        "version": "decision2-9b-heads-seal-v1",
        "code_commit": args.code_commit,
        "sources": {},
    }
    for grid in args.grid:
        summary = json.loads((grid / "GRID.json").read_text(encoding="utf-8"))
        runs = sorted(path for path in grid.iterdir() if (path / "BEST.json").is_file())
        source = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))[
            "source"
        ]
        entry = {
            "features_manifest_sha256": summary["features_manifest_sha256"],
            "arms": {},
        }
        for arm, result in summary["arms"].items():
            record = {
                "chosen_layer": result.get("chosen_layer"),
                "stopped": result.get("stopped"),
                "layer_screen": {},
                "runs": [],
            }
            for layer, best in result.get("layers", {}).items():
                record["layer_screen"][str(layer)] = {
                    "select_correct": best["select"]["correct"],
                    "select_macro": best["select"]["family_macro_accuracy"],
                    "select_brier": best["select"]["family_macro_brier"],
                    "selected_step": best["selected_step"],
                }
            for seed, best in (result.get("seeds") or {}).items():
                if best is None:
                    continue
                name = f"{arm}-L{best['layer']}-s{best['seed']}"
                folder = grid / name
                calibration = json.loads(
                    (folder / "calibration.json").read_text(encoding="utf-8")
                )
                record["runs"].append(
                    {
                        "run": name,
                        "seed": int(seed),
                        "layer": best["layer"],
                        "selected_step": best["selected_step"],
                        "head_sha256": pins.file_sha256(folder / "head.safetensors"),
                        "calibration_sha256": pins.file_sha256(
                            folder / "calibration.json"
                        ),
                        "best_sha256": pins.file_sha256(folder / "BEST.json"),
                        "config_sha256": pins.file_sha256(folder / "config.json"),
                        "select": {
                            "correct": best["select"]["correct"],
                            "macro": best["select"]["family_macro_accuracy"],
                            "brier": best["select"]["family_macro_brier"],
                            "by_type": {
                                k: v["correct"]
                                for k, v in best["select"]["by_type"].items()
                            },
                            "score_relative_correct": best["select_score_relative"][
                                "by_type"
                            ]
                            .get("score", {})
                            .get("correct"),
                        },
                        "cal_after": calibration["after"],
                        "temperatures": calibration["temperature_by_readout"],
                        "head_parameters": best["head_parameters"],
                        "reload_max_abs_drift": best["reload_max_abs_drift"],
                    }
                )
            entry["arms"][arm] = record
        seal["sources"][source] = entry
    args.output.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"seal": str(args.output), "sha256": pins.file_sha256(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
