"""M6-SUMMARY.json for one M6 formal run from its sealed receipts (stdlib only; no gold read).

    python3 -m v2.06b.m6_summary --run RUN --mlx-run RUN-mlx --gates DIR --released NAME \
        --comparator NAME=DIR ... [--control] [--answers-diff FILE]

Reads REPORT.json, PAIRED-vs-<NAME>.json, GPU-TIME.json and M6-CACHE.json of RUN and RUN-mlx,
`mlx-diag.score.json`, and the gate outputs in DIR (`paired-vs-<released>.json`, `types.json`).
Successor rule (M6 prereg section 4; skipped with --control): the paired post-key v3 95% CI
lower bound vs the released model is > 0, the upper bound of the paired H interval is >= 0,
and no decision type collapsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(run: Path, name: str) -> dict[str, Any]:
    path = run / f"PAIRED-vs-{name}.json"
    d = load(path)
    return {
        "delta_v3": d["point"]["delta"]["score"],
        "ci95_v3": [d["ci95"]["low"], d["ci95"]["high"]],
        "delta_H": d["point"]["delta"]["H"],
        "ci95_H": [
            d["axis_ci95"]["H"]["delta"]["low"],
            d["axis_ci95"]["H"]["delta"]["high"],
        ],
        "delta_T": d["point"]["delta"]["T"],
        "ci95_T": [
            d["axis_ci95"]["T"]["delta"]["low"],
            d["axis_ci95"]["T"]["delta"]["high"],
        ],
        "right_v3": d["point"]["right"]["score"],
        "replicates": d.get("replicates"),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    report = load(args.run / "REPORT.json")
    typed = report["panels"]["typed-final"]["by_type"]
    public = report["panels"]["public231"]
    mlx = load(args.mlx_run / "mlx-diag.score.json")
    hours = {r.name: load(r / "GPU-TIME.json") for r in (args.run, args.mlx_run)}
    types = load(args.gates / "types.json")["types"]
    human = load(args.gates / f"paired-vs-{args.released}.json")
    comparators = {}
    for entry in args.comparator:
        name, _, directory = entry.partition("=")
        comparators[name] = {"run_dir": directory, **paired(args.run, name)}
    collapsed = {k: v["verdict"] for k, v in types.items() if v["verdict"] != "OK"}
    out: dict[str, Any] = {
        "schema": "dev2-06b-m6-summary/1",
        "label": "post-key same-panel",
        "name": args.name,
        "run_dir": str(args.run),
        "mlx_run_dir": str(args.mlx_run),
        "model_path": (
            report["model"].get("path")
            if isinstance(report.get("model"), dict)
            else None
        ),
        "seal_sha256": report.get("seal_sha256"),
        "v3": report["v3"]["score"],
        "T": report["v3"]["T"],
        "H": report["v3"]["H"],
        "public231": {
            "correct": public["correct"],
            "items": public["items"],
            **{t: public["tiers"][t]["correct"] for t in ("easy", "standard", "hard")},
        },
        "typed_final": {
            k: {"correct": v["correct"], "n": v["n"]} for k, v in typed.items()
        },
        "mlx_diag": {
            "overall": mlx["type_macro_accuracy"],
            "english": mlx["english_type_macro_accuracy"],
            "non_english": mlx["non_english_type_macro_accuracy"],
            "items": mlx["items"],
            "invalid_or_missing": mlx["invalid_or_missing"],
            "scope": mlx.get("scope"),
        },
        "paired": comparators,
        "gate_human_transfer_vs_released": {
            "released": args.released,
            "delta_H": human["point"]["delta"]["H"],
            "ci95_H": [
                human["axis_ci95"]["H"]["delta"]["low"],
                human["axis_ci95"]["H"]["delta"]["high"],
            ],
            "ci95_v3": [human["ci95"]["low"], human["ci95"]["high"]],
        },
        "type_collapse": {
            "verdicts": {k: v["verdict"] for k, v in types.items()},
            "accuracy": {k: v["accuracy"] for k, v in types.items()},
            "any_collapsed": bool(collapsed),
        },
        "loaded_parameters": report["parameters"]["loaded"],
        "gpu_time": {
            "runs": {k: v["gpu_hours"] for k, v in hours.items()},
            "gpu_hours": sum(v["gpu_hours"] for v in hours.values()),
            "image_ids": sorted({v["image_id"] for v in hours.values()}),
        },
        "triton_cache": {
            r.name: load(r / "M6-CACHE.json")
            for r in (args.run, args.mlx_run)
            if (r / "M6-CACHE.json").is_file()
        },
    }
    for record in out["triton_cache"].values():
        record.pop("recorded_utc", None)
    if args.answers_diff:
        diff = load(args.answers_diff)
        out["answers_vs_reference"] = {
            "reference": diff["right"],
            "answers_equal": diff["answers_equal"],
            **diff["total"],
        }
    if args.control:
        out["successor"] = {"skipped": "control run"}
    else:
        released = comparators[args.released]
        checks = {
            "v3_lower_bound_gt_0": released["ci95_v3"][0] > 0,
            "H_upper_bound_ge_0": out["gate_human_transfer_vs_released"]["ci95_H"][1]
            >= 0,
            "no_type_collapsed": not collapsed,
        }
        out["successor"] = {
            "released": args.released,
            "released_run_dir": released["run_dir"],
            "checks": checks,
            "verdict": all(checks.values()),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mlx-run", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--released", default="released")
    parser.add_argument("--comparator", action="append", default=[])
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--answers-diff", type=Path)
    args = parser.parse_args(argv)
    out = summarize(args)
    with (args.run / "M6-SUMMARY.json").open("x", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, sort_keys=True)
        stream.write("\n")
    brief = {
        k: out[k] for k in ("v3", "T", "H", "public231", "typed_final", "mlx_diag")
    }
    brief["successor"] = out["successor"].get("verdict", "skipped")
    brief["vs_released"] = out["paired"].get(args.released, {}).get("ci95_v3")
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
