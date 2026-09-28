"""Post-seal pipeline check: Lux 1.0's own head on frozen Lux readout features.

Applies the unchanged published Lux head and temperature to the readout-time
joint features and scores them with the unchanged development scorers. If the
feature path is faithful, this reproduces Lux 1.0's historical same-input
development readout; it changes no sealed head or selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import pins
from .lux_teacher import load_lux_head, teacher_distributions
from .readout import predictions_for, proxy, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lux-root", type=Path, required=True)
    parser.add_argument(
        "--features", type=Path, required=True, help="lux1-9b readout extraction"
    )
    parser.add_argument("--prompts", action="append", required=True)
    parser.add_argument("--dev-gold", type=Path, required=True)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from training.model.infer import load_prompts

    pins.verify_source("lux1-9b", args.lux_root)
    head, temperatures = load_lux_head(args.lux_root)
    temps = {
        "choice": temperatures["choice"],
        "noul": temperatures["noul"],
        "score_relative": temperatures["score"],
        "score_absolute": temperatures["score"],
    }
    items = {}
    for spec in args.prompts:
        name, path = spec.split("=", 1)
        pins.verify_data(name, path)
        items[name] = load_prompts(Path(path))
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "model_sha256": pins.SOURCES["lux1-9b"]["files"]["decision_head.safetensors"],
        "adapter_sha256": "lux-published-head-on-frozen-features",
        "calibration_sha256": pins.SOURCES["lux1-9b"]["files"]["temperature.json"],
        "tokens": lambda row: row["j_tokens"],
    }
    reports = {}
    for panel in ("dev", "css_pilot"):
        rows, output = teacher_distributions(args.features / panel, head, temperatures)
        records = [
            {
                "id": o["id"],
                "task_type": r["task_type"],
                "keys": r["keys"],
                "valid": o["valid"],
                "relative": o.get("logits"),
            }
            for r, o in zip(rows, output)
        ]
        predictions = predictions_for(
            records, rows, items[panel], temps, "relative", identity
        )
        write_jsonl(args.output / f"{panel}.predictions.jsonl", predictions)
    for filename, gold, module, report in (
        ("dev.predictions.jsonl", args.dev_gold, "benchmark.score", "dev.score.json"),
        (
            "css_pilot.predictions.jsonl",
            args.css_gold,
            "transfer.score",
            "css_pilot.score.json",
        ),
    ):
        command = [
            sys.executable,
            "-m",
            module,
            "--gold",
            str(gold),
            "--predictions",
            str(args.output / filename),
            "--output",
            str(args.output / report),
        ]
        if module == "benchmark.score":
            command += [
                "--model-id",
                "lux1-published-head-frozen-features",
                "--model-revision",
                "bd45a30a",
                "--backend",
                "decision2-9b-frozen-features",
            ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        reports[report] = json.loads((args.output / report).read_text(encoding="utf-8"))
    summary = {
        **proxy(reports["dev.score.json"], reports["css_pilot.score.json"]),
        "typed_by_type": {
            k: reports["dev.score.json"]["by_type"][k]["correct_n"]
            for k in ("choice", "noul", "score")
        },
        "css_tasks": {
            k: v["macro_f1_all"]
            for k, v in reports["css_pilot.score.json"]["tasks"].items()
        },
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
