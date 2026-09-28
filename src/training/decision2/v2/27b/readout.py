"""Container-side CAL fit and one development readout for a completed LoRA arm.

Runs the pipeline's own ``training.model.calibrate`` on the frozen BEST
checkpoint, then ``training.model.infer`` once per gold-free panel with that
calibration. Scoring happens separately; this module never reads labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(argv: list[str]) -> None:
    subprocess.run([sys.executable, *argv], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--cal", required=True)
    parser.add_argument("--panel", action="append", required=True, help="NAME=PROMPTS")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / "COMPLETE.json").is_file():
        raise SystemExit("Run is not complete")
    best = json.loads((args.run_dir / "BEST.json").read_text(encoding="utf-8"))[
        "checkpoint"
    ]
    calibration = args.out_dir / "calibration.json"
    run(
        [
            "-m",
            "training.model.calibrate",
            "--run-dir",
            str(args.run_dir),
            "--cal",
            args.cal,
            "--source-path",
            args.source_path,
            "--output",
            str(calibration),
            "--batch-size",
            "1",
            "--device",
            "cuda:0",
        ]
    )
    for spec in args.panel:
        name, prompts = spec.split("=", 1)
        run(
            [
                "-m",
                "training.model.infer",
                "--checkpoint",
                str(args.run_dir / best),
                "--source-path",
                args.source_path,
                "--calibration",
                str(calibration),
                "--input",
                prompts,
                "--output",
                str(args.out_dir / f"{name}.predictions.jsonl"),
                "--model-id",
                args.model_id,
                "--model-revision",
                best,
                "--max-length",
                "4096",
            ]
        )
    print(
        json.dumps({"best": best, "panels": [s.split("=", 1)[0] for s in args.panel]})
    )


if __name__ == "__main__":
    main()
