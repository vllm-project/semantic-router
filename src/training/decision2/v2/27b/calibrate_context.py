"""Pinned CAL temperature fit recorded at an explicit inference context.

``training.model.calibrate`` records the run's training limit as the inference
context. A package served at a longer limit needs a calibration whose recorded
context equals that limit (the package loaders and the release builder check
it). This wrapper passes the longer limit to the unchanged fitting code; CAL
rows that fit the training limit give identical logits, which the sidecar
receipt shows by comparing ``logits_sha256`` with the reference calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.model import calibrate as pinned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument(
        "--reference", type=Path, help="Calibration at the training limit"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = pinned.selected_run

    def selected_run(run_dir: Path, cal_path: Path) -> dict:
        selected = original(run_dir, cal_path)
        if args.context < selected["contract"]["max_length"]:
            raise SystemExit(
                "The inference context may not be shorter than the training limit"
            )
        return {
            **selected,
            "contract": {**selected["contract"], "max_length": args.context},
        }

    pinned.selected_run = selected_run
    report = pinned.calibrate(
        args.run_dir,
        args.cal,
        args.output,
        source_path=args.source_path,
        batch_size=1,
        device_name="cuda:0",
    )
    receipt = {
        "schema": "decision2-27b-calibration-context/1",
        "context": args.context,
        "calibration_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "logits_sha256": report["logits_sha256"],
        "temperature_by_type": report["temperature_by_type"],
    }
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        receipt["reference"] = {
            "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "max_length": reference["inference"]["max_length"],
            "logits_identical": reference["logits_sha256"] == report["logits_sha256"],
            "temperatures_identical": reference["temperature_by_type"]
            == report["temperature_by_type"],
            "model_sha256_identical": reference["model_sha256"]
            == report["model_sha256"],
        }
    sidecar = args.output.with_name(args.output.stem + ".context.json")
    sidecar.write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
