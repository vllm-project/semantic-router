"""Reconstruct T=1 probabilities from a calibrated native prediction file.

This is a development diagnostic, not a model inference adapter or releasable
prediction receipt. Positive temperature preserves every categorical argmax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def undo_map(values: dict[str, float], temperature: float) -> dict[str, float]:
    raw = {key: max(0.0, float(value)) ** temperature for key, value in values.items()}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("cannot invert an all-zero probability map")
    return {key: value / total for key, value in raw.items()}


def transform(predictions: Path, calibration: Path, output: Path) -> dict:
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError(output)
    fit = json.loads(calibration.read_text(encoding="utf-8"))
    fitted = fit["temperature_by_type"]
    parent_sha = sha256(predictions)
    calibration_sha = sha256(calibration)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    types = dict.fromkeys(("choice", "noul", "score"), 0)
    with predictions.open(encoding="utf-8") as source, output.open(
        "x", encoding="utf-8"
    ) as sink:
        for line in source:
            row = json.loads(line)
            if row.get("model_sha256") != fit["model_sha256"]:
                raise ValueError("model hash differs from the frozen CAL fit")
            if row.get("calibration_sha256") != calibration_sha:
                raise ValueError("prediction is not bound to the supplied CAL fit")
            for answer in row["answers"].values():
                kind = answer["type"]
                temperature = float(fitted[kind])
                if kind == "noul":
                    p = float(answer["noul"])
                    answer["noul"] = undo_map(
                        {"false": 1.0 - p, "true": p}, temperature
                    )["true"]
                else:
                    answer["probabilities"] = undo_map(
                        answer["probabilities"], temperature
                    )
                    if kind == "score":
                        answer["score"] = sum(
                            int(level) * probability
                            for level, probability in answer["probabilities"].items()
                        )
                types[kind] += 1
            row["diagnostic_original_calibration_sha256"] = row.pop(
                "calibration_sha256"
            )
            row["diagnostic_transform"] = "invert_native_temperature_to_t1_v1"
            sink.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    receipt = {
        "diagnostic_only": True,
        "method": "p_uncal proportional to p_cal**fitted_temperature",
        "parent_predictions_sha256": parent_sha,
        "calibration_sha256": calibration_sha,
        "output_sha256": sha256(output),
        "items": count,
        "questions_by_type": types,
    }
    manifest = output.with_name(output.name + ".manifest.json")
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            transform(args.predictions, args.calibration, args.output), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
