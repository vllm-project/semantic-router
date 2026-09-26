"""Apply a frozen CAL temperature to positive uncalibrated pilot probabilities.

This is a gold-blind development shortcut. It cannot reconstruct logits when a
softmax probability underflowed to zero, so it rejects such inputs. Final
release evaluation uses training.model.infer with --calibration instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from training.model.calibration import validate_temperatures
from training.model.data import canonical, file_sha256
from training.model.infer import ADAPTER_VERSION, normalized_answer, write_output

VERSION = "decision2-development-posthoc-temperature/1"


def _positive_probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 < value < 1


def reweight_answer(
    answer: dict[str, Any], temperatures: dict[str, float]
) -> dict[str, Any]:
    kind = answer.get("type")
    if kind not in temperatures:
        raise ValueError("Unknown answer type in uncalibrated predictions")
    if "error" in answer:
        return dict(answer)
    if kind == "noul":
        if set(answer) != {"type", "noul"} or not _positive_probability(answer["noul"]):
            raise ValueError(
                "Noul pilot probability is missing or on an underflow boundary"
            )
        p_true = answer["noul"]
        return normalized_answer(
            kind,
            ["false", "true"],
            [math.log1p(-p_true), math.log(p_true)],
            temperatures[kind],
        )
    key = "choice" if kind == "choice" else "score"
    if set(answer) != {"type", key, "probabilities"}:
        raise ValueError("Choice/Score pilot answer has unexpected fields")
    probabilities = answer["probabilities"]
    if (
        not isinstance(probabilities, dict)
        or len(probabilities) < 2
        or any(
            not isinstance(option, str) or not _positive_probability(value)
            for option, value in probabilities.items()
        )
        or abs(sum(probabilities.values()) - 1) > 1e-6
    ):
        raise ValueError(
            "Pilot probability map is missing, invalid, or has an underflowed zero"
        )
    if kind == "score" and set(probabilities) != {
        str(index) for index in range(len(probabilities))
    }:
        raise ValueError("Score candidate levels are incomplete")
    return normalized_answer(
        kind,
        list(probabilities),
        [math.log(value) for value in probabilities.values()],
        temperatures[kind],
    )


def build(
    predictions_path: Path,
    calibration_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = predictions_path.with_name(predictions_path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_sha = file_sha256(predictions_path)
    if (
        manifest.get("adapter_version") != ADAPTER_VERSION
        or manifest.get("predictions_sha256") != source_sha
    ):
        raise ValueError(
            "Source predictions are not an intact native uncalibrated adapter run"
        )
    if manifest.get("model_sha256") != calibration.get("model_sha256") or manifest.get(
        "model_revision"
    ) != calibration.get("selected_checkpoint"):
        raise ValueError("CAL is not bound to the uncalibrated source checkpoint")
    temperatures = validate_temperatures(calibration.get("temperature_by_type"))
    calibration_sha = file_sha256(calibration_path)
    script_sha = file_sha256(Path(__file__))
    adapter_sha = hashlib.sha256(
        canonical(
            {
                "source_adapter_sha256": manifest["adapter_sha256"],
                "posthoc_script_sha256": script_sha,
                "version": VERSION,
            }
        ).encode("utf-8")
    ).hexdigest()
    transformed: list[dict[str, Any]] = []
    seen: set[str] = set()
    question_count = 0
    with predictions_path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or row["id"] in seen
            ):
                raise ValueError(
                    f"Duplicate or missing prediction ID on line {line_no}"
                )
            seen.add(row["id"])
            answers = row.get("answers")
            if not isinstance(answers, dict) or not answers:
                raise ValueError(f"Missing answers on line {line_no}")
            if (
                row.get("model_sha256") != manifest["model_sha256"]
                or row.get("adapter_sha256") != manifest["adapter_sha256"]
            ):
                raise ValueError(
                    f"Prediction identity differs from source manifest on line {line_no}"
                )
            updated = dict(row)
            updated["answers"] = {
                question_id: reweight_answer(answer, temperatures)
                for question_id, answer in answers.items()
            }
            updated["adapter_sha256"] = adapter_sha
            updated["calibration_sha256"] = calibration_sha
            transformed.append(updated)
            question_count += len(answers)
    counts = manifest.get("counts", {})
    if len(transformed) != counts.get("items") or question_count != counts.get(
        "questions"
    ):
        raise ValueError("Source prediction counts differ from its manifest")
    output_manifest = {
        **manifest,
        "adapter_version": VERSION,
        "adapter_sha256": adapter_sha,
        "source_adapter_sha256": manifest["adapter_sha256"],
        "execution": "Source native inference latencies retained; positive probabilities posthoc temperature transformed without model forward pass; development only",
        "posthoc": {
            "source_prediction_filename": predictions_path.name,
            "source_predictions_sha256": source_sha,
            "source_manifest_sha256": file_sha256(manifest_path),
            "script_sha256": script_sha,
            "zero_probability_policy": "reject; native calibrated inference required",
            "scope": "development only; final release requires native calibrated adapter",
        },
        "calibration": {
            "file_sha256": calibration_sha,
            "cal_sha256": calibration["cal_sha256"],
            "temperature_by_type": temperatures,
            "binding": "posthoc_from_positive_uncalibrated_probabilities",
        },
    }
    output_manifest.pop("predictions_sha256", None)
    return transformed, output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output == args.predictions:
        parser.error("Output must differ from source predictions")
    predictions, manifest = build(args.predictions, args.calibration)
    write_output(args.output, predictions, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "predictions_sha256": manifest["predictions_sha256"],
                "model_sha256": manifest["model_sha256"],
                "items": len(predictions),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
