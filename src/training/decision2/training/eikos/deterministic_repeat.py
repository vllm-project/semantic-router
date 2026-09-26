"""Compare two frozen, gold-free deterministic Eikos package predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.eikos.io import atomic_json
from training.eikos.repeatability import compare, load, sha

EXPECTED_ITEMS = 1430


def _manifest(predictions: Path, prompts: Path, package: Path) -> dict[str, Any]:
    manifest_path = Path(str(predictions) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("predictions_sha256") != sha(predictions):
        raise ValueError("Prediction bytes differ from native manifest")
    if manifest.get("input_sha256") != sha(prompts):
        raise ValueError("Prompt bytes differ from native manifest")
    if manifest.get("model_sha256") != sha(package / "SHA256SUMS"):
        raise ValueError("Model bytes differ from native manifest")
    if manifest.get("calibration_sha256") != sha(package / "calib.json"):
        raise ValueError("Calibration bytes differ from native manifest")
    if (
        manifest.get("input_items") != EXPECTED_ITEMS
        or manifest.get("evaluated_items") != EXPECTED_ITEMS
        or manifest.get("max_items") is not None
        or manifest.get("counts", {}).get("items") != EXPECTED_ITEMS
        or manifest["counts"].get("valid_questions") != EXPECTED_ITEMS
        or manifest["counts"].get("invalid_questions") != 0
    ):
        raise ValueError("Native receipt is not the complete valid pilot")
    runtime = manifest.get("runtime", {})
    if (
        runtime.get("torch_deterministic_algorithms") is not True
        or runtime.get("flash_linear_attention") != "0.5.2"
    ):
        raise ValueError("Native receipt lacks frozen deterministic FLA runtime")
    return manifest


def audit(
    *,
    predictions_a: Path,
    predictions_b: Path,
    prompts: Path,
    package: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    left, right = (
        _manifest(predictions_a, prompts, package),
        _manifest(predictions_b, prompts, package),
    )
    for key in (
        "adapter_version",
        "model_id",
        "model_revision",
        "model_sha256",
        "calibration_sha256",
        "input_sha256",
        "collector_source_sha256",
        "runtime",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"Independent-process receipts differ in {key}")
    comparison = compare(load(predictions_a), load(predictions_b))
    gate = (
        comparison["categorical_mismatch_n"] == 0
        and comparison["max_option_probability_drift"] <= 1e-6
    )
    report = {
        "schema_version": "decision2-eikos-deterministic-repeat/1",
        "scope": "gold-free runtime repeatability only; no score or model selection",
        "input_items": EXPECTED_ITEMS,
        "prompts_sha256": sha(prompts),
        "model_sha256": left["model_sha256"],
        "calibration_sha256": left["calibration_sha256"],
        "collector_source_sha256": left["collector_source_sha256"],
        "runtime": left["runtime"],
        "predictions_sha256": {
            "first": sha(predictions_a),
            "second": sha(predictions_b),
        },
        "manifests_sha256": {
            "first": sha(Path(str(predictions_a) + ".manifest.json")),
            "second": sha(Path(str(predictions_b) + ".manifest.json")),
        },
        "comparison": comparison,
        "predeclared_numeric_repeat_gate_pass": gate,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        predictions_a=args.predictions_a,
        predictions_b=args.predictions_b,
        prompts=args.prompts,
        package=args.package,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "categorical_mismatch_n": report["comparison"][
                    "categorical_mismatch_n"
                ],
                "max_option_probability_drift": report["comparison"][
                    "max_option_probability_drift"
                ],
                "gate_pass": report["predeclared_numeric_repeat_gate_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
