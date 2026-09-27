"""Gold-free native output parity for the reduced 4B Hugging Face export.

The public model removes private, non-runtime provenance and regenerates its
SHA256SUMS. This script verifies all inference file bytes against the scored
private package, then runs the public native Decision readout on the original
gold-free prompts and compares every answer with sealed scored predictions.
The resulting receipt stays private; a full pass is required before publishing.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.eikos import shared_answer
from inference.run import digest, load_prompts
from publication import export_hf_v3 as export
from training.eikos.native import load_decider
from training.eikos.published_infer import use_torch_reference_gated_delta
from training.eikos.verify_export import compare_answers, verify_sums

VERSION = "decision2-hf-public-runtime-parity/2"
MAX_PROBABILITY_DRIFT = 1e-6


def _scored_manifest(
    path: Path,
    *,
    prompts_path: Path,
    predictions_path: Path,
    item_count: int,
    private: dict[str, Any],
    calibration_sha: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "model_id": export.MODEL_ID,
        "model_revision": private["model_revision"],
        "model_sha256": private["native_model_sha256"],
        "package_manifest_sha256": private["native_model_sha256"],
        "calibration_sha256": calibration_sha,
        "input_sha256": export._sha(prompts_path),
        "predictions_sha256": export._sha(predictions_path),
        "input_items": item_count,
        "evaluated_items": item_count,
        "max_items": None,
        "adapter_version": "decision2-eikos-semif-native-v1",
    }
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in exact.items()):
        raise ValueError("Scored native manifest does not bind the complete panel")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("torch_deterministic_algorithms") is not True
    ):
        raise ValueError("Scored native manifest lacks deterministic runtime details")
    return value


def _references(
    path: Path,
    prompts: list[dict[str, Any]],
    scored_sha: str,
    calibration_sha: str,
    revision: str,
) -> dict[str, dict[str, Any]]:
    wanted = {row["id"]: row for row in prompts}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            value = json.loads(line)
            key = value.get("id")
            if key not in wanted or key in result:
                raise ValueError(f"Reference row {number} has unknown or duplicate ID")
            prompt = wanted[key]
            if (
                value.get("model_id") != export.MODEL_ID
                or value.get("model_revision") != revision
                or value.get("adapter_version") != "decision2-eikos-semif-native-v1"
                or value.get("model_sha256") != scored_sha
                or value.get("calibration_sha256") != calibration_sha
                or value.get("source_input_sha256")
                != digest({"state": prompt["state"], "questions": prompt["questions"]})
                or not isinstance(value.get("answers"), dict)
                or set(value["answers"]) != set(prompt["questions"])
            ):
                raise ValueError(f"Reference row {number} is not frozen scored output")
            result[key] = value
    if set(result) != set(wanted):
        raise ValueError("Reference predictions omit original gold-free prompts")
    return result


def _compare(
    decider: Any,
    prompts: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    answers = 0
    invalid = 0
    mismatches = 0
    max_drift = 0.0
    for row in prompts:
        questions = row["questions"]
        try:
            raw = decider.decide_all(state=row["state"], questions=questions)
            actual = {
                key: shared_answer(questions[key], value)
                for key, (value, _) in raw.items()
            }
        except ValueError as error:
            if "tokens > 16000" not in str(error):
                raise
            actual = {
                key: {"type": question["type"], "error": "context_overflow"}
                for key, question in questions.items()
            }
        expected = references[row["id"]]["answers"]
        if set(actual) != set(questions) or set(expected) != set(questions):
            raise ValueError("Native runtime omitted a question")
        for key in questions:
            left, right = expected[key], actual[key]
            answers += 1
            if "error" in left or "error" in right:
                invalid += 1
                if left != right:
                    mismatches += 1
                continue
            same, drift, _ = compare_answers(left, right)
            if not same:
                mismatches += 1
            max_drift = max(max_drift, drift)
    if mismatches or max_drift > MAX_PROBABILITY_DRIFT:
        raise ValueError(
            f"Public native parity failed: {mismatches} categorical changes, "
            f"maximum probability drift {max_drift}"
        )
    return {
        "items": len(prompts),
        "answers": answers,
        "invalid_answers": invalid,
        "categorical_mismatches": mismatches,
        "maximum_probability_drift": max_drift,
        "permitted_probability_drift": MAX_PROBABILITY_DRIFT,
    }


def attest(
    *,
    private_package: Path,
    public_export: Path,
    prompts_path: Path,
    predictions_path: Path,
    scored_native_manifest_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not device.startswith("cuda:"):
        raise ValueError("Use the qualified native GPU inference path")
    private, _ = export._checked_source(private_package)
    public = export.verify(public_export)
    if (
        public["source_private_package_manifest_sha256"]
        != export._sha(private_package / "PACKAGE_MANIFEST.json")
        or public["scored_native_model_sha256"] != private["native_model_sha256"]
    ):
        raise ValueError("Public export is not bound to the scored private package")
    native = public_export / "model"
    if verify_sums(native) != len(public["inference_files_sha256"]):
        raise ValueError("Public native model file count changed")
    prompts = load_prompts(prompts_path)
    calibration_sha = export._sha(private_package / "native" / "calib.json")
    scored_manifest = _scored_manifest(
        scored_native_manifest_path,
        prompts_path=prompts_path,
        predictions_path=predictions_path,
        item_count=len(prompts),
        private=private,
        calibration_sha=calibration_sha,
    )
    references = _references(
        predictions_path,
        prompts,
        private["native_model_sha256"],
        calibration_sha,
        private["model_revision"],
    )
    import fla
    import torch
    import transformers

    torch.use_deterministic_algorithms(True)
    backend = use_torch_reference_gated_delta()
    runtime = {
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "flash_linear_attention": fla.__version__,
        "device": device,
        "device_architecture": str(
            getattr(torch.cuda.get_device_properties(device), "gcnArchName", "unknown")
        ).split(":", 1)[0],
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        **backend,
    }
    if any(
        scored_manifest["runtime"].get(key) != value for key, value in runtime.items()
    ):
        raise ValueError("Public runtime differs from the scored native runtime")
    decider = load_decider(native, None, native / "calib.json", device=device)
    count = _compare(decider, prompts, references)
    receipt = {
        "schema_version": VERSION,
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": export.MODEL_ID,
        "model_revision": private["model_revision"],
        "scored_native_model_sha256": private["native_model_sha256"],
        "public_runtime_manifest_sha256": public["public_runtime_manifest_sha256"],
        "public_export_manifest_sha256": export._sha(
            public_export / "evaluation" / "manifest.json"
        ),
        "prompts_sha256": export._sha(prompts_path),
        "scored_predictions_sha256": export._sha(predictions_path),
        "scored_native_manifest_sha256": export._sha(scored_native_manifest_path),
        "runtime": runtime,
        "comparison": count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as staged:
        json.dump(receipt, staged, indent=2, sort_keys=True, allow_nan=False)
        staged.write("\n")
        temp = Path(staged.name)
    try:
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-package", type=Path, required=True)
    parser.add_argument("--public-export", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--scored-predictions", type=Path, required=True)
    parser.add_argument("--scored-native-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = attest(
        private_package=args.private_package,
        public_export=args.public_export,
        prompts_path=args.prompts,
        predictions_path=args.scored_predictions,
        scored_native_manifest_path=args.scored_native_manifest,
        output=args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {"model_id": result["model_id"], "items": result["comparison"]["items"]}
        )
    )


if __name__ == "__main__":
    main()
