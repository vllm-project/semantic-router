"""Collect gold-free predictions from an attested standalone Eikos 4B package.

This is the only inference entrypoint for the frozen Decision 2.0 Eikos
candidate. It loads the package's own native SemIf server and calibration,
binds every answer to the exact package digest, and writes an atomic manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from inference.eikos import shared_answer
from inference.run import digest, load_prompts, synchronize

from training.eikos.native import load_decider
from training.eikos.rights import (
    SCHEMA,
    SCOPE,
    verify_noncommercial_package_attestation,
)
from training.eikos.verify_export import verify_sums
from training.model.data import file_sha256
from training.model.infer import write_output

ADAPTER_VERSION = "decision2-eikos-semif-native-v1"
MODEL_ID = "llm-semantic-router/dev-2.0-4b"


def package_identity(
    path: Path, *, rights_attestation: Path | None = None, require_rights: bool = False
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    file_count = verify_sums(path)
    receipt = json.loads(
        (path / "decision2_provenance.json").read_text(encoding="utf-8")
    )
    decision_config = json.loads(
        (path / "decision_config.json").read_text(encoding="utf-8")
    )
    expected_config = {
        "prompt_version": "letter-v1-semif",
        "readout": "letter-logit",
        "calib": "calib.json",
        "max_one_pass": 100,
    }
    if (
        receipt.get("model_name") != "dev-2.0-4b"
        or receipt.get("source_revision") != "582ffb13f19a4da3f455e3db198584190bd7755b"
        or decision_config != expected_config
        or receipt.get("calibration_sha256") != file_sha256(path / "calib.json")
        or not (path / "LICENSE").is_file()
        or not (path / "LICENSE-Qwen").is_file()
        or not (path / "NOTICE").is_file()
    ):
        raise ValueError(
            "Standalone Eikos package does not match frozen native contract"
        )
    rights_mode = receipt.get("rights_mode", "noncommercial_research")
    if rights_mode == "clean" and (
        receipt.get("publication_eligible") is not True
        or receipt.get("publication_scope") != SCOPE
        or receipt.get("rights_schema_version") != SCHEMA
        or receipt.get("rights_manifest_sha256")
        != receipt.get("training_data_manifest_sha256")
    ):
        raise ValueError("Clean Eikos package lacks matching publication provenance")
    if rights_mode not in {"clean", "noncommercial_research"}:
        raise ValueError("Unexpected Eikos package rights mode")
    if rights_mode == "noncommercial_research":
        if rights_attestation is None and require_rights:
            raise ValueError(
                "Noncommercial Eikos package requires an exact rights attestation"
            )
        if rights_attestation is not None:
            verify_noncommercial_package_attestation(rights_attestation, receipt)
            embedded_hash = receipt.get("rights_attestation_sha256")
            if embedded_hash is not None and embedded_hash != file_sha256(
                rights_attestation
            ):
                raise ValueError("Noncommercial rights statement changed after export")
    return {
        "model_sha256": file_sha256(path / "SHA256SUMS"),
        "calibration_sha256": receipt["calibration_sha256"],
        "selected_checkpoint": receipt["selected_checkpoint"],
        "source_revision": receipt["source_revision"],
        "source_release": receipt["source_release"],
        "package_files_checked": file_count,
        "rights_mode": rights_mode,
        "rights_attestation_sha256": (
            file_sha256(rights_attestation) if rights_attestation is not None else None
        ),
    }


def collect(
    *,
    model_path: Path,
    prompts: Path,
    output: Path,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    device: str = "cuda:0",
    max_items: int | None = None,
    rights_attestation: Path | None = None,
) -> dict[str, Any]:
    if model_id != MODEL_ID:
        raise ValueError(f"Eikos 4B Decision 2.0 model ID must be {MODEL_ID}")
    if not device.startswith("cuda:"):
        raise ValueError("Published Eikos inference requires the qualified GPU path")
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError(output)
    identity = package_identity(
        model_path, rights_attestation=rights_attestation, require_rights=True
    )
    revision = model_revision or identity["selected_checkpoint"]
    if revision != identity["selected_checkpoint"]:
        raise ValueError(
            "Requested model revision differs from frozen selected checkpoint"
        )
    rows = load_prompts(prompts)
    all_items = len(rows)
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        rows = rows[:max_items]
    native = load_decider(model_path, None, model_path / "calib.json", device=device)
    import fla
    import torch

    runtime = {
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "transformers": version("transformers"),
        "flash_linear_attention": fla.__version__,
        "device_architecture": str(
            getattr(torch.cuda.get_device_properties(device), "gcnArchName", "unknown")
        ).split(":", 1)[0],
        "device": device,
        "qualification": "BF16 ROCm source-author parity unvalidated; package-native PyTorch path",
    }
    predictions = []
    counts = {
        "items": 0,
        "questions": 0,
        "valid_questions": 0,
        "invalid_questions": 0,
        "over_budget_questions": 0,
    }
    for row in rows:
        payload = {"state": row["state"], "questions": row["questions"]}
        synchronize(device)
        start = time.perf_counter()
        try:
            results = native.decide_all(**payload)
            answers = {
                key: shared_answer(row["questions"][key], answer)
                for key, (answer, _) in results.items()
            }
            usage = {
                "input_tokens": sum(tokens for _, tokens in results.values()),
                "output_tokens": 0,
            }
            invalid_reason = None
        except ValueError as exc:
            if "tokens > 16000" not in str(exc):
                raise
            answers = {
                key: {"type": question["type"], "error": "context_overflow"}
                for key, question in row["questions"].items()
            }
            usage = None
            invalid_reason = "context_overflow"
        synchronize(device)
        latency_ms = 1000 * (time.perf_counter() - start)
        if set(answers) != set(row["questions"]) or not math.isfinite(latency_ms):
            raise ValueError(f"{row['id']}: incomplete native package answer")
        n = len(answers)
        counts["items"] += 1
        counts["questions"] += n
        counts["valid_questions"] += n if invalid_reason is None else 0
        counts["invalid_questions"] += n if invalid_reason is not None else 0
        counts["over_budget_questions"] += n if invalid_reason is not None else 0
        predictions.append(
            {
                "id": row["id"],
                "answers": answers,
                "usage": usage,
                "latency_ms": latency_ms,
                "source_input_sha256": digest(payload),
                "model": f"{model_id}@{revision}",
                "model_id": model_id,
                "model_revision": revision,
                "model_sha256": identity["model_sha256"],
                "calibration_sha256": identity["calibration_sha256"],
                "rights_mode": identity["rights_mode"],
                "rights_attestation_sha256": identity["rights_attestation_sha256"],
                "backend": "eikos-semif-native",
                "adapter_version": ADAPTER_VERSION,
                "runtime_qualification": runtime["qualification"],
                "invalid_reason": invalid_reason,
            }
        )
    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "model_sha256": identity["model_sha256"],
        "calibration_sha256": identity["calibration_sha256"],
        "rights_mode": identity["rights_mode"],
        "rights_attestation_sha256": identity["rights_attestation_sha256"],
        "calibration": {
            "file_sha256": identity["calibration_sha256"],
            "binding": "embedded_package",
        },
        "source_revision": identity["source_revision"],
        "source_release": identity["source_release"],
        "package_files_checked": identity["package_files_checked"],
        "package_manifest_sha256": identity["model_sha256"],
        "collector_source_sha256": file_sha256(Path(__file__)),
        "input_sha256": file_sha256(prompts),
        "input_items": all_items,
        "evaluated_items": len(rows),
        "max_items": max_items,
        "max_length": 16000,
        "execution": "one benchmark item at a time; all its questions batched by original serve.Decider",
        "truncation_policy": "none; native context overflow produces invalid answers",
        "counts": counts,
        "runtime": runtime,
    }
    write_output(output, predictions, manifest)
    return {
        "output": str(output),
        "items": len(rows),
        "model_sha256": identity["model_sha256"],
        "calibration_sha256": identity["calibration_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--rights-attestation",
        type=Path,
        help="Required for packages trained with restricted noncommercial research sources",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                prompts=args.input,
                output=args.output,
                model_id=args.model_id,
                model_revision=args.model_revision,
                device=args.device,
                max_items=args.max_items,
                rights_attestation=args.rights_attestation,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
