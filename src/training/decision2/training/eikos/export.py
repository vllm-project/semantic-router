"""Export a selected Eikos LoRA as a standalone native SemIf candidate.

The export is local and intentionally has no release model card or HF upload.
Evaluate the merged result against the selected adapter before publication.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from training.eikos.io import atomic_json
from training.eikos.native import selected_checkpoint
from training.eikos.rights import (
    SCOPE,
    SPLIT_NAMES,
    clean_manifest,
    verify_noncommercial_attestation,
)
from training.model.data import file_sha256

NATIVE_FILES = (
    "decision_core.py",
    "letter_adapter.py",
    "serve.py",
    "serve_vllm.sh",
    "decision_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "LICENSE-Qwen",
    "NOTICE",
)


def verify_calibration(
    calibration: Path, report_path: Path, selected: dict[str, Any]
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("checkpoint") != selected["name"]
        or report.get("adapter_weights_sha256") != selected["adapter_weights_sha256"]
        or report.get("calibration_sha256") != file_sha256(calibration)
    ):
        raise ValueError("Calibration is not bound to the native-selected adapter")
    config = json.loads(calibration.read_text(encoding="utf-8"))
    if (
        set(config) != {"b", "w", "features", "mode"}
        or config["features"] != ["log_ntok", "log_nopts", "noul", "score"]
        or not isinstance(config["w"], list)
        or len(config["w"]) != 4
    ):
        raise ValueError(
            "Calibration does not implement the native Eikos temperature contract"
        )
    return report


def export(
    *,
    model_path: Path,
    run: Path,
    calibration: Path,
    report: Path,
    data_manifest: Path,
    output: Path,
    rights_attestation: Path | None = None,
    model_name: str = "dev-2.0-4b",
) -> dict[str, Any]:
    if model_name != "dev-2.0-4b":
        raise ValueError("Public 4B family model name must be dev-2.0-4b")
    selected = selected_checkpoint(run, model_path)
    calibration_report = verify_calibration(calibration, report, selected)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    manifest_sha = file_sha256(data_manifest)
    data_sha = provenance["data_sha256"]
    counts = (
        provenance["train_input_examples"],
        provenance["select_examples"],
        provenance["cal_examples_audited_only"],
    )
    if provenance.get("rights_manifest_sha256") == manifest_sha:
        if (
            provenance.get("publication_eligible") is not True
            or provenance.get("publication_scope") != SCOPE
        ):
            raise ValueError("Clean LoRA provenance has different publication terms")
        data_receipt = clean_manifest(
            data_manifest,
            dict(
                zip(
                    SPLIT_NAMES,
                    (
                        data_sha["train"],
                        data_sha["select"],
                        data_sha["cal_audited_only"],
                    ),
                )
            ),
            dict(zip(SPLIT_NAMES, counts)),
        )
        rights_mode = "clean"
        rights_sha = None
        source_rights = data_receipt["source_rights"]
        conditions = data_receipt["publication_conditions"]
    else:
        if rights_attestation is None:
            raise ValueError(
                "Original Eikos pilot requires a noncommercial research attestation"
            )
        statement = verify_noncommercial_attestation(
            rights_attestation, data_manifest, run / "provenance.json", data_sha, counts
        )
        data_receipt = json.loads(data_manifest.read_text(encoding="utf-8"))
        rights_mode = "noncommercial_research"
        rights_sha = file_sha256(rights_attestation)
        source_rights = {
            "source_groups": statement["source_groups"],
            "rights_conditions": statement["rights_conditions"],
        }
        conditions = [statement["publication_scope"]]
    if provenance["prompt_version"] != "letter-v1-semif":
        raise ValueError("Unexpected Eikos prompt version")
    source_config = json.loads(
        (model_path / "decision_config.json").read_text(encoding="utf-8")
    )
    if source_config != {
        "prompt_version": "letter-v1-semif",
        "readout": "letter-logit",
        "calib": "calib.json",
        "max_one_pass": 100,
    }:
        raise ValueError("Source Eikos decision config differs from the tested release")
    if output.exists():
        raise FileExistsError(output)
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(pending)
    pending.mkdir(parents=True)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        expected_parameters = sum(parameter.numel() for parameter in base.parameters())
        adapted = PeftModel.from_pretrained(base, selected["adapter"])
        merged = adapted.merge_and_unload()
        if (
            sum(parameter.numel() for parameter in merged.parameters())
            != expected_parameters
        ):
            raise RuntimeError("Merged Eikos parameter count differs from source model")
        if any("lora_" in name for name in merged.state_dict()):
            raise RuntimeError("Merged Eikos still contains adapter parameters")
        merged.save_pretrained(pending, safe_serialization=True, max_shard_size="4GB")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.save_pretrained(pending)
        for name in NATIVE_FILES:
            shutil.copy2(model_path / name, pending / name)
        shutil.copy2(calibration, pending / "calib.json")
        if (
            json.loads((pending / "decision_config.json").read_text(encoding="utf-8"))[
                "calib"
            ]
            != "calib.json"
        ):
            raise ValueError(
                "Export calibration is not discoverable by the native server"
            )
        manifest = {
            "model_name": model_name,
            "candidate_status": "unreleased_requires_merged_parity_and_final_evaluation",
            "architecture": "Qwen3.5ForCausalLM with Eikos native SemIf letter-logit readout",
            "source_model": provenance["model_id"],
            "source_revision": provenance["model_revision"],
            "source_release": selected["source_release"],
            "selected_checkpoint": selected["name"],
            "native_select_accuracy": selected["native_selection_metrics"][
                "micro_accuracy"
            ],
            "native_select_brier": selected["native_selection_metrics"][
                "family_macro_brier"
            ],
            "adapter_weights_sha256": selected["adapter_weights_sha256"],
            "adapter_config_sha256": selected["adapter_config_sha256"],
            "training_provenance_sha256": selected["provenance_sha256"],
            "calibration_sha256": file_sha256(calibration),
            "calibration_report_sha256": file_sha256(report),
            "calibration_data_sha256": calibration_report["cal_sha256"],
            "training_data_sha256": provenance["data_sha256"],
            "training_data_manifest_sha256": manifest_sha,
            "rights_manifest_sha256": manifest_sha,
            "rights_schema_version": data_receipt["schema_version"],
            "rights_mode": rights_mode,
            "rights_attestation_sha256": rights_sha,
            "publication_eligible": True,
            "publication_scope": (
                data_receipt["publication_scope"]
                if rights_mode == "clean"
                else statement["publication_scope"]
            ),
            "training_source_rights": source_rights,
            "training_source_counts": data_receipt.get("counts", {}).get("source"),
            "publication_conditions": conditions,
            "effective_train_payload_sha256": provenance[
                "effective_train_payload_sha256"
            ],
            "quarantine_sha256": provenance["quarantine_sha256"],
            "training_code_sha256": provenance["code_sha256"],
            "source_notice": "Eikos MIT contribution + Qwen Apache-2.0 base; original LICENSE, LICENSE-Qwen and NOTICE copied verbatim",
        }
        atomic_json(pending / "decision2_provenance.json", manifest)
        all_files = sorted(path for path in pending.rglob("*") if path.is_file())
        with (pending / "SHA256SUMS").open("w", encoding="utf-8") as stream:
            for path in all_files:
                stream.write(f"{file_sha256(path)}  {path.relative_to(pending)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, output)
        return {
            "output": str(output),
            "model_name": model_name,
            "source_revision": provenance["model_revision"],
            "selected_checkpoint": selected["name"],
            "weight_shards": sorted(
                path.name for path in output.glob("model*.safetensors")
            ),
            "sha256_manifest": file_sha256(output / "SHA256SUMS"),
            "status": "candidate_needs_parity_and_final_evaluation",
        }
    except BaseException:
        shutil.rmtree(pending)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--rights-attestation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="dev-2.0-4b")
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                model_path=args.model_path,
                run=args.run,
                calibration=args.calibration,
                report=args.calibration_report,
                data_manifest=args.data_manifest,
                output=args.output,
                rights_attestation=args.rights_attestation,
                model_name=args.model_name,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
