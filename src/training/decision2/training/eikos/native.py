"""Attest a selected Eikos LoRA and load its released serving readout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from inference.eikos import REVISION, verify_release

from training.model.data import file_sha256


def selected_checkpoint(
    run: Path, model_path: Path, *, checkpoint: str | None = None
) -> dict[str, Any]:
    run = run.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("model_revision") != REVISION:
        raise ValueError("Eikos LoRA provenance has a different source revision")
    actual_release = verify_release(model_path, REVISION)
    if provenance.get("source_release") != actual_release:
        raise ValueError("Eikos LoRA provenance source release mismatch")
    if file_sha256(run / "quarantine.jsonl") != provenance.get("quarantine_sha256"):
        raise ValueError("Eikos LoRA quarantine roster changed")
    complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    best = json.loads((run / "BEST.json").read_text(encoding="utf-8"))
    native_selection = run / "NATIVE_BEST.json"
    native_best = None
    if checkpoint is None and not native_selection.is_file():
        raise ValueError(
            "Native serving SELECT rerank is required before calibration or inference"
        )
    if checkpoint is None:
        native_best = json.loads(native_selection.read_text(encoding="utf-8"))
        if native_best.get("select_sha256") != provenance["data_sha256"]["select"]:
            raise ValueError("Native SELECT selection refers to different data")
        name = native_best["checkpoint"]
    else:
        name = checkpoint
    if name == "source":
        raise ValueError(
            "SELECT kept the original Eikos checkpoint; no improved LoRA to load"
        )
    if not name.startswith("checkpoint-") or "/" in name or ".." in name:
        raise ValueError("Invalid checkpoint name")
    if complete.get("status") != "complete" or complete.get("best") != best.get(
        "checkpoint"
    ):
        raise ValueError(
            "Eikos LoRA run has not passed final SELECT checkpoint selection"
        )
    checkpoint_path = (run / name).resolve(strict=True)
    if checkpoint_path.parent != run:
        raise ValueError("Checkpoint escaped its run directory")
    info = json.loads((checkpoint_path / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        info.get("source_release") != actual_release
        or info.get("prompt_version") != "letter-v1-semif"
    ):
        raise ValueError("Checkpoint source or prompt contract differs")
    adapter = checkpoint_path / "adapter"
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError("Selected Eikos LoRA has no complete adapter")
    adapter_config = json.loads(config.read_text(encoding="utf-8"))
    expected_lora = provenance["lora"]
    targets = set(adapter_config.get("target_modules", []))
    expected_targets = set(expected_lora["target_modules"])
    expected_suffixes = {target.rsplit(".", 1)[-1] for target in expected_targets}
    if (
        adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("r") != expected_lora["rank"]
        or adapter_config.get("lora_alpha") != expected_lora["alpha"]
        or adapter_config.get("lora_dropout") != expected_lora["dropout"]
        or adapter_config.get("bias") != "none"
        or targets not in (expected_targets, expected_suffixes)
    ):
        raise ValueError("Selected adapter config differs from training provenance")
    weights_hash = file_sha256(weights)
    if (
        native_best is not None
        and native_best["adapter_sha256_by_checkpoint"].get(name) != weights_hash
    ):
        raise ValueError("Selected adapter differs from native SELECT comparison")
    return {
        "name": name,
        "path": checkpoint_path,
        "adapter": adapter,
        "provenance_sha256": file_sha256(provenance_path),
        "adapter_config_sha256": file_sha256(config),
        "adapter_weights_sha256": weights_hash,
        "selection_metrics": info["metrics"],
        "native_selection_metrics": (
            native_best["metrics"][name] if native_best else None
        ),
        "source_release": actual_release,
    }


def load_decider(
    model_path: Path, adapter: Path | None, calibration: Path, *, device: str = "cuda:0"
):
    model_path = model_path.resolve(strict=True)
    adapter = adapter.resolve(strict=True) if adapter is not None else None
    calibration = calibration.resolve(strict=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PROMPT_STYLE"] = "semif"
    sys.path.insert(0, str(model_path))
    import decision_core
    from serve import Decider

    decision_core.set_max_one_pass(100)
    args = argparse.Namespace(
        model=str(model_path),
        adapter=str(adapter) if adapter is not None else None,
        temp=1.0,
        calib=str(calibration),
        max_tokens=16000,
        verify_budget=0,
        vllm_url=None,
        sglang_url=None,
        device=device,
        sym=False,
    )
    return Decider(args)
