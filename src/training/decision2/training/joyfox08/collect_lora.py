"""Collect gold-free native Joyfox predictions with a frozen LoRA checkpoint.

This is a development-only comparator path for the hard-label control and the
single-variable soft-replay arm. It never opens DEV/CSS labels or sealed keys.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from inference import joyfox
from inference.run import digest, load_prompts, synchronize

from training.model.data import file_sha256

DEV_PROMPTS_SHA256 = "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
CSS_PILOT_PROMPTS_SHA256 = (
    "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda"
)
PROMPT_HASHES = {"dev": DEV_PROMPTS_SHA256, "css_pilot": CSS_PILOT_PROMPTS_SHA256}
ADAPTER_HASHES = {
    "hard_step64": "31cf36c30a6fd268dafadf52fc82c838fdec9f929b4b7adb5214fa7dce574abf",
}


def collect(
    *,
    model_path: Path,
    source_path: Path,
    adapter_path: Path,
    adapter_sha256: str,
    arm: str,
    panel: str,
    prompts: Path,
    output: Path,
    max_items: int | None = None,
) -> dict:
    from peft import PeftModel

    if arm not in {"hard_step64", "soft_step64"} or panel not in PROMPT_HASHES:
        raise ValueError("Unknown frozen arm or development panel")
    if output.exists() or file_sha256(prompts) != PROMPT_HASHES[panel]:
        raise ValueError("Output exists or frozen development prompts changed")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    actual_adapter_sha = file_sha256(adapter_path / "adapter_model.safetensors")
    if actual_adapter_sha != adapter_sha256:
        raise ValueError("Adapter weights differ from frozen step-64 hash")
    if arm in ADAPTER_HASHES and actual_adapter_sha != ADAPTER_HASHES[arm]:
        raise ValueError("Hard-label control checkpoint differs from previous run")
    source = joyfox.verify_release(model_path, source_path, joyfox.MODEL_REVISION)
    rows = load_prompts(prompts)
    if len(rows) != {"dev": 1600, "css_pilot": 1430}[panel]:
        raise ValueError("Development panel count changed")
    engine, runtime = joyfox.load_native(model_path, source_path, "cuda:0", 1024)
    engine.model.backbone = PeftModel.from_pretrained(
        engine.model.backbone, adapter_path, is_trainable=False
    )
    engine.model.requires_grad_(False)
    engine.model.eval()
    if any(parameter.requires_grad for parameter in engine.model.parameters()):
        raise RuntimeError("Development inference model must be frozen")
    identity = {
        **source,
        "adapter_version": "joyfox-native1024-frozen-lora-v1",
        "adapter_arm": arm,
        "adapter_sha256": actual_adapter_sha,
        "native_cutoff_len": 1024,
        "panel": panel,
        "prompts_sha256": PROMPT_HASHES[panel],
        "actual_loaded_parameters": sum(p.numel() for p in engine.model.parameters()),
        **runtime,
    }
    selected = rows if max_items is None else rows[:max_items]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in selected:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize("cuda:0")
            started = time.perf_counter()
            answers, status = joyfox._predict(engine, row)
            synchronize("cuda:0")
            latency_ms = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise RuntimeError("Nonfinite native inference latency")
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": answers,
                        "status": status,
                        "latency_ms": latency_ms,
                        "source_input_sha256": digest(payload),
                        **identity,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
    return {
        **identity,
        "items": len(selected),
        "output_sha256": file_sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model-path", "source-path", "adapter-path", "prompts", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    for option in ("adapter-sha256", "arm", "panel"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--max-items", type=int)
    print(json.dumps(collect(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
