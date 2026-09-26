"""Collect Kev-4B's native typed decisions from gold-free benchmark prompts.

This invokes the pinned release code's Checkpoint, SystemOneRequest, encoder,
calibrated pointer readout, and TypeSafe-compatible answer formatter. It does
not use the faster bf16 HTTP server or optional fused/CUDA-graph paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .run import (
    ADAPTER_VERSION,
    completed_rows,
    digest,
    file_digest,
    load_prompts,
    local_revision,
    synchronize,
)

KEV_MODEL_ID = "jaredpalmer/kev-4b"
KEV_MODEL_REVISION = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
KEV_SOURCE_REVISION = "6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63"
KEV_BASE_ID = "Qwen/Qwen3.5-4B-Base"
KEV_BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def verify_provenance(model_path: Path, source_path: Path) -> dict[str, Any]:
    """Check the published weights and code against the model's provenance."""
    provenance = json.loads(
        (model_path / "provenance.json").read_text(encoding="utf-8")
    )
    if provenance.get("git_commit") != KEV_SOURCE_REVISION:
        raise ValueError(
            "Kev release provenance does not name the pinned source commit"
        )
    if provenance.get("config", {}).get("base_revision") != KEV_BASE_REVISION:
        raise ValueError(
            "Kev release provenance does not name the pinned base revision"
        )
    expected_adapter = provenance.get("measured_checkpoint", {}).get("adapter_sha256")
    if file_digest(model_path / "adapter_model.safetensors") != expected_adapter:
        raise ValueError("Kev adapter differs from the release provenance")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != KEV_SOURCE_REVISION:
        raise ValueError("Kev source checkout is not at the release commit")
    for relative, expected in provenance["source_hashes"].items():
        file = source_path / relative
        if not file.is_file() or file_digest(file) != expected:
            raise ValueError(f"Kev source differs from release provenance: {relative}")
    return provenance


def load_native(model_path: Path, source_path: Path, device: str):
    provenance = verify_provenance(model_path, source_path)
    # The Qwen base must already be in the HF cache from a pinned `hf download`
    # on the experiment host. Forbid an implicit Hub interaction at load time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    loaded = sys.modules.get("kev")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if (
            loaded_file is None
            or Path(loaded_file).resolve().parent != (source_path / "kev").resolve()
        ):
            raise RuntimeError(
                "A different Kev package is already imported; use one model per process"
            )
    sys.path.insert(0, str(source_path))
    from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, ContextOverflow

    checkpoint = Checkpoint(str(model_path))
    if (
        checkpoint.meta.base != KEV_BASE_ID
        or checkpoint.meta.base_revision != KEV_BASE_REVISION
    ):
        raise ValueError("Kev head metadata does not match the pinned base")
    # The default LoadOptions is Kev's exact reported evaluation path: FP32,
    # merged LoRA, fitted checkpoint temperature, eager PyTorch readout.
    tokenizer, model = checkpoint.load(device, LoadOptions())

    def decide(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        request = SystemOneRequest(state=state, questions=questions, model="kev-latest")
        record, metadata = to_record(request)
        try:
            encoded = model.encode(
                tokenizer,
                record,
                max_state=SERVE_MAX_STATE,
                max_branch=SERVE_MAX_BRANCH,
                strict=True,
            )
        except ContextOverflow:
            return {
                "model": f"{KEV_MODEL_ID}@{KEV_MODEL_REVISION}",
                "answers": {
                    name: {"type": spec["type"], "invalid_reason": "context_overflow"}
                    for name, spec in questions.items()
                },
                "status": "context_overflow",
                "usage": None,
            }
        probabilities = model.probs(encoded)
        if len(probabilities) != len(metadata):
            raise ValueError("Kev returned the wrong number of question distributions")
        answers = to_answers([values.tolist() for values in probabilities], metadata)
        return {
            "model": f"{KEV_MODEL_ID}@{KEV_MODEL_REVISION}",
            "answers": answers,
            "usage": {
                "input_tokens": len(encoded["ids"]),
                # The native API calls this output_tokens for billing; no
                # tokens are generated by the pointer classifier.
                "output_tokens": output_tokens(tokenizer, answers),
            },
        }

    runtime = {
        "source_revision": KEV_SOURCE_REVISION,
        "base_revision": KEV_BASE_REVISION,
        "calibration_temperature": checkpoint.meta.temperature,
        "inference_dtype": "float32",
        "inference_path": "Checkpoint.load/DecisionModel.probs/api.to_answers",
        "source_files_verified": len(provenance["source_hashes"]),
        "release_runtime": {
            "torch": provenance.get("torch"),
            "gpu": provenance.get("gpu"),
        },
    }
    return model, decide, runtime


def model_fingerprint(model_path: Path) -> str:
    return digest(
        {
            "provenance": file_digest(model_path / "provenance.json"),
            "head": file_digest(model_path / "head.pt"),
            "adapter": file_digest(model_path / "adapter_model.safetensors"),
        }
    )


def collect(
    *,
    model_path: Path,
    source_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if revision != KEV_MODEL_REVISION:
        raise ValueError(
            "Kev model revision differs from the pinned published revision"
        )
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    model_path, source_path = model_path.resolve(strict=True), source_path.resolve(
        strict=True
    )
    revision_attested = local_revision(model_path, revision)
    if not revision_attested:
        raise ValueError(
            "Kev local download does not attest the pinned Hugging Face revision"
        )
    fingerprint = model_fingerprint(model_path)
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = completed_rows(
            output, rows, "kev", revision, fingerprint, KEV_MODEL_ID, revision_attested
        )
    else:
        completed = set()
    model, decide, runtime = load_native(model_path, source_path, device)
    assert model is not None
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            response = decide(**payload)
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("answers"), dict)
                or response["answers"].keys() != row["questions"].keys()
            ):
                raise ValueError(f"{row['id']}: Kev answers do not match question IDs")
            receipt = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency_ms,
                "usage": response.get("usage"),
                "status": response.get("status", "ok"),
                "model": response.get("model"),
                "backend": "kev",
                "model_id": KEV_MODEL_ID,
                "adapter_version": ADAPTER_VERSION,
                "model_revision": revision,
                "revision_attested": revision_attested,
                "model_config_sha256": fingerprint,
                "source_input_sha256": digest(payload),
                **runtime,
            }
            target.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            target.flush()
    return {
        "backend": "kev",
        "model_id": KEV_MODEL_ID,
        "revision": revision,
        "revision_attested": revision_attested,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "output": str(output),
        "model_config_sha256": fingerprint,
        **runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=KEV_MODEL_REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                source_path=args.source_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
                device=args.device,
                resume=args.resume,
                max_items=args.max_items,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
