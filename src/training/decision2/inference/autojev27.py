"""Pinned, package-native AutoJev-27B typed-decision comparator.

This reads the released DecisionModel/answer functions from a separate, pinned
source checkout. Unsupported questions stay in the panel as invalid answers.
It does not use Decision 2.0 weights or any evaluation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest, load_prompts, local_revision, synchronize
from scripts.baseline_attestation_v3 import _tree, _weights

MODEL_ID = "denis-pplx/autojev-27b"
MODEL_REVISION = "6f5b557e037f5edb25c7dc92dbc6553e5a19c015"
SOURCE_REVISION = "ee63c1515980491a742f0bd0685c8dc5ca1f00c3"
BASE_ID = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BACKEND = "autojev-native-eager"
ADAPTER_VERSION = "autojev27-native-v1"


def verify_release(
    model_path: Path, source_path: Path, revision: str
) -> dict[str, Any]:
    """Recompute the full released package/source inventory before inference."""
    if revision != MODEL_REVISION or not local_revision(model_path, revision):
        raise ValueError("AutoJev weights lack the pinned HF CLI revision")
    commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_path), "status", "--porcelain"], text=True
    ).strip()
    if commit != SOURCE_REVISION or dirty:
        raise ValueError("AutoJev native runtime is not the pinned clean source")
    config = json.loads(
        (model_path / "decision_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("format_version") != 1
        or config.get("base_model") != BASE_ID
        or config.get("revision") != BASE_REVISION
        or not isinstance(config.get("codes"), list)
        or not isinstance(config.get("token_ids"), list)
        or len(config["codes"]) != 255
        or len(config["token_ids"]) != 255
        or type(config.get("temperature")) not in (int, float)
        or not math.isfinite(config["temperature"])
        or config["temperature"] <= 0
    ):
        raise ValueError("AutoJev native decision config differs from pinned contract")
    files = _tree(model_path)
    runtime_files = _tree(source_path)
    weights, parameter_count = _weights(model_path, files)
    if (
        not any(name.startswith("model-") for name in weights)
        or "readout.safetensors" not in weights
    ):
        raise ValueError("AutoJev full backbone or decision readout is missing")
    native_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    runtime_sha = hashlib.sha256(
        json.dumps(runtime_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "native_model_sha256": native_sha,
        "runtime_source_sha256": runtime_sha,
        "model_config_sha256": files["decision_config.json"],
        "loaded_parameters": parameter_count,
        "calibration_temperature": float(config["temperature"]),
    }


def _admission_reason(error: ValueError) -> str | None:
    message = str(error).lower()
    if "8192-token limit" in message:
        return "context_overflow"
    if "1 to 255 options" in message or "option" in message and "255" in message:
        return "candidate_limit"
    return None


def _native_answer(
    model: Any, state: Any, question: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    from autojev.model import answer

    row = {"state": state, "question": question}
    batch = model.prepare([row], max_length=8192)
    import torch

    with torch.inference_mode():
        probabilities = (model(batch) / model.temperature).softmax(-1)[0]
    values = probabilities[: batch.counts[0]].float().cpu().tolist()
    return answer(question, values), batch.input_tokens


def collect(
    *,
    model_path: Path,
    source_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    max_items: int | None = None,
) -> dict[str, Any]:
    if not device.startswith("cuda:") or max_items is not None and max_items < 1:
        raise ValueError("AutoJev requires an explicit GPU and positive item limit")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows = load_prompts(prompts)
    if max_items is not None:
        rows = rows[:max_items]
    model_path = model_path.resolve(strict=True)
    source_path = source_path.resolve(strict=True)
    release = verify_release(model_path, source_path, revision)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_path / "src"))
    from autojev.model import DecisionModel

    model = DecisionModel(checkpoint=model_path, device=device, train=False)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_count != release["loaded_parameters"]:
        raise ValueError("AutoJev loaded parameter count differs from package")
    identity = {
        "backend": BACKEND,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "adapter_version": ADAPTER_VERSION,
        "native_model_sha256": release["native_model_sha256"],
        "runtime_source_sha256": release["runtime_source_sha256"],
        "model_config_sha256": release["model_config_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            synchronize(device)
            start = time.perf_counter()
            answers: dict[str, dict[str, Any]] = {}
            tokens = 0
            for key, question in row["questions"].items():
                try:
                    answers[key], used = _native_answer(model, row["state"], question)
                    tokens += used
                except ValueError as error:
                    reason = _admission_reason(error)
                    if reason is None:
                        raise
                    answers[key] = {"type": question["type"], "error": reason}
            synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            if not math.isfinite(elapsed) or set(answers) != set(row["questions"]):
                raise ValueError("AutoJev native inference omitted a question")
            record = {
                "id": row["id"],
                "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "latency_ms": elapsed,
                "source_input_sha256": digest(
                    {"state": row["state"], "questions": row["questions"]}
                ),
                "model": f"{MODEL_ID}@{revision}",
                "runtime_qualification": "pytorch_bf16_rocm_pending_repeatability",
                **identity,
            }
            stream.write(
                json.dumps(
                    record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    return {
        "input_items": len(rows),
        "output_sha256": file_digest(output),
        "loaded_parameters": actual_count,
        **identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
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
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
