"""Collect the pinned Joyfox 0.8B model through its published decision engine.

Only gold-free prompts enter this process. A request beyond the model's native
1024-token row limit receives an explicit invalid answer instead of truncation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest, load_prompts, local_revision, synchronize

MODEL_ID = "joyfox/Qwen3.5-0.8B-JEV"
MODEL_REVISION = "ae7b7040aeff7802f6f2bcfdd27f08a72d5cd969"
SOURCE_REVISION = "2677b5a3714489847668175de793e2d92fe183f0"
MODEL_HASHES = {
    "decision_config.json": "859322c0adc3f2bb5a8b9b7448369ae2de3d216a3b7deff5ae2f8f13c786e0b2",
    "head.safetensors": "14ee01eeeb6cf9f9c0d10ee44ab42720a53e78a2e318447820006ccd795252f9",
    "backbone/config.json": "c4c9d1d4879f95e13d6fb8945461e26da64dbe8e90052c557e597f242bfeea9b",
    "backbone/model.safetensors": "af4cfee3e639739d2a5d19f76923555ba017c801b65bdcdc1f3f8a686bbc9992",
    "tokenizer/chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer/tokenizer_config.json": "83febe739ecefd99f984f546dfdc10149cedf293b9ee0291b99894341188b613",
    "tokenizer/tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
}
ADAPTER_VERSION = "joyfox-native-v1"
EXTENDED_ADAPTER_VERSION = "joyfox-native-extended-context-v1"


def collector_identity(release: dict[str, Any], cutoff_len: int) -> dict[str, Any]:
    """Keep the published 1,024-token contract distinct from context ablations."""
    if cutoff_len < 1 or cutoff_len > 4096:
        raise ValueError("Joyfox cutoff must be between 1 and 4096 tokens")
    if cutoff_len == 1024:
        return release
    return {
        **release,
        "adapter_version": EXTENDED_ADAPTER_VERSION,
        "cutoff_len": cutoff_len,
    }


def verify_release(
    model_path: Path, source_path: Path, revision: str
) -> dict[str, Any]:
    if revision != MODEL_REVISION or not local_revision(model_path, revision):
        raise ValueError("Joyfox download lacks its pinned Hugging Face revision")
    for relative, expected in MODEL_HASHES.items():
        if file_digest(model_path / relative) != expected:
            raise ValueError(f"Joyfox artifact differs: {relative}")
    config = json.loads((model_path / "decision_config.json").read_text())
    expected_config = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "format_version": 2,
        "execution": "rows",
        "adapter": False,
        "option_isolation": False,
        "head_dim": 128,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Joyfox decision configuration changed")
    commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_REVISION:
        raise ValueError("Joyfox native source revision changed")
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if changed or not (source_path / "src/jev_inference/engine.py").is_file():
        raise ValueError("Joyfox native source has changed or is incomplete")
    return {
        "backend": "joyfox",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "source_revision": commit,
        "model_config_sha256": MODEL_HASHES["decision_config.json"],
        "model_head_sha256": MODEL_HASHES["head.safetensors"],
        "adapter_version": ADAPTER_VERSION,
    }


def completed_ids(
    output: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    completed: set[str] = set()
    with output.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            result = json.loads(line)
            item_id = result.get("id")
            if item_id not in expected or item_id in completed:
                raise ValueError(f"{output}:{line_number}: unknown or duplicate ID")
            if any(result.get(key) != value for key, value in identity.items()):
                raise ValueError(f"{output}:{line_number}: stale Joyfox identity")
            fingerprint, questions = expected[item_id]
            if (
                result.get("source_input_sha256") != fingerprint
                or not isinstance(result.get("answers"), dict)
                or set(result["answers"]) != questions
            ):
                raise ValueError(f"{output}:{line_number}: stale input or answer map")
            completed.add(item_id)
    return completed


def load_native(model_path: Path, source_path: Path, device: str, cutoff_len: int):
    if device != "cuda:0":
        raise ValueError("Expose one ROCm GPU as cuda:0 for Joyfox")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    loaded = sys.modules.get("jev_inference")
    source_package = (source_path / "src/jev_inference").resolve()
    if loaded is not None and Path(loaded.__file__).resolve().parent != source_package:
        raise RuntimeError("A different jev_inference package is already loaded")
    sys.path.insert(0, str(source_path / "src"))
    import torch
    import transformers
    from jev_inference import DecisionEngine

    if not torch.cuda.is_available():
        raise RuntimeError("Joyfox native inference requires ROCm GPU")
    engine = DecisionEngine.load(
        model_path, device=device, dtype="bfloat16", cutoff_len=cutoff_len
    )
    if next(engine.model.parameters()).device.type != "cuda":
        raise RuntimeError("Joyfox silently fell back to CPU")
    return engine, {
        "runtime_qualification": "unvalidated_rocm_native",
        "runtime_torch": torch.__version__,
        "runtime_hip": torch.version.hip,
        "runtime_transformers": transformers.__version__,
        "actual_parameters": sum(param.numel() for param in engine.model.parameters()),
    }


def _predict(engine: Any, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {"state": row["state"], "questions": row["questions"]}
    try:
        answers = engine.predict(payload)
    except ValueError as exc:
        if "input requires" not in str(exc) and "state exceeds" not in str(exc):
            raise
        return (
            {
                qid: {"type": question["type"], "invalid_reason": "context_overflow"}
                for qid, question in row["questions"].items()
            },
            "context_overflow",
        )
    if not isinstance(answers, dict) or set(answers) != set(row["questions"]):
        raise ValueError(f"{row['id']}: Joyfox native answer keys differ")
    return answers, "ok"


def collect(
    *,
    model_path: Path,
    source_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    cutoff_len: int = 1024,
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    model_path, source_path = model_path.resolve(strict=True), source_path.resolve(
        strict=True
    )
    identity = collector_identity(
        verify_release(model_path, source_path, revision), cutoff_len
    )
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = completed_ids(output, rows, identity)
    else:
        completed = set()
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    if not remaining:
        return {**identity, "input_items": len(rows), "collected_now": 0}
    engine, runtime = load_native(model_path, source_path, device, cutoff_len)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            answers, status = _predict(engine, row)
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise ValueError("Nonfinite Joyfox inference latency")
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": answers,
                        "status": status,
                        "latency_ms": latency_ms,
                        "source_input_sha256": digest(payload),
                        **identity,
                        **runtime,
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
        **runtime,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff-len", type=int, default=1024)
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
                cutoff_len=args.cutoff_len,
                resume=args.resume,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
