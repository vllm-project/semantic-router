"""Collect gold-free predictions from a fixed GLiNER2.5 continuation checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from . import gliner25
from .run import digest, file_digest, load_prompts, synchronize

ADAPTER_VERSION = "gliner25-native-exclusive-v3"
BACKEND = "gliner25-native-continuation-pilot"


def identity(checkpoint: Path, weights_sha256: str) -> dict[str, Any]:
    if os.environ.get("GLINER2_SOURCE_COMMIT") != gliner25.LIBRARY_COMMIT:
        raise RuntimeError("GLiNER2 source commit is not pinned")
    if file_digest(checkpoint / "model.safetensors") != weights_sha256:
        raise ValueError("Candidate checkpoint weights mismatch")
    return {
        "backend": BACKEND,
        "model_id": "research/gliner25-0.6b-step64",
        "model_revision": weights_sha256,
        "revision_attested": True,
        "library_commit": gliner25.LIBRARY_COMMIT,
        "adapter_version": ADAPTER_VERSION,
        "prompt_projection": "state-text; instruction-and-criteria-native-schema",
        "model_weights_sha256": weights_sha256,
        "model_config_sha256": file_digest(checkpoint / "config.json"),
        "tokenizer_sha256": file_digest(checkpoint / "tokenizer.json"),
    }


def collect(
    *, checkpoint: Path, weights_sha256: str, prompts: Path, output: Path, device: str
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    checkpoint = checkpoint.resolve(strict=True)
    model_identity = identity(checkpoint, weights_sha256)
    rows = load_prompts(prompts)

    import torch
    from gliner2.classification.engine import Classifier

    native = (
        Classifier.from_pretrained(str(checkpoint), device=device, dtype=torch.float32)
        .to(device)
        .eval()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    valid = overflow = 0
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            answers = {}
            invalid_reason = None
            for name, question in row["questions"].items():
                try:
                    answers[name] = gliner25.score_question(
                        native, row["state"], question
                    )
                    valid += 1
                except gliner25.NativeContextOverflow as exc:
                    answers[name] = {
                        "type": question["type"],
                        "error": "context_overflow",
                        "native_input_tokens": exc.tokens,
                        "native_max_positions": exc.limit,
                    }
                    invalid_reason = "context_overflow"
                    overflow += 1
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency_ms):
                raise ValueError("Nonfinite native latency")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": latency_ms,
                "usage": None,
                "source_input_sha256": digest(payload),
                "model": model_identity["model_id"],
                "runtime_qualification": "pytorch_fp32_rocm_unvalidated",
                "context_policy": "native-tokenizer-default",
                "invalid_reason": invalid_reason,
                **model_identity,
            }
            stream.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    return {
        "input_sha256": file_digest(prompts),
        "predictions_sha256": file_digest(output),
        "collector_sha256": file_digest(Path(__file__)),
        "items": len(rows),
        "valid_answers": valid,
        "context_overflows": overflow,
        "model_identity": model_identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    report = collect(
        checkpoint=args.checkpoint,
        weights_sha256=args.weights_sha256,
        prompts=args.input,
        output=args.output,
        device=args.device,
    )
    args.receipt.write_text(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
