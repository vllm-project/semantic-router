"""One bounded, input-only Qwen3-0.6B option-branch cache feasibility run.

This technical probe produces no model score or decision prediction. It must
run only on an authorized GPU with an exact official source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from .data import file_sha256
from .option_isolation_cache import branch_tokens, independent_endpoints
from .source import source_fingerprint

SOURCE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
CONFIG_SHA256 = "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"
WEIGHT_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
PARITY_ABS_LIMIT = 0.01
PARITY_COSINE_FLOOR = 0.999
PEAK_BYTES_LIMIT = 40 * 1024**3
CACHED_255_SECONDS_LIMIT = 120


def synthetic_row(kind: str, count: int) -> dict[str, Any]:
    state = " ".join(f"record{i} has value {i % 17}." for i in range(128))
    if kind == "noul":
        options = [
            {"key": "false", "description": "The condition does not hold"},
            {"key": "true", "description": "The condition holds"},
        ]
    elif kind == "score":
        options = [
            {"key": str(i), "description": f"Evidence strength level {i}"}
            for i in range(count)
        ]
    else:
        options = [
            {"key": f"item_{i:03d}", "description": f"Select record {i} if it matches"}
            for i in range(count)
        ]
    return {
        "state": state,
        "task_type": kind,
        "instructions": "Select the supported outcome from the records.",
        "options": options,
    }


def source_check(source: Path) -> dict[str, Any]:
    files = source_fingerprint(source)["files_sha256"]
    if files.get("config.json") != CONFIG_SHA256:
        raise ValueError("Official source config hash mismatch")
    weight_files = sorted(source.glob("*.safetensors"))
    if len(weight_files) != 1 or files.get(weight_files[0].name) != WEIGHT_SHA256:
        raise ValueError("Official source weight hash mismatch")
    return {
        "source_revision": SOURCE_REVISION,
        "files_sha256": files,
    }


def _measure(
    backbone: Any,
    prefix: list[int],
    tails: list[list[int]],
    *,
    pad_id: int,
    reuse_prefix: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        vectors = independent_endpoints(
            backbone,
            prefix,
            tails,
            pad_id=pad_id,
            device=torch.device("cuda:0"),
            chunk_size=8,
            max_length=8192,
            reuse_prefix=reuse_prefix,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    return vectors, {"seconds": elapsed, "peak_allocated_bytes": peak}


def _compare(cached: torch.Tensor, naive: torch.Tensor) -> dict[str, float]:
    if cached.shape != naive.shape or not torch.isfinite(cached).all():
        raise ValueError("Branch hidden states are missing or nonfinite")
    if not torch.isfinite(naive).all():
        raise ValueError("Reference hidden states are nonfinite")
    abs_max = (cached - naive).abs().max().item()
    cosine_min = torch.nn.functional.cosine_similarity(cached, naive).min().item()
    if not math.isfinite(abs_max) or not math.isfinite(cosine_min):
        raise ValueError("Invalid branch parity statistics")
    return {"max_abs": abs_max, "min_cosine": cosine_min}


def probe(source: Path) -> dict[str, Any]:
    import transformers

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    source_identity = source_check(source)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One BF16 CUDA/ROCm GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Official source tokenizer has no pad/EOS ID")
    full, loading_info = Qwen3ForCausalLM.from_pretrained(
        source,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        loading_info.get(name)
        for name in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("Official Qwen source did not load exactly")
    backbone = full.model.to("cuda:0").eval()
    del full
    rows = [
        ("choice", 2),
        ("choice", 3),
        ("choice", 10),
        ("choice", 255),
        ("noul", 2),
        ("score", 3),
        ("score", 10),
    ]
    cases: list[dict[str, Any]] = []
    overall_pass = True
    measured_seconds = 0.0
    for kind, count in rows:
        row = synthetic_row(kind, count)
        prefix, tails = branch_tokens(row, tokenizer)
        cached, cached_cost = _measure(
            backbone, prefix, tails, pad_id=pad_id, reuse_prefix=True
        )
        checked_indices = (
            [0, 1, 2, 3, count - 4, count - 3, count - 2, count - 1]
            if count == 255
            else list(range(count))
        )
        reference, reference_cost = _measure(
            backbone,
            prefix,
            [tails[index] for index in checked_indices],
            pad_id=pad_id,
            reuse_prefix=False,
        )
        measured_seconds += cached_cost["seconds"] + reference_cost["seconds"]
        parity = _compare(cached[checked_indices], reference)
        passed = (
            parity["max_abs"] <= PARITY_ABS_LIMIT
            and parity["min_cosine"] >= PARITY_COSINE_FLOOR
        )
        order_parity = None
        if kind == "choice" and count == 10:
            reordered = {**row, "options": list(reversed(row["options"]))}
            reordered_prefix, reordered_tails = branch_tokens(reordered, tokenizer)
            if reordered_prefix != prefix:
                raise ValueError("Reordering options changed the shared prefix")
            reversed_cached, reverse_cost = _measure(
                backbone,
                prefix,
                reordered_tails,
                pad_id=pad_id,
                reuse_prefix=True,
            )
            measured_seconds += reverse_cost["seconds"]
            order_parity = _compare(reversed_cached, cached.flip(0))
            passed = (
                passed
                and order_parity["max_abs"] <= PARITY_ABS_LIMIT
                and order_parity["min_cosine"] >= PARITY_COSINE_FLOOR
            )
        if count == 255:
            passed = (
                passed
                and cached_cost["peak_allocated_bytes"] < PEAK_BYTES_LIMIT
                and cached_cost["seconds"] < CACHED_255_SECONDS_LIMIT
            )
        overall_pass &= passed
        cases.append(
            {
                "kind": kind,
                "options": count,
                "prefix_tokens": len(prefix),
                "tail_tokens": [len(tail) for tail in tails],
                "checked_indices": checked_indices,
                "cached": cached_cost,
                "reference_checked": reference_cost,
                "parity": parity,
                "order_parity": order_parity,
                "pass": passed,
            }
        )
    return {
        "schema": "decision2-option-isolation-prefix-cache-technical/1",
        "status": "PASS_TECHNICAL_ONLY" if overall_pass else "HOLD_TECHNICAL",
        "source": source_identity,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "case_count": len(cases),
        "measured_model_seconds": measured_seconds,
        "cases": cases,
        "no_model_score_or_prediction": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Technical receipt already exists")
    result = probe(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_count": result["case_count"],
                "receipt_sha256": file_sha256(args.output),
                "code_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            name: file_sha256(Path(__file__).with_name(name))
                            for name in (
                                "option_isolation.py",
                                "option_isolation_cache.py",
                                "option_isolation_cache_probe.py",
                            )
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
