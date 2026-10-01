"""BF16 decision latency of a Decision checkpoint on M1's fixed SELECT roster (one GPU; never a score).

As Milestone 1 (``v2.27b.backbone_probe``, native stage): the 65 SELECT rows with the smallest
SHA-256 of ``"27b-probe/" + id``, one prompt per forward, CUDA-synchronized wall time, the first
prompt dropped as warm-up, p50 / p95 over the other 64. Here the forward is the whole decision
path of the checkpoint: backbone parameters BF16-resident (unmerged LoRA included, the pinned
experts implementation), head in FP32 (it casts itself), the checkpoint's own prompt encoder.
Gated-delta checkpoints (the dense Qwen3.5 reference, Qwen3.5-MoE) run on the image FLA kernel
path (``typed_collect_kernel.kernel_runtime``: an existing ``TRITON_CACHE_DIR`` is required).

    python3 -m v2.27b.moe.latency --checkpoint CKPT --source-path BASE --select SELECT.jsonl \
        --output latency.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

ROSTER_ROWS = 65
MAX_LENGTH = 32768
DENSE_QWEN35 = "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"


def gated_delta(checkpoint: Path) -> bool:
    metadata = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )
    arch = str(metadata.get("architecture"))
    return arch == DENSE_QWEN35 or arch.startswith("qwen3.5-moe")


def roster(
    rows: list[dict[str, Any]], count: int = ROSTER_ROWS
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(("27b-probe/" + row["id"]).encode()).hexdigest(),
    )[:count]


def summary(values: list[float], tokens: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "mean_ms": statistics.fmean(values),
        "tokens_mean": statistics.fmean(tokens),
        "tokens_per_second": sum(tokens) / (sum(values) / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    kernel_runtime = None
    if gated_delta(args.checkpoint):
        kernel = importlib.import_module("v2.27b.typed_collect_kernel")
        kernel_runtime = kernel.kernel_runtime()

    import torch
    import transformers

    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, collate, encoder_for
    from training.model.infer import checkpoint_fingerprint

    device = torch.device("cuda:0")
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = DecisionModel.from_checkpoint(
        args.checkpoint, source_path=args.source_path
    )
    model.backbone.to(device=device, dtype=torch.bfloat16)
    model.head.to(device=device, dtype=torch.float32)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    resident = torch.cuda.memory_allocated(device)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    encode = encoder_for(model.metadata)
    rows = roster(load_partition(args.select, "select"))
    latencies, tokens = [], []
    with torch.inference_mode():
        for row in rows:
            item = encode(row, tokenizer, MAX_LENGTH)
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in collate([item], pad_id).items()
            }
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            model(**batch)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - tick) * 1000)
            tokens.append(len(item["ids"]))
    result = {
        "schema": "decision2-27b-moe-latency/1",
        "checkpoint": str(args.checkpoint),
        "model_sha256": identity["model_sha256"],
        "architecture": model.metadata.get("architecture"),
        "experts_implementation": model.metadata.get("experts_implementation"),
        "prompt_version": model.metadata.get("prompt_version"),
        "precision": "BF16-resident backbone (unmerged LoRA), FP32 head; one prompt per forward",
        "roster": {
            "rule": "65 SELECT rows by SHA-256 of '27b-probe/' + id (M1); first dropped as warm-up",
            "ids_sha256": hashlib.sha256(
                json.dumps([r["id"] for r in rows]).encode()
            ).hexdigest(),
        },
        "load_seconds": load_seconds,
        "resident_bytes_after_load": resident,
        "peak_bytes": torch.cuda.max_memory_allocated(device),
        "decision_latency": summary(latencies[1:], tokens[1:]),
        "kernel_runtime": kernel_runtime,
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["decision_latency"], sort_keys=True))


if __name__ == "__main__":
    main()
