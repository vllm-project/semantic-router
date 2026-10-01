"""Experts-kernel probe for one MoE base (GPU, container; never trains, never selects).

Loads the official base with a seeded random head on the MoE pipeline, then:
1. parity: SELECT rows (FP32 parameters, BF16 autocast, FP32 head, inference mode)
   under ``eager`` and ``grouped_mm`` experts; max |dp| and argmax changes;
2. speed: with the milestone's rank-32 LoRA attached and gradient checkpointing,
   seconds per training row (forward + backward, micro-batch 1) over the first
   ``--train-rows`` rows of the frozen mixture, per implementation, after one warm-up row;
3. peak allocated memory.
Writes one JSON receipt (aggregates only).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from training.model.decision_model import DecisionModel, collate, encoder_for
from training.model.lora import attach_lora


def rows(path: Path, count: int) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                out.append(json.loads(line))
            if len(out) == count:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--source-stage", choices=("base", "posttrained"), required=True
    )
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select-rows", type=int, default=64)
    parser.add_argument("--train-rows", type=int, default=12)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(20260926)
    started = time.perf_counter()
    model, tokenizer = DecisionModel.from_base(
        args.model_path,
        args.revision,
        256,
        source_stage=args.source_stage,
        experts_implementation="eager",
    )
    load_seconds = time.perf_counter() - started
    model = model.float().to(device).eval()
    encode = encoder_for(model.metadata)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    select = [
        encode(row, tokenizer, args.max_length)
        for row in rows(args.select, args.select_rows)
    ]

    def probabilities(implementation: str) -> list[list[float]]:
        model.backbone.set_experts_implementation(implementation)
        out = []
        with torch.inference_mode():
            for item in select:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in collate([item], pad_id).items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch)
                out.append(
                    logits.float().softmax(-1)[0, : len(item["keys"])].cpu().tolist()
                )
        return out

    parity = {}
    reference = probabilities("eager")
    for implementation in ("eager", "grouped_mm"):
        try:
            values = probabilities(implementation)
        except Exception as exc:  # noqa: BLE001 - recorded, decides the pin
            parity[implementation] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            continue
        worst = max(
            max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(values, reference)
        )
        changed = sum(
            max(range(len(x)), key=x.__getitem__)
            != max(range(len(y)), key=y.__getitem__)
            for x, y in zip(values, reference)
        )
        parity[implementation] = {
            "rows": len(values),
            "max_abs_probability_diff": worst,
            "argmax_changes": changed,
        }

    attach_lora(
        model,
        rank=32,
        alpha=64,
        dropout=0.05,
        source_kind=args.source_stage,
        source_fingerprint={
            "source_name": args.model_path.name,
            "files_sha256": {"probe": "not-hashed"},
        },
    )
    model = model.float().to(device).train()
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    train = [
        encode(row, tokenizer, args.max_length)
        for row in rows(args.train, args.train_rows + 1)
    ]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    speed = {}
    for implementation in ("eager", "grouped_mm"):
        if "error" in parity.get(implementation, {}):
            continue
        model.backbone.base_model.model.set_experts_implementation(implementation)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for index, item in enumerate(train):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in collate([item], pad_id).items()
            }
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"])
            loss.backward()
            torch.cuda.synchronize(device)
            if index:
                timings.append(time.perf_counter() - tick)
            model.zero_grad(set_to_none=True)
        tokens = sum(len(item["ids"]) for item in train[1:])
        speed[implementation] = {
            "rows": len(timings),
            "tokens": tokens,
            "seconds_per_row": sum(timings) / len(timings),
            "tokens_per_second": tokens / sum(timings),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    result = {
        "schema_version": "decision2-27b-moe-experts-probe/1",
        "model": args.model_path.name,
        "revision": args.revision,
        "metadata": {
            k: model.metadata.get(k)
            for k in ("architecture", "prompt_version", "moe", "text_parameter_count")
        },
        "load_seconds": load_seconds,
        "trainable_parameters_rank32": trainable,
        "lora_targets": len(model.metadata["lora"]["target_modules"]),
        "parity_vs_eager": parity,
        "speed": speed,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
