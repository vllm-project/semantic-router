"""Read-only, pinned 0.6B Choice/Noul/Score gradient-conflict preflight.

The default plan is CPU-only and does not load model weights. ``--measure`` is
an explicit one-GPU BF16 diagnostic; neither mode constructs an optimizer or
accepts SELECT, CAL, benchmark prompts, or their answer keys. Receipts contain
aggregate gradient statistics and hashes, never training text or gold labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import digest, file_sha256, load_partition
from .decision_model import PROMPT_VERSION, TASK_TYPES, DecisionModel, collate, encode
from .loss import per_example_loss
from .plan import epoch_batches

SCHEMA = "decision2-06b-gradient-conflict-preflight/1"
SOURCE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
SOURCE_FILES = {
    "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
}
TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
TRAIN_COUNT = 7_455
TRAIN_TOKENS = 4_094_489
SEED = 20260926
ACCUMULATION = 16
WINDOW_COUNT = 8
MAX_LENGTH = 8192
COSINE_THRESHOLD = -0.05
MIN_CONFLICT_WINDOWS = 3
MAX_MEASURE_SECONDS = 360.0
RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "created_utc",
    "failure_class",
    "source_revision",
    "source_files_sha256",
    "code_sha256",
    "train_sha256",
    "train_rows",
    "train_native_tokens",
    "prompt_version",
    "seed",
    "microbatch",
    "accumulation",
    "planned_control_updates",
    "schedule_sha256",
    "windows",
    "conflict_cosine_threshold",
    "min_conflict_windows",
    "optimizer_constructed",
    "optimizer_steps",
    "device_precision",
    "measured_windows",
    "conflict_windows",
    "measurement_seconds",
    "measurement_gpu_hours",
}
WINDOW_FIELDS = {
    "optimizer_window",
    "rows",
    "native_tokens",
    "type_counts",
    "roster_sha256",
    "ordinary_backbone_norm",
    "task_backbone_norms",
    "pairwise_cosines",
    "max_reconstruction_error",
    "choice_or_score_conflict",
}


def select_windows(
    rows: list[dict[str, Any]], lengths: list[int]
) -> tuple[list[tuple[int, list[int]]], list[dict[str, Any]], str]:
    """Recreate the trainer's epoch-0 windows, selecting by type only."""
    if len(rows) != len(lengths) or not rows or any(length < 1 for length in lengths):
        raise ValueError("Invalid TRAIN rows or encoded lengths")
    batches = epoch_batches(
        lengths, [], epoch=0, seed=SEED, microbatch=1, replay_fraction=0.0
    )
    flat = [index for batch in batches for source, index in batch if source == "train"]
    if len(flat) != len(rows) or set(flat) != set(range(len(rows))):
        raise ValueError("Trainer schedule omitted or duplicated TRAIN rows")
    schedule_sha256 = digest(flat)
    selected: list[tuple[int, list[int]]] = []
    manifest: list[dict[str, Any]] = []
    for start in range(0, len(batches), ACCUMULATION):
        window = batches[start : start + ACCUMULATION]
        indices = [index for batch in window for source, index in batch]
        if any(source != "train" for batch in window for source, _ in batch):
            raise ValueError("Unexpected replay item in frozen control schedule")
        if len(indices) != ACCUMULATION:
            continue
        counts = {
            kind: sum(rows[index]["task_type"] == kind for index in indices)
            for kind in TASK_TYPES
        }
        if not all(counts.values()):
            continue
        ordinal = start // ACCUMULATION
        selected.append((ordinal, indices))
        manifest.append(
            {
                "optimizer_window": ordinal,
                "rows": len(indices),
                "native_tokens": sum(lengths[index] for index in indices),
                "type_counts": counts,
                "roster_sha256": digest(
                    [
                        {
                            "id": rows[index]["id"],
                            "input_sha256": rows[index]["input_sha256"],
                            "task_type": rows[index]["task_type"],
                            "tokens": lengths[index],
                        }
                        for index in indices
                    ]
                ),
            }
        )
        if len(selected) == WINDOW_COUNT:
            break
    if len(selected) != WINDOW_COUNT:
        raise ValueError("Frozen TRAIN has fewer than eight three-type windows")
    return selected, manifest, schedule_sha256


def _dot(left: list[Any], right: list[Any]) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for a, b in zip(left, right):
        total.add_(torch.dot(a.flatten(), b.flatten()).double())
    value = float(total.item())
    if not math.isfinite(value):
        raise ValueError("Nonfinite gradient dot product")
    return value


def measure_window(
    model: Any,
    items: list[dict[str, Any]],
    *,
    pad_id: int,
    device: Any,
) -> dict[str, Any]:
    """Measure ordinary and per-type backbone gradients, with zero updates."""
    import torch

    if not items or {item["task_type"] for item in items} != set(TASK_TYPES):
        raise ValueError("One window must contain all three native task types")
    if not model.training:
        raise ValueError("Use the control's training mode for gradient measurement")
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.grad is not None for p in all_parameters):
        raise ValueError("Need trainable backbone and clean parameter gradients")
    versions = [p._version for p in all_parameters]
    standard = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    typed = {
        kind: [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
        for kind in TASK_TYPES
    }
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for item in items:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in collate([item], pad_id).items()
        }
        with autocast:
            logits = model(**batch)
            terms = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )
            loss = terms["total"].sum() / len(items)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite ordinary control loss")
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        finite = torch.ones((), dtype=torch.bool, device=device)
        for position, gradient in enumerate(gradients):
            if gradient is None:
                continue
            finite.logical_and_(torch.isfinite(gradient).all())
            value = gradient.detach().float()
            standard[position].add_(value)
            typed[item["task_type"]][position].add_(value)
        if not bool(finite.item()):
            raise ValueError("Nonfinite backbone gradient")
    if [p._version for p in all_parameters] != versions or any(
        p.grad is not None for p in all_parameters
    ):
        raise ValueError("Read-only diagnostic changed model parameters or .grad")
    maximum_error_tensor = torch.zeros((), dtype=torch.float32, device=device)
    for position, ordinary in enumerate(standard):
        reconstructed = sum(typed[kind][position] for kind in TASK_TYPES)
        maximum_error_tensor = torch.maximum(
            maximum_error_tensor, (ordinary - reconstructed).abs().max()
        )
    maximum_error = float(maximum_error_tensor.item())
    if maximum_error > 1e-5:
        raise ValueError("Per-type gradients do not reconstruct ordinary control")
    norms = {
        kind: math.sqrt(max(0.0, _dot(typed[kind], typed[kind]))) for kind in TASK_TYPES
    }
    norm_control = math.sqrt(max(0.0, _dot(standard, standard)))
    pairs: dict[str, float | None] = {}
    for left, right in (("choice", "noul"), ("choice", "score"), ("noul", "score")):
        denominator = norms[left] * norms[right]
        pairs[f"{left}_{right}"] = (
            max(-1.0, min(1.0, _dot(typed[left], typed[right]) / denominator))
            if denominator > 1e-12
            else None
        )
    conflict = any(
        value is not None and value <= COSINE_THRESHOLD for value in pairs.values()
    )
    return {
        "ordinary_backbone_norm": norm_control,
        "task_backbone_norms": norms,
        "pairwise_cosines": pairs,
        "max_reconstruction_error": maximum_error,
        "choice_or_score_conflict": conflict,
    }


def validate_receipt(receipt: dict[str, Any]) -> None:
    """Reject unexpected fields so raw examples cannot enter the receipt."""
    if (
        set(receipt) - RECEIPT_FIELDS
        or receipt.get("schema_version") != SCHEMA
        or receipt.get("optimizer_constructed") is not False
        or receipt.get("optimizer_steps") != 0
        or not isinstance(receipt.get("created_utc"), str)
        or not re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00", receipt["created_utc"]
        )
    ):
        raise ValueError("Gradient receipt contains an invalid top-level field")
    status = receipt.get("status")
    if status == "HOLD_TECHNICAL":
        if set(receipt) != {
            "schema_version",
            "status",
            "created_utc",
            "failure_class",
            "optimizer_constructed",
            "optimizer_steps",
        } or not re.fullmatch(
            r"[A-Za-z]+(?:Error|Exception)", receipt["failure_class"]
        ):
            raise ValueError("Technical HOLD receipt contains extra details")
        return
    if status not in {
        "PLAN_ONLY",
        "PASS_CONFLICT_PRECHECK",
        "HOLD_NO_MEASURED_CONFLICT",
    }:
        raise ValueError("Unknown gradient receipt status")
    base_fields = {
        "schema_version",
        "status",
        "created_utc",
        "source_revision",
        "source_files_sha256",
        "code_sha256",
        "train_sha256",
        "train_rows",
        "train_native_tokens",
        "prompt_version",
        "seed",
        "microbatch",
        "accumulation",
        "planned_control_updates",
        "schedule_sha256",
        "windows",
        "conflict_cosine_threshold",
        "min_conflict_windows",
        "optimizer_constructed",
        "optimizer_steps",
    }
    measured_fields = {
        "device_precision",
        "measured_windows",
        "conflict_windows",
        "measurement_seconds",
        "measurement_gpu_hours",
    }
    if set(receipt) != base_fields | (
        measured_fields if status != "PLAN_ONLY" else set()
    ):
        raise ValueError("Gradient receipt has unexpected or missing fields")
    if (
        receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_files_sha256") != SOURCE_FILES
        or receipt.get("train_sha256") != TRAIN_SHA256
        or receipt.get("train_rows") != TRAIN_COUNT
        or receipt.get("train_native_tokens") != TRAIN_TOKENS
        or receipt.get("prompt_version") != PROMPT_VERSION
        or receipt.get("seed") != SEED
        or receipt.get("microbatch") != 1
        or receipt.get("accumulation") != ACCUMULATION
        or receipt.get("planned_control_updates") != 466
        or receipt.get("conflict_cosine_threshold") != COSINE_THRESHOLD
        or receipt.get("min_conflict_windows") != MIN_CONFLICT_WINDOWS
        or not isinstance(receipt.get("windows"), list)
        or len(receipt["windows"]) != WINDOW_COUNT
        or not isinstance(receipt.get("schedule_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt["schedule_sha256"])
    ):
        raise ValueError("Gradient receipt differs from the frozen control")
    expected_code = {
        "gradient_conflict_preflight.py",
        "data.py",
        "decision_model.py",
        "loss.py",
        "plan.py",
    }
    if (
        not isinstance(receipt.get("code_sha256"), dict)
        or set(receipt["code_sha256"]) != expected_code
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in receipt["code_sha256"].values()
        )
    ):
        raise ValueError("Gradient receipt needs source code hashes")
    for window in receipt["windows"]:
        if (
            not isinstance(window, dict)
            or set(window) - WINDOW_FIELDS
            or type(window.get("optimizer_window")) is not int
            or window["optimizer_window"] < 0
            or window.get("rows") != ACCUMULATION
            or not isinstance(window.get("native_tokens"), int)
            or window["native_tokens"] < 1
            or not isinstance(window.get("type_counts"), dict)
            or set(window.get("type_counts", {})) != set(TASK_TYPES)
            or any(
                type(value) is not int or value < 1
                for value in window["type_counts"].values()
            )
            or sum(window["type_counts"].values()) != ACCUMULATION
            or not isinstance(window.get("roster_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", window["roster_sha256"])
        ):
            raise ValueError("Gradient window receipt contains raw or invalid data")
        if status == "PLAN_ONLY" and set(window) != {
            "optimizer_window",
            "rows",
            "native_tokens",
            "type_counts",
            "roster_sha256",
        }:
            raise ValueError("Plan-only receipt contains measured data")
        if status != "PLAN_ONLY" and set(window) != WINDOW_FIELDS:
            raise ValueError("Measured window receipt is incomplete")
        if status != "PLAN_ONLY":
            pairs = window["pairwise_cosines"]
            norms = window["task_backbone_norms"]
            if (
                not isinstance(norms, dict)
                or set(norms) != set(TASK_TYPES)
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    for value in norms.values()
                )
                or not isinstance(pairs, dict)
                or set(pairs) != {"choice_noul", "choice_score", "noul_score"}
                or any(
                    value is not None
                    and (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not -1 <= value <= 1
                    )
                    for value in pairs.values()
                )
                or type(window["choice_or_score_conflict"]) is not bool
                or any(
                    type(window[key]) not in (int, float)
                    or not math.isfinite(window[key])
                    or window[key] < 0
                    for key in ("ordinary_backbone_norm", "max_reconstruction_error")
                )
                or window["max_reconstruction_error"] > 1e-5
            ):
                raise ValueError("Measured window statistics are invalid")
    ordinals = [window["optimizer_window"] for window in receipt["windows"]]
    if ordinals != sorted(set(ordinals)):
        raise ValueError("Frozen optimizer windows are not unique and ordered")
    if status != "PLAN_ONLY" and (
        receipt.get("measured_windows") != WINDOW_COUNT
        or type(receipt.get("conflict_windows")) is not int
        or not 0 <= receipt["conflict_windows"] <= WINDOW_COUNT
        or receipt.get("device_precision")
        != "BF16 backbone autocast; FP32 weights, head and loss"
        or any(
            type(receipt.get(key)) not in (int, float)
            or not math.isfinite(receipt[key])
            or receipt[key] < 0
            for key in ("measurement_seconds", "measurement_gpu_hours")
        )
    ):
        raise ValueError("Measured receipt lacks complete frozen windows")
    if status != "PLAN_ONLY" and (
        receipt["conflict_windows"]
        != sum(window["choice_or_score_conflict"] for window in receipt["windows"])
        or (status == "PASS_CONFLICT_PRECHECK")
        != (receipt["conflict_windows"] >= MIN_CONFLICT_WINDOWS)
    ):
        raise ValueError("Measured conflict status disagrees with frozen threshold")


def write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    """Write only aggregate, finite, content-free diagnostics atomically."""
    validate_receipt(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    pending.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    train = Path(args.train)
    source = Path(args.model_path)
    if file_sha256(train) != TRAIN_SHA256:
        raise ValueError("Frozen TRAIN SHA-256 mismatch")
    source_files = {name: file_sha256(source / name) for name in SOURCE_FILES}
    if source_files != SOURCE_FILES:
        raise ValueError("Official Qwen source files differ from frozen control")
    rows = load_partition(train, "train")
    if len(rows) != TRAIN_COUNT:
        raise ValueError("Frozen TRAIN row count mismatch")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    encoded = [encode(row, tokenizer, MAX_LENGTH) for row in rows]
    lengths = [len(item["ids"]) for item in encoded]
    if sum(lengths) != TRAIN_TOKENS:
        raise ValueError("Frozen tokenizer exposure differs from control")
    selected, manifest, schedule_sha256 = select_windows(rows, lengths)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PLAN_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": SOURCE_REVISION,
        "source_files_sha256": source_files,
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "gradient_conflict_preflight.py",
                "data.py",
                "decision_model.py",
                "loss.py",
                "plan.py",
            )
        },
        "train_sha256": TRAIN_SHA256,
        "train_rows": TRAIN_COUNT,
        "train_native_tokens": TRAIN_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "microbatch": 1,
        "accumulation": ACCUMULATION,
        "planned_control_updates": 466,
        "schedule_sha256": schedule_sha256,
        "windows": manifest,
        "conflict_cosine_threshold": COSINE_THRESHOLD,
        "min_conflict_windows": MIN_CONFLICT_WINDOWS,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    if not args.measure:
        return receipt
    if (
        args.device != "cuda"
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Measured control gradients require one BF16 CUDA/ROCm device")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    model, model_tokenizer = DecisionModel.from_base(
        source, SOURCE_REVISION, 256, source_stage="base", head_variant="shared"
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Reloaded official tokenizer vocabulary differs")
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Official tokenizer needs a pad or EOS token")
    model = model.float().to(torch.device("cuda:0"))
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.train()
    started = time.monotonic()
    measured = []
    for _, indices in selected:
        if time.monotonic() - started > MAX_MEASURE_SECONDS:
            raise TimeoutError("Frozen gradient diagnostic exceeded six minutes")
        result = measure_window(
            model,
            [encoded[index] for index in indices],
            pad_id=pad_id,
            device=torch.device("cuda:0"),
        )
        measured.append(result)
    elapsed = time.monotonic() - started
    if elapsed > MAX_MEASURE_SECONDS:
        raise TimeoutError("Frozen gradient diagnostic exceeded six minutes")
    conflicts = sum(result["choice_or_score_conflict"] for result in measured)
    for item, result in zip(receipt["windows"], measured):
        item.update(result)
    receipt.update(
        {
            "status": (
                "PASS_CONFLICT_PRECHECK"
                if conflicts >= MIN_CONFLICT_WINDOWS
                else "HOLD_NO_MEASURED_CONFLICT"
            ),
            "device_precision": "BF16 backbone autocast; FP32 weights, head and loss",
            "measured_windows": len(measured),
            "conflict_windows": conflicts,
            "measurement_seconds": elapsed,
            "measurement_gpu_hours": elapsed / 3600,
        }
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Private frozen TRAIN JSONL")
    parser.add_argument(
        "--model-path", required=True, help="Pinned local official Qwen source"
    )
    parser.add_argument(
        "--output", required=True, help="Private aggregate JSON receipt"
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Explicit read-only BF16 GPU gradient measurement",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.measure != (args.device == "cuda"):
        parser.error(
            "Use the default CPU plan or explicitly pair --measure with --device cuda"
        )
    output = Path(args.output)
    if output.exists() or output.with_name(output.name + ".pending").exists():
        parser.error("Receipt output already exists; preserve the prior result")
    try:
        receipt = run(args)
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        write_receipt(
            output,
            {
                "schema_version": SCHEMA,
                "status": "HOLD_TECHNICAL",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "failure_class": type(exc).__name__,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
            },
        )
        raise
    write_receipt(output, receipt)


if __name__ == "__main__":
    main()
