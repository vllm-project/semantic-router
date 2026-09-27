"""Pinned three-type, long-TRAIN-row Gemma numeric gate, never an evaluation.

``prepare-lock`` and ``meta`` are CPU-only. ``run`` requires a separately
approved one-GPU private launcher and may make at most three TRAIN updates.
No SELECT, CAL, DEV, JevArena or JevBench path is accepted by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from scripts.preflight_gemma4_native import (
    SOURCE_ID,
    SOURCE_REVISION,
    inspect,
    private_output,
    sha256_file,
    verify_loaded_state,
)
from scripts.preflight_gemma4_one_step import (
    TRAIN_SHA256,
    pinned_train_row,
    read_private_lock,
)
from training.model.data import load_partition
from training.model.decision_model import CandidateHead, encode
from training.model.gemma4 import (
    GEMMA_PROMPT_VERSION,
    assert_zero_lora_b,
    attach_text_lora,
    encode_gemma,
)
from training.model.loss import LOSS_VERSION, per_example_loss

PROBE_VERSION = "gemma4-three-type-long-train-reload-v1"
LOCK_VERSION = "gemma4-three-type-long-train-prereg-v1"
QWEN_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)
TASK_TYPES = ("choice", "noul", "score")
PINNED_LONG_LENGTHS = {"choice": 4090, "noul": 4076, "score": 4044}
MAX_LENGTH = 4096
GPU_ORDINAL = "3"
SEED = 20260926
LORA_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
BRIER_WEIGHT = 0.5
GRAD_CLIP = 1.0
MAX_GRAD_NORM = 1e6
MAX_RELOAD_LOGIT_DRIFT = 1e-3
MAX_WALL_SECONDS = 2700


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = {
        "long_train_probe": Path(__file__),
        "source_probe": root / "scripts/preflight_gemma4_native.py",
        "one_step_row_validator": root / "scripts/preflight_gemma4_one_step.py",
        "gemma_adapter": root / "training/model/gemma4.py",
        "shared_prompt_and_head": root / "training/model/decision_model.py",
        "data_canonicalizer": root / "training/model/data.py",
        "loss": root / "training/model/loss.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def select_long_rows(
    train_file: Path, qwen_tokenizer: Any, gemma_tokenizer: Any
) -> list[dict[str, Any]]:
    """Pick the unique longest Gemma-admitted row per type in both-model cohort."""
    if sha256_file(train_file) != TRAIN_SHA256:
        raise ValueError("TRAIN file differs from frozen rights-clean v2")
    rows = load_partition(train_file, "train")
    lines = train_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != len(rows):
        raise ValueError("TRAIN row and raw-line counts differ")
    winners: dict[str, tuple[int, str, dict[str, Any]]] = {}
    for row, line in zip(rows, lines, strict=True):
        qwen = encode(row, qwen_tokenizer, 10**8)
        gemma = encode_gemma(row, gemma_tokenizer, 10**8)
        if len(qwen["ids"]) > MAX_LENGTH or len(gemma["ids"]) > MAX_LENGTH:
            continue
        specification = {
            "id": row["id"],
            "group_id": row["group_id"],
            "row_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
            "prompt_sha256": gemma["prompt_sha256"],
            "token_ids_sha256": gemma["token_ids_sha256"],
            "token_count": len(gemma["ids"]),
            "task_type": row["task_type"],
            "family": row["family"],
            "language": row["language"],
            "label_index": row["label"],
            "option_count": len(row["options"]),
        }
        # A stable ID breaks equal-length ties; the selection is not based on
        # gold correctness or on model outputs.
        candidate = (len(gemma["ids"]), row["id"], specification)
        previous = winners.get(row["task_type"])
        if previous is None or candidate[:2] > previous[:2]:
            winners[row["task_type"]] = candidate
    if set(winners) != set(TASK_TYPES):
        raise ValueError("TRAIN common cohort lacks one of the three task types")
    selected = [winners[kind][2] for kind in TASK_TYPES]
    if {
        item["task_type"]: item["token_count"] for item in selected
    } != PINNED_LONG_LENGTHS:
        raise ValueError("Pinned longest TRAIN lengths changed")
    return selected


def expected_lock(
    source: dict[str, Any],
    selected: list[dict[str, Any]],
    runner_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_config_sha256": source["config_sha256"],
        "gemma_tokenizer_sha256": source["tokenizer_json_sha256"],
        "qwen_tokenizer_sha256": QWEN_TOKENIZER_SHA256,
        "source_shard_sha256": source["weights"]["shard_sha256"],
        "code_sha256": code_hashes(),
        "runner_sha256": runner_sha256,
        "train_file_sha256": TRAIN_SHA256,
        "train_rows": selected,
        "gpu_ordinal": GPU_ORDINAL,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "head_dim": 256,
        "lora_lr": LORA_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "objective": "ce_brier",
        "brier_weight": BRIER_WEIGHT,
        "gradient_checkpointing": True,
        "grad_clip": GRAD_CLIP,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_reload_logit_drift": MAX_RELOAD_LOGIT_DRIFT,
        "optimizer_updates": len(TASK_TYPES),
        "max_wall_seconds": MAX_WALL_SECONDS,
        "select_cal_formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def checked_inputs(
    snapshot: Path,
    train_file: Path,
    qwen_path: Path,
    runner: Path,
    lock_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """No-device full-file and two-tokenizer admission, with lock verification."""
    from transformers import AutoTokenizer

    source = inspect(snapshot)
    if sha256_file(qwen_path / "tokenizer.json") != QWEN_TOKENIZER_SHA256:
        raise ValueError("Qwen tokenizer differs from the pinned official source")
    qwen = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    gemma = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    selected = select_long_rows(train_file, qwen, gemma)
    lock = None
    if lock_path is not None:
        lock = read_private_lock(lock_path)
        expected = expected_lock(source, selected, sha256_file(runner))
        if lock != expected:
            raise ValueError("Three-type long-input run differs from private lock")
        # Also exercise the original independent row verifier on each locked
        # line, not only the deterministic longest-row selector.
        for spec in selected:
            encoded, row_hash = pinned_train_row(train_file, gemma, spec)
            if (
                row_hash != spec["row_sha256"]
                or len(encoded["ids"]) != spec["token_count"]
            ):
                raise ValueError("Locked long TRAIN row changed")
    return source, selected, lock


def dynamic_logits(backbone: Any, head: Any, encoded: dict[str, Any]) -> Any:
    import torch

    device = next(head.parameters()).device
    ids = torch.tensor([encoded["ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    hidden = backbone(
        input_ids=ids, attention_mask=mask, use_cache=False
    ).last_hidden_state
    choices = hidden[:, encoded["candidate_positions"], :]
    query = hidden[:, encoded["query_position"], :]
    logits = head(choices, query)
    if (
        tuple(logits.shape) != (1, len(encoded["keys"]))
        or not torch.isfinite(logits).all()
    ):
        raise RuntimeError("Gemma long-input TRAIN logits have wrong shape or values")
    return logits


def run(
    snapshot: Path,
    train_file: Path,
    qwen_path: Path,
    runner: Path,
    lock_path: Path,
    package: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Three fixed TRAIN updates, followed by exact package and logit reload."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    source, specs, lock = checked_inputs(
        snapshot, train_file, qwen_path, runner, lock_path
    )
    assert lock is not None
    if (
        os.getenv("ROCR_VISIBLE_DEVICES") != GPU_ORDINAL
        or os.getenv("HIP_VISIBLE_DEVICES") is not None
        or device != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Long-input run needs exactly the locked BF16 ROCm GPU")
    if not package.is_absolute() or package.exists() or package.is_symlink():
        raise ValueError("Long-input package must be a new private absolute path")
    parent = package.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
    ):
        raise ValueError("Long-input package parent must be owner-held mode 0700")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded_rows = [pinned_train_row(train_file, tokenizer, spec)[0] for spec in specs]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    full, info = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Gemma long-input source load omitted official weights")
    loaded = verify_loaded_state(full, snapshot, source["weights"])
    full.requires_grad_(False)
    if any(p.requires_grad for p in full.model.vision_tower.parameters()) or any(
        p.requires_grad for p in full.model.embed_vision.parameters()
    ):
        raise RuntimeError("Gemma vision parameters became trainable")
    base = full.model.language_model
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    adapter, plan = attach_text_lora(base, rank=8, alpha=16, dropout=0.05)
    assert_zero_lora_b(adapter, 60)
    head = CandidateHead(2816, 256).to(device)
    lora_named = {name: p for name, p in adapter.named_parameters() if p.requires_grad}
    head_named = dict(head.named_parameters())
    if (
        len(lora_named) != 120
        or sum(p.numel() for p in lora_named.values()) != 3_645_440
        or sum(p.numel() for p in head_named.values()) != 2_895_360
        or any(not p.requires_grad for p in head_named.values())
    ):
        raise RuntimeError("Gemma long-input trainable set differs from q/o plan")
    trainables = [*lora_named.values(), *head_named.values()]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(lora_named.values()), "lr": LORA_LR},
            {"params": list(head_named.values()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {
        id(p) for p in trainables
    }:
        raise RuntimeError("Gemma long-input optimizer contains wrong parameters")
    step_receipts = []
    adapter.train()
    head.train()
    for encoded in encoded_rows:
        before_b = {
            name: p.detach().float().cpu().clone()
            for name, p in lora_named.items()
            if ".lora_B." in name
        }
        before_head = {name: p.detach().cpu().clone() for name, p in head_named.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = dynamic_logits(adapter, head, encoded)
            target = torch.tensor([encoded["label"]], dtype=torch.long, device=device)
            candidate_mask = torch.ones_like(logits, dtype=torch.bool)
            loss = per_example_loss(
                logits,
                target,
                candidate_mask,
                objective="ce_brier",
                brier_weight=BRIER_WEIGHT,
            )["total"].mean()
        if not torch.isfinite(loss) or not 0 < loss.item() < 1e6:
            raise RuntimeError("Gemma long-input TRAIN loss failed finite bound")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainables):
            raise RuntimeError("Gemma long-input gradient missing or nonfinite")
        b_nonzero = sum(
            torch.count_nonzero(p.grad).item() > 0
            for name, p in lora_named.items()
            if ".lora_B." in name
        )
        if b_nonzero != 60:
            raise RuntimeError("Gemma long-input omitted a q/o LoRA B gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainables, GRAD_CLIP)
        if not torch.isfinite(norm) or not 0 < norm.item() < MAX_GRAD_NORM:
            raise RuntimeError("Gemma long-input gradient norm failed finite bound")
        optimizer.step()
        torch.cuda.synchronize()
        changed_b = sum(
            not torch.equal(p.detach().float().cpu(), before_b[name])
            for name, p in lora_named.items()
            if name in before_b
        )
        changed_head = sum(
            not torch.equal(p.detach().cpu(), before_head[name])
            for name, p in head_named.items()
        )
        if changed_b != 60 or changed_head < 1:
            raise RuntimeError("Gemma long-input update omitted adapter or head")
        step_receipts.append(
            {
                "task_type": encoded["task_type"],
                "train_row_sha256": next(
                    spec["row_sha256"]
                    for spec in specs
                    if spec["task_type"] == encoded["task_type"]
                ),
                "token_count": len(encoded["ids"]),
                "option_count": len(encoded["keys"]),
                "train_loss": float(loss.item()),
                "unclipped_gradient_norm": float(norm.item()),
                "lora_b_nonzero_gradient_count": b_nonzero,
                "lora_b_changed_count": changed_b,
                "head_changed_tensor_count": changed_head,
            }
        )

    adapter.eval()
    head.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        saved_logits = [
            dynamic_logits(adapter, head, row).float().cpu() for row in encoded_rows
        ]
    adapter_state = {name: p.detach().cpu().clone() for name, p in lora_named.items()}
    head_state = {
        name: p.detach().cpu().clone() for name, p in head.state_dict().items()
    }
    package.mkdir(mode=0o700, parents=False, exist_ok=False)
    adapter.save_pretrained(package / "adapter", safe_serialization=True)
    save_file(
        {name: value.float().contiguous() for name, value in head_state.items()},
        str(package / "decision_head.safetensors"),
    )
    adapter_file = package / "adapter/adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("Gemma long-input package has no adapter")
    package_hashes = {
        "adapter_safetensors_sha256": sha256_file(adapter_file),
        "adapter_config_sha256": sha256_file(package / "adapter/adapter_config.json"),
        "decision_head_sha256": sha256_file(package / "decision_head.safetensors"),
    }
    del optimizer, adapter, head, full, base, logits, loss, target
    del lora_named, head_named, trainables, before_b, before_head, norm
    gc.collect()
    torch.cuda.empty_cache()

    second_source = inspect(snapshot)
    if second_source != source:
        raise RuntimeError("Official Gemma source changed between independent loads")
    second, second_info = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(
        second_info.get(key)
        for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("Reloaded Gemma source omitted official weights")
    if verify_loaded_state(second, snapshot, source["weights"]) != loaded:
        raise RuntimeError("Gemma source parameter identity changed on reload")
    second.requires_grad_(False)
    second.model.language_model.config.use_cache = False
    reloaded_adapter = PeftModel.from_pretrained(
        second.model.language_model, package / "adapter", is_trainable=False
    )
    named_reload = dict(reloaded_adapter.named_parameters())
    reloaded_lora = {
        name for name in named_reload if ".lora_A." in name or ".lora_B." in name
    }
    if reloaded_lora != set(adapter_state) or any(
        not torch.equal(value, named_reload[name].detach().cpu())
        for name, value in adapter_state.items()
    ):
        raise RuntimeError("Gemma long-input LoRA package changed across reload")
    reloaded_head = CandidateHead(2816, 256).to(device)
    reloaded_head.load_state_dict(
        load_file(str(package / "decision_head.safetensors")), strict=True
    )
    if any(
        not torch.equal(value.float(), reloaded_head.state_dict()[name].detach().cpu())
        for name, value in head_state.items()
    ):
        raise RuntimeError("Gemma long-input Decision head changed across reload")
    reloaded_adapter.eval()
    reloaded_head.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        replay_logits = [
            dynamic_logits(reloaded_adapter, reloaded_head, row).float().cpu()
            for row in encoded_rows
        ]
    drifts = [
        (first - second).abs().max().item()
        for first, second in zip(saved_logits, replay_logits, strict=True)
    ]
    if any(
        not math.isfinite(drift) or drift > MAX_RELOAD_LOGIT_DRIFT for drift in drifts
    ):
        raise RuntimeError("Gemma long-input native logits changed across reload")
    torch.cuda.synchronize()
    return {
        "probe_version": PROBE_VERSION,
        "status": "three_type_long_train_reload_passed",
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source": source,
        "loaded": loaded,
        "prompt_version": GEMMA_PROMPT_VERSION,
        "loss_version": LOSS_VERSION,
        "train_file_sha256": TRAIN_SHA256,
        "train_steps": step_receipts,
        "adapter_plan": plan,
        "package_sha256": package_hashes,
        "reload_max_logit_abs_drift_by_type": dict(
            zip(TASK_TYPES, drifts, strict=True)
        ),
        "reload_limit": MAX_RELOAD_LOGIT_DRIFT,
        "code_sha256": code_hashes(),
        "lock_sha256": sha256_file(lock_path),
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": len(step_receipts),
        "select_cal_formal_labels_read": False,
        "model_quality_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-lock", "meta", "run"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare-lock":
        if args.lock is not None or args.package is not None:
            parser.error("prepare-lock cannot use lock/package")
    elif args.lock is None or (args.mode == "run") != (args.package is not None):
        parser.error("meta/run require --lock; only run requires --package")
    os.umask(0o077)
    snapshot = args.snapshot.resolve(strict=True)
    train_file = args.train_file.resolve(strict=True)
    qwen_path = args.qwen_tokenizer.resolve(strict=True)
    runner = args.runner.resolve(strict=True)
    if args.mode == "prepare-lock":
        source, selected, _ = checked_inputs(
            snapshot, train_file, qwen_path, runner, None
        )
        result = expected_lock(source, selected, sha256_file(runner))
    elif args.mode == "meta":
        source, selected, lock = checked_inputs(
            snapshot, train_file, qwen_path, runner, args.lock
        )
        assert lock is not None
        result = {
            "probe_version": PROBE_VERSION,
            "source": source,
            "train_file_sha256": TRAIN_SHA256,
            "long_train_rows": [
                {
                    "task_type": item["task_type"],
                    "token_count": item["token_count"],
                    "option_count": item["option_count"],
                    "row_sha256": item["row_sha256"],
                }
                for item in selected
            ],
            "prompt_version": GEMMA_PROMPT_VERSION,
            "loss_version": LOSS_VERSION,
            "code_sha256": code_hashes(),
            "runner_sha256": sha256_file(runner),
            "lock_sha256": sha256_file(args.lock),
            "gpu_used": False,
            "optimizer_updates": 0,
            "select_cal_formal_labels_read": False,
            "model_quality_evaluated": False,
        }
    else:
        result = run(
            snapshot,
            train_file,
            qwen_path,
            runner,
            args.lock,
            args.package,
            device=args.device,
        )
    private_output(args.output, result)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "source_revision": SOURCE_REVISION,
                "receipt_sha256": sha256_file(args.output),
                "model_quality_evaluated": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
