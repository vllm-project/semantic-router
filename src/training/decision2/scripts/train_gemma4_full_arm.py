"""Locked official-Gemma Decision development arm; never a benchmark runner.

``prepare-lock`` and ``cpu-audit`` need no GPU. ``run`` needs a separately
approved, offline single-GPU launcher and reads only TRAIN/SELECT. CAL, DEV,
formal, public or external benchmark
path is accepted. All output stays in a new private directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import stat
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.preflight_gemma4_native import (
    SOURCE_ID,
    SOURCE_REVISION,
    inspect,
    sha256_file,
    verify_loaded_state,
)
from training.model.gemma4_full_plan import (
    ACCUMULATION,
    ADVANCE_MACRO,
    CHECKPOINT_STEPS,
    MAX_LENGTH,
    MAX_WALL_SECONDS,
    PLANNED_UPDATES,
    SEED,
    SELECT_COUNT,
    SELECT_STEPS,
    admitted_roster,
    checkpoint_choice,
    digest,
    schedule_receipt,
    select_stop_reason,
    select_summary,
)

LOCK_VERSION = "decision2-gemma4-full-development-arm-v1"
TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
SELECT_SHA256 = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
QWEN_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)
IDENTITY_RECEIPT_SHA256 = (
    "80ed1da6f56dd68382bc1c50d3dfa99a41720fb264fb083a357b5d9e9be0b4e1"
)
LONG_GATE_RECEIPT_SHA256 = (
    "cfaa871d3c424209c4bad2bae9bcba3089df1d80288f6702f50a46293baab47e"
)
GPU_ORDINAL = "3"
IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
LORA_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
BRIER_WEIGHT = 0.5
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1e6


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = {
        "full_runner": Path(__file__),
        "full_plan": root / "training/model/gemma4_full_plan.py",
        "source_probe": root / "scripts/preflight_gemma4_native.py",
        "gemma_adapter": root / "training/model/gemma4.py",
        "shared_head_prompt": root / "training/model/decision_model.py",
        "data": root / "training/model/data.py",
        "loss": root / "training/model/loss.py",
        "schedule": root / "training/model/plan.py",
        "long_gate": root / "scripts/preflight_gemma4_long_train.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _owner_private(path: Path, *, directory: bool) -> None:
    item = path.resolve(strict=True)
    mode = stat.S_IMODE(item.stat().st_mode)
    if item.stat().st_uid != os.getuid() or mode != (0o700 if directory else 0o600):
        raise ValueError("Private arm lock/output permissions or ownership differ")


def _check_prior_receipts(identity: Path, long_gate: Path) -> None:
    for path, sha, status in (
        (identity, IDENTITY_RECEIPT_SHA256, "source_to_fresh_lora_identity_passed"),
        (long_gate, LONG_GATE_RECEIPT_SHA256, "three_type_long_train_reload_passed"),
    ):
        if sha256_file(path) != sha or json.loads(path.read_text())["status"] != status:
            raise ValueError("Prerequisite Gemma identity/long gate differs")


def _full_admission(
    snapshot: Path, train: Path, select: Path, qwen_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """No device: strict source/data hashes, same-row admission and schedule."""
    from training.model.data import check_partition_isolation, load_partition
    from training.model.decision_model import encode
    from training.model.gemma4 import encode_gemma
    from transformers import AutoTokenizer

    for path, expected in ((train, TRAIN_SHA256), (select, SELECT_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError("Frozen TRAIN/SELECT file differs")
    if sha256_file(qwen_path / "tokenizer.json") != QWEN_TOKENIZER_SHA256:
        raise ValueError("Official Qwen cohort tokenizer differs")
    source = inspect(snapshot)
    qwen = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    gemma = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    check_partition_isolation({"train": train_rows, "select": select_rows})
    if len(select_rows) != SELECT_COUNT:
        raise ValueError("SELECT count differs from frozen rights-clean v2")
    q_lengths, g_lengths = [], []
    for row in train_rows:
        q_lengths.append(len(encode(row, qwen, 10**8)["ids"]))
        g_lengths.append(len(encode_gemma(row, gemma, 10**8)["ids"]))
    indices, cohort = admitted_roster(train_rows, q_lengths, g_lengths)
    admitted = [train_rows[index] for index in indices]
    lengths = [g_lengths[index] for index in indices]
    select_lengths = [
        len(encode_gemma(row, gemma, 10**8)["ids"]) for row in select_rows
    ]
    if any(not 0 < length <= MAX_LENGTH for length in select_lengths):
        raise ValueError(
            "SELECT contains an overlength Gemma input; no truncation allowed"
        )
    schedule = schedule_receipt(lengths)
    admission = {
        "cohort": cohort,
        "schedule": schedule,
        "select_count": len(select_rows),
        "select_type_counts": dict(
            sorted(Counter(row["task_type"] for row in select_rows).items())
        ),
        "select_ordered_ids_sha256": digest([row["id"] for row in select_rows]),
        "select_token_lengths_sha256": digest(select_lengths),
        "select_max_tokens": max(select_lengths),
        "cal_never_opened": True,
    }
    return source, admitted, select_rows, admission


def expected_lock(
    source: dict[str, Any], admission: dict[str, Any], launcher_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_config_sha256": source["config_sha256"],
        "source_tokenizer_sha256": source["tokenizer_json_sha256"],
        "source_shard_sha256": source["weights"]["shard_sha256"],
        "qwen_tokenizer_sha256": QWEN_TOKENIZER_SHA256,
        "train_sha256": TRAIN_SHA256,
        "select_sha256": SELECT_SHA256,
        "cal_never_opened": True,
        "identity_receipt_sha256": IDENTITY_RECEIPT_SHA256,
        "long_gate_receipt_sha256": LONG_GATE_RECEIPT_SHA256,
        "code_sha256": code_hashes(),
        "launcher_sha256": launcher_sha,
        "image_id": IMAGE_ID,
        "admission": admission,
        "gpu_ordinal": GPU_ORDINAL,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "lora": {
            "target": "all-text-q/o-only",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
        "head_dim": 256,
        "objective": "ce_brier",
        "brier_weight": BRIER_WEIGHT,
        "lora_lr": LORA_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_gradient_norm": MAX_GRAD_NORM,
        "clip_norm": 1.0,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "select_steps": list(SELECT_STEPS),
        "checkpoint_choice": "family macro accuracy; lower Brier; earliest step",
        "futility_at_256": "best SELECT family macro <0.70",
        "score_collapse_at_256": "<3 predicted levels or >95% one level",
        "advance_select_family_macro": ADVANCE_MACRO,
        "no_public_or_formal_labels": True,
    }


def checked_lock(
    snapshot: Path,
    train: Path,
    select: Path,
    qwen_path: Path,
    identity: Path,
    long_gate: Path,
    launcher: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _owner_private(lock_path.parent, directory=True)
    _owner_private(lock_path, directory=False)
    _check_prior_receipts(identity, long_gate)
    source, admitted, select_rows, admission = _full_admission(
        snapshot, train, select, qwen_path
    )
    expected = expected_lock(source, admission, sha256_file(launcher))
    if json.loads(lock_path.read_text()) != expected:
        raise ValueError("Prospective Gemma full arm differs from private lock")
    return source, admitted, select_rows, expected


def learning_factor(step: int) -> float:
    warmup = max(1, round(PLANNED_UPDATES * WARMUP_RATIO))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, PLANNED_UPDATES - warmup))
    return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2


def _write_new_json(path: Path, data: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "w") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _select_readout(
    adapter: Any, head: Any, encoded: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from scripts.preflight_gemma4_long_train import dynamic_logits

    adapter.eval()
    head.eval()
    records = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for item in encoded:
            logits = dynamic_logits(adapter, head, item).float()[0]
            p = torch.softmax(logits, dim=-1).cpu().tolist()
            best = max(p)
            matches = [i for i, value in enumerate(p) if abs(value - best) <= 1e-8]
            records.append(
                {
                    "id": item["id"],
                    "task_type": item["task_type"],
                    "family": item["family"],
                    "label": item["label"],
                    "predicted": matches[0] if len(matches) == 1 else -1,
                    "probabilities": p,
                    "prompt_sha256": item["prompt_sha256"],
                }
            )
    summary = select_summary(records)
    adapter.train()
    head.train()
    return records, summary


def _save_checkpoint(
    output: Path,
    adapter: Any,
    head: Any,
    optimizer: Any,
    step: int,
    metrics: dict[str, Any] | None,
    lock_sha: str,
    cursor: int,
) -> dict[str, str]:
    import torch
    from safetensors.torch import load_file, save_file

    name = f"checkpoint-{step:07d}"
    pending = output / f"{name}.pending"
    destination = output / name
    if pending.exists() or destination.exists():
        raise ValueError("Gemma checkpoint already exists")
    pending.mkdir(mode=0o700)
    adapter.save_pretrained(pending / "adapter", safe_serialization=True)
    head_state = {
        name: value.detach().float().cpu().contiguous()
        for name, value in head.state_dict().items()
    }
    save_file(head_state, str(pending / "decision_head.safetensors"))
    if any(
        not torch.equal(
            value, load_file(str(pending / "decision_head.safetensors"))[name]
        )
        for name, value in head_state.items()
    ):
        raise RuntimeError("Saved Gemma Decision head differs before checkpoint commit")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "cursor": cursor,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "lock_sha256": lock_sha,
        },
        pending / "trainer_state.pt",
    )
    hashes = {
        name: sha256_file(pending / name)
        for name in (
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
            "decision_head.safetensors",
            "trainer_state.pt",
        )
    }
    _write_new_json(
        pending / "CHECKPOINT.json",
        {
            "schema_version": "decision2-gemma4-development-checkpoint-v1",
            "step": step,
            "cursor": cursor,
            "lock_sha256": lock_sha,
            "select_metrics": metrics,
            "files_sha256": hashes,
        },
    )
    os.replace(pending, destination)
    return hashes


def _run(
    snapshot: Path,
    train_file: Path,
    select_file: Path,
    qwen_path: Path,
    identity: Path,
    long_gate: Path,
    launcher: Path,
    lock_path: Path,
    output: Path,
) -> None:
    import torch
    from scripts.preflight_gemma4_long_train import dynamic_logits
    from training.model.decision_model import CandidateHead
    from training.model.gemma4 import assert_zero_lora_b, attach_text_lora, encode_gemma
    from training.model.loss import per_example_loss
    from training.model.plan import epoch_batches
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    started = time.monotonic()
    source, train_rows, select_rows, lock = checked_lock(
        snapshot,
        train_file,
        select_file,
        qwen_path,
        identity,
        long_gate,
        launcher,
        lock_path,
    )
    if (
        os.getenv("ROCR_VISIBLE_DEVICES") != GPU_ORDINAL
        or os.getenv("HIP_VISIBLE_DEVICES") is not None
        or os.getenv("DECISION2_IMAGE_ID") != IMAGE_ID
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Gemma full arm requires exact approved ROCm device/image")
    if output.exists() or output.is_symlink():
        raise ValueError("Gemma full arm output must be new; no implicit resume")
    _owner_private(output.parent, directory=True)
    output.mkdir(mode=0o700)
    lock_sha = sha256_file(lock_path)
    _write_new_json(
        output / "RUN.json",
        {
            "status": "started",
            "lock_sha256": lock_sha,
            "source_revision": SOURCE_REVISION,
            "train_count": len(train_rows),
            "select_count": len(select_rows),
            "planned_updates": PLANNED_UPDATES,
            "formal_labels_read": False,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    train_items = [encode_gemma(row, tokenizer, MAX_LENGTH) for row in train_rows]
    select_items = [encode_gemma(row, tokenizer, MAX_LENGTH) for row in select_rows]
    lengths = [len(item["ids"]) for item in train_items]
    if schedule_receipt(lengths) != lock["admission"]["schedule"]:
        raise RuntimeError("GPU run schedule differs from CPU lock")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    full, loading = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(
        loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("Official Gemma source load omitted weights")
    loaded = verify_loaded_state(full, snapshot, source["weights"])
    full.requires_grad_(False)
    text = full.model.language_model
    text.config.use_cache = False
    text.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    adapter, plan = attach_text_lora(text, rank=8, alpha=16, dropout=0.05)
    assert_zero_lora_b(adapter, 60)
    head = CandidateHead(2816, 256).to("cuda:0")
    lora_params = [p for p in adapter.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    if (
        loaded["loaded_text_parameters"] != 25_233_141_760
        or plan["combined_trainable_parameters"] != 6_540_800
        or sum(p.numel() for p in [*lora_params, *head_params]) != 6_540_800
        or any(p.requires_grad for p in full.model.vision_tower.parameters())
        or any(p.requires_grad for p in full.model.embed_vision.parameters())
    ):
        raise RuntimeError("Gemma full arm source/trainable identity differs")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LORA_LR, "peak_lr": LORA_LR},
            {"params": head_params, "lr": HEAD_LR, "peak_lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    batches = epoch_batches(
        lengths, [], epoch=0, seed=SEED, microbatch=1, replay_fraction=0
    )
    order = [index for batch in batches for _, index in batch]
    if digest(order) != lock["admission"]["schedule"]["ordered_row_indices_sha256"]:
        raise RuntimeError("Gemma full arm order differs from private lock")
    metrics: dict[int, dict[str, Any]] = {}
    train_elapsed = 0.0
    seen_tokens = 0
    step = 0
    for cursor in range(0, len(order), ACCUMULATION):
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("Gemma full arm crossed frozen wall ceiling")
        window = order[cursor : cursor + ACCUMULATION]
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * learning_factor(step)
        optimizer.zero_grad(set_to_none=True)
        window_start = time.monotonic()
        for index in window:
            item = train_items[index]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = dynamic_logits(adapter, head, item)
                target = torch.tensor(
                    [item["label"]], dtype=torch.long, device="cuda:0"
                )
                mask = torch.ones_like(logits, dtype=torch.bool)
                loss = per_example_loss(
                    logits,
                    target,
                    mask,
                    objective="ce_brier",
                    brier_weight=BRIER_WEIGHT,
                )["total"].mean()
            if not torch.isfinite(loss) or not 0 < loss.item() < 1e6:
                raise RuntimeError("Gemma full arm TRAIN loss nonfinite/out of bound")
            (loss / len(window)).backward()
            seen_tokens += len(item["ids"])
        norm = torch.nn.utils.clip_grad_norm_([*lora_params, *head_params], 1.0)
        if not torch.isfinite(norm) or not 0 < norm.item() < MAX_GRAD_NORM:
            raise RuntimeError("Gemma full arm gradient norm invalid")
        if any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in [*lora_params, *head_params]
        ):
            raise RuntimeError("Gemma full arm missing/nonfinite trainable gradient")
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        train_elapsed += time.monotonic() - window_start
        if step in CHECKPOINT_STEPS:
            select_records = None
            summary = None
            if step in SELECT_STEPS:
                select_records, summary = _select_readout(adapter, head, select_items)
                metrics[step] = summary
            hashes = _save_checkpoint(
                output,
                adapter,
                head,
                optimizer,
                step,
                summary,
                lock_sha,
                cursor + len(window),
            )
            if select_records is not None:
                _write_new_json(output / f"select-step-{step:07d}.json", select_records)
            event = {
                "step": step,
                "seen_tokens": seen_tokens,
                "train_seconds": train_elapsed,
                "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
                "checkpoint_files_sha256": hashes,
                "select_metrics": summary,
            }
            _write_new_json(output / f"milestone-{step:07d}.json", event)
            if step == 16:
                estimate = (
                    (train_elapsed / seen_tokens)
                    * (lock["admission"]["schedule"]["unpadded_tokens"] - seen_tokens)
                    * 1.1
                )
                if time.monotonic() - started + estimate > MAX_WALL_SECONDS:
                    _write_new_json(
                        output / "STOP.json",
                        {"reason": "projected_above_24_GPUh", "step": step},
                    )
                    return
            if summary is not None:
                if step == 256:
                    best = max(
                        item["family_macro_accuracy"] for item in metrics.values()
                    )
                    summary = {**summary, "family_macro_accuracy": best}
                reason = select_stop_reason(step, summary)
                if reason is not None:
                    _write_new_json(
                        output / "STOP.json", {"reason": reason, "step": step}
                    )
                    return
        if step >= PLANNED_UPDATES:
            break
    if (
        step != PLANNED_UPDATES
        or seen_tokens != lock["admission"]["schedule"]["unpadded_tokens"]
    ):
        raise RuntimeError("Gemma full arm ended with wrong updates/tokens")
    chosen = checkpoint_choice(metrics)
    _write_new_json(
        output / "COMPLETE.json",
        {
            "status": "development_complete",
            "updates": step,
            "best_step": chosen,
            "best_select_macro": metrics[chosen]["family_macro_accuracy"],
            "advance_to_independent_diagnosis": metrics[chosen]["family_macro_accuracy"]
            >= ADVANCE_MACRO,
            "formal_labels_read": False,
            "lock_sha256": lock_sha,
        },
    )
    del full, adapter, head, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-lock", "cpu-audit", "run"))
    for name in (
        "snapshot",
        "train",
        "select",
        "qwen-tokenizer",
        "identity",
        "long-gate",
        "launcher",
        "lock",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_paths = [
        getattr(args, name.replace("-", "_"))
        for name in (
            "snapshot",
            "train",
            "select",
            "qwen-tokenizer",
            "identity",
            "long-gate",
            "launcher",
        )
    ]
    snapshot, train, select, qwen_path, identity, long_gate, launcher = [
        path.resolve(strict=True) for path in source_paths
    ]
    lock_path = args.lock.absolute()
    if args.mode == "prepare-lock":
        if args.output is not None or lock_path.exists():
            raise ValueError("Prepare-lock needs new private lock and no output")
        _owner_private(lock_path.parent, directory=True)
        _check_prior_receipts(identity, long_gate)
        source, _, _, admission = _full_admission(snapshot, train, select, qwen_path)
        _write_new_json(
            lock_path, expected_lock(source, admission, sha256_file(launcher))
        )
    elif args.mode == "cpu-audit":
        if args.output is not None:
            raise ValueError("CPU audit never accepts output")
        _, _, _, lock = checked_lock(
            snapshot,
            train,
            select,
            qwen_path,
            identity,
            long_gate,
            launcher,
            lock_path,
        )
        print(
            json.dumps(
                {
                    "status": "cpu_pass",
                    "lock_sha256": sha256_file(lock_path),
                    "train_count": lock["admission"]["cohort"]["admitted_count"],
                    "planned_updates": lock["admission"]["schedule"]["updates"],
                    "select_count": lock["admission"]["select_count"],
                },
                sort_keys=True,
            )
        )
    else:
        if args.output is None:
            raise ValueError("GPU run needs a new private output directory")
        _run(
            snapshot,
            train,
            select,
            qwen_path,
            identity,
            long_gate,
            launcher,
            lock_path,
            args.output.absolute(),
        )


if __name__ == "__main__":
    main()
