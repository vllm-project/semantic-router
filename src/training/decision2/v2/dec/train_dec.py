"""Single-GPU LoRA continuation trainer for decoder-track factor contrasts.

Every arm shares the source model, rights-clean partitions, ordered batches,
optimizer and SELECT rule of the control; the registered factor is the only
intended difference:

* ``--type-balance inverse``: per-example loss weights that give Choice, Noul
  and Score equal total TRAIN weight (window-normalized, as in the shared trainer).
* ``--teacher``: KL(teacher || student) on every TRAIN row, from a pinned
  own-family teacher distribution file keyed by row ID and input hash.
* ``--residual``: zero-gated residual readouts from ``dec_model``.

Run from ``src/training/decision2`` as ``python3 -m v2.dec.train_dec``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from training.model.data import check_partition_isolation, file_sha256, load_partition
from training.model.decision_model import PROMPT_VERSION, DecisionModel, collate, encode
from training.model.lora import adapter_parameters, attach_lora
from training.model.loss import LOSS_VERSION, per_example_loss
from training.model.plan import epoch_batches, planned_updates
from training.model.source import source_fingerprint
from training.model.train import atomic_json, evaluate, fsync_tree, learning_factor

from .dec_model import RESIDUALS, DecModel

TRAINER_VERSION = "dec-factor-trainer/1"
TASK_TYPES = ("choice", "noul", "score")
SHARED_FILES = (
    "data.py",
    "decision_model.py",
    "loss.py",
    "plan.py",
    "source.py",
    "lora.py",
    "train.py",
    "infer.py",
)
OWN_FILES = ("train_dec.py", "dec_model.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_hashes() -> dict[str, str]:
    shared = Path(__file__).resolve().parents[2] / "training" / "model"
    hashes = {
        f"training/model/{name}": file_sha256(shared / name) for name in SHARED_FILES
    }
    hashes.update(
        {
            f"v2/dec/{name}": file_sha256(Path(__file__).with_name(name))
            for name in OWN_FILES
        }
    )
    return hashes


def type_weights(rows: list[dict[str, Any]], mode: str) -> dict[str, float]:
    """Per-type example weights; ``inverse`` equalizes each type's total weight."""
    if mode == "none":
        return dict.fromkeys(TASK_TYPES, 1.0)
    if mode != "inverse":
        raise ValueError("type balance must be none or inverse")
    counts = Counter(row["task_type"] for row in rows)
    if any(counts[kind] == 0 for kind in TASK_TYPES):
        raise ValueError("Inverse type balance needs every native type in TRAIN")
    return {kind: len(rows) / (len(TASK_TYPES) * counts[kind]) for kind in TASK_TYPES}


def load_teacher(path: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Map TRAIN id to a complete option-key distribution bound to its input hash."""
    by_id = {row["id"]: row for row in rows}
    teacher: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            row = by_id.get(record.get("id"))
            if row is None or record["id"] in teacher:
                raise ValueError(f"{path}:{line_number}: unknown or repeated TRAIN id")
            if record.get("input_sha256") != row["input_sha256"]:
                raise ValueError(f"{path}:{line_number}: teacher input hash differs")
            probs = record.get("teacher_probs")
            keys = [option["key"] for option in row["options"]]
            if not isinstance(probs, dict) or set(probs) != set(keys):
                raise ValueError(
                    f"{path}:{line_number}: teacher keys differ from options"
                )
            values = [probs[key] for key in keys]
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in values
                )
                or abs(sum(values) - 1.0) > 1e-6
            ):
                raise ValueError(f"{path}:{line_number}: invalid teacher distribution")
            teacher[record["id"]] = probs
    if set(teacher) != set(by_id):
        raise ValueError("Teacher file must cover every TRAIN row exactly once")
    return teacher


def checkpoint_steps(planned: int, schedule: str, save_every: int) -> set[int]:
    """Updates after which SELECT runs and a checkpoint is saved."""
    if schedule == "even8":
        return {round(planned * k / 8) for k in range(1, 9)} - {0} | {planned}
    if schedule == "every":
        return {s for s in range(1, planned + 1) if s % save_every == 0} | {planned}
    raise ValueError("checkpoint schedule must be even8 or every")


def selection_key(metrics: dict[str, Any], step: int, rule: str) -> tuple[float, ...]:
    """Larger is better; matrix-v1 breaks accuracy ties by the earlier step."""
    if rule == "matrix-v1":
        return (metrics["family_macro_accuracy"], -step)
    if rule == "shared":
        return (metrics["family_macro_accuracy"], -metrics["family_macro_brier"], -step)
    raise ValueError("selection rule must be matrix-v1 or shared")


def attach_teacher_probs(item: dict[str, Any], probs: dict[str, float]) -> None:
    values = [float(probs[key]) for key in item["keys"]]
    total = math.fsum(values)
    item["teacher_probs"] = [value / total for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Decision 1.0 package"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument(
        "--cal", type=Path, required=True, help="Hashed for isolation only"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Registered arm name")
    parser.add_argument("--type-balance", choices=("none", "inverse"), default="none")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0)
    parser.add_argument("--residual", action="append", default=[], choices=RESIDUALS)
    parser.add_argument("--residual-lr", type=float, default=5e-4)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument(
        "--checkpoint-schedule", choices=("even8", "every"), default="even8"
    )
    parser.add_argument(
        "--selection", choices=("matrix-v1", "shared"), default="matrix-v1"
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--zero-step-only", action="store_true")
    args = parser.parse_args()
    if bool(args.teacher) != (args.teacher_kl_weight > 0):
        parser.error("--teacher and a positive --teacher-kl-weight go together")
    if not math.isfinite(args.teacher_kl_weight) or args.teacher_kl_weight < 0:
        parser.error("teacher KL weight must be finite and nonnegative")
    for name in ("lora_lr", "head_lr", "residual_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A ROCm/CUDA BF16 device is required")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run requires a new or empty output directory")
    train_rows = load_partition(args.train, "train")
    select_rows = load_partition(args.select, "select")
    cal_rows = load_partition(args.cal, "cal")
    check_partition_isolation(
        {"train": train_rows, "select": select_rows, "cal": cal_rows}
    )
    weights_by_type = type_weights(train_rows, args.type_balance)
    teacher = load_teacher(args.teacher, train_rows) if args.teacher else None
    data_sha = {
        "train": file_sha256(args.train),
        "select": file_sha256(args.select),
        "cal": file_sha256(args.cal),
    }
    if args.teacher:
        data_sha["teacher"] = file_sha256(args.teacher)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    source = source_fingerprint(args.model_path)
    base, tokenizer = DecisionModel.from_decision1(args.model_path, args.head_dim)
    model = DecModel.wrap(base, tuple(args.residual))
    attach_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        source_kind="decision1",
        source_fingerprint=source,
    )
    model = model.float().to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.metadata.update({"training_mode": "lora", "loss_version": LOSS_VERSION})
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")

    train_items = [encode(row, tokenizer, args.max_length) for row in train_rows]
    if teacher is not None:
        for row, item in zip(train_rows, train_items):
            attach_teacher_probs(item, teacher[row["id"]])
    select_items = [encode(row, tokenizer, args.max_length) for row in select_rows]
    train_lengths = [len(item["ids"]) for item in train_items]
    example_weights = [weights_by_type[row["task_type"]] for row in train_rows]
    planned = planned_updates(
        len(train_items),
        0,
        0.0,
        args.microbatch,
        args.accumulation,
        args.epochs,
        args.max_steps,
    )
    save_steps = checkpoint_steps(planned, args.checkpoint_schedule, args.save_every)
    contract = {
        "trainer_version": TRAINER_VERSION,
        "arm": args.arm,
        "prompt_version": PROMPT_VERSION,
        "loss_version": LOSS_VERSION,
        "objective": "ce_brier",
        "brier_weight": args.brier_weight,
        "model_source": source,
        "init_kind": "decision1",
        "data_sha256": data_sha,
        "type_balance": args.type_balance,
        "type_weights": weights_by_type,
        "teacher_kl_weight": args.teacher_kl_weight,
        "residuals": sorted(args.residual),
        "residual_lr": args.residual_lr if args.residual else None,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "lr": args.lora_lr,
            "target_modules": model.metadata["lora"]["target_modules"],
            "peft_version": version("peft"),
        },
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "microbatch": args.microbatch,
        "accumulation": args.accumulation,
        "eval_batch": args.eval_batch,
        "max_length": args.max_length,
        "head_dim": args.head_dim,
        "checkpoint_schedule": args.checkpoint_schedule,
        "checkpoint_steps": sorted(save_steps),
        "seed": args.seed,
        "planned_updates": planned,
        "train_count": len(train_items),
        "zero_step_only": args.zero_step_only,
        "selection": (
            "SELECT family-macro accuracy desc, then earliest step (matrix v1)"
            if args.selection == "matrix-v1"
            else "SELECT family-macro accuracy desc, normalized Brier asc, earliest step"
        ),
    }
    groups = [
        {
            "params": adapter_parameters(model),
            "lr": args.lora_lr,
            "peak_lr": args.lora_lr,
            "name": "lora",
        },
        {
            "params": list(model.head.parameters()),
            "lr": args.head_lr,
            "peak_lr": args.head_lr,
            "name": "head",
        },
    ]
    if model.residual_parameters():
        groups.append(
            {
                "params": model.residual_parameters(),
                "lr": args.residual_lr,
                "peak_lr": args.residual_lr,
                "name": "residual",
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, foreach=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "provenance.json",
        {
            "created_utc": utc_now(),
            "contract": contract,
            "model_source": source,
            "code_sha256": code_hashes(),
            "source_commit": os.environ.get("DEC_SOURCE_COMMIT"),
            "source_tree": os.environ.get("DEC_SOURCE_TREE"),
            "image_id": os.environ.get("DEC_IMAGE_ID"),
            "train_examples": len(train_items),
            "train_type_counts": dict(Counter(row["task_type"] for row in train_rows)),
            "train_tokens": sum(train_lengths),
            "train_max_tokens": max(train_lengths),
            "select_examples": len(select_items),
            "cal_examples_audited_only": len(cal_rows),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "trainable_by_group": {
                g["name"]: sum(p.numel() for p in g["params"]) for g in groups
            },
            "versions": {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "transformers": version("transformers"),
                "peft": version("peft"),
            },
            "device_name": torch.cuda.get_device_name(device),
            "precision": "FP32 frozen backbone; FP32 LoRA/head/residual and Adam; BF16 autocast backbone; FP32 head/loss",
            "calibration_policy": "CAL rows are never fed to this trainer or checkpoint selector",
        },
    )
    metrics_file = (output / "train-metrics.jsonl").open("a", encoding="utf-8")

    def log(event: dict[str, Any]) -> None:
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        metrics_file.write(text + "\n")
        metrics_file.flush()
        print(text, flush=True)

    step = 0
    started_utc = utc_now()
    started = time.perf_counter()

    def save_checkpoint(select_metrics: dict[str, Any]) -> None:
        name = f"checkpoint-{step:07d}"
        pending = output / f"{name}.pending"
        if (output / name).exists():
            raise ValueError(f"Refusing to overwrite {name}")
        if pending.exists():
            shutil.rmtree(pending)
        model.save(pending, tokenizer)
        atomic_json(
            pending / "checkpoint.json",
            {
                "step": step,
                "dev_metrics": select_metrics,
                "complete": True,
                "saved_utc": utc_now(),
            },
        )
        fsync_tree(pending)
        os.replace(pending, output / name)
        checkpoints = [
            p
            for p in output.glob("checkpoint-*")
            if p.is_dir() and not p.name.endswith(".pending")
        ]

        def rank(path: Path) -> tuple[float, ...]:
            metrics = json.loads((path / "checkpoint.json").read_text())["dev_metrics"]
            return selection_key(metrics, int(path.name.split("-")[-1]), args.selection)

        best = max(checkpoints, key=rank)
        atomic_json(output / "LATEST.json", {"checkpoint": name, "step": step})
        atomic_json(
            output / "BEST.json",
            {"checkpoint": best.name, "selection": contract["selection"]},
        )
        log(
            {"event": "checkpoint", "step": step, "checkpoint": name, "best": best.name}
        )

    baseline = evaluate(
        model,
        select_items,
        pad_id=pad_id,
        batch_size=args.eval_batch,
        device=device,
        output=output,
        tag="select-baseline",
    )
    log(
        {
            "event": "baseline",
            "metrics": {k: v for k, v in baseline.items() if k != "by_family"},
        }
    )
    if args.zero_step_only:
        save_checkpoint(baseline)
        metrics_file.close()
        return
    model.train()
    batches = epoch_batches(
        train_lengths,
        [],
        epoch=0,
        seed=args.seed,
        microbatch=args.microbatch,
        replay_fraction=0.0,
    )
    all_batches = [
        batch
        for epoch in range(args.epochs)
        for batch in (
            batches
            if epoch == 0
            else epoch_batches(
                train_lengths,
                [],
                epoch=epoch,
                seed=args.seed,
                microbatch=args.microbatch,
                replay_fraction=0.0,
            )
        )
    ]
    for window in range(0, len(all_batches), args.accumulation):
        if step >= planned:
            break
        window_batches = all_batches[window : window + args.accumulation]
        indices = [index for batch in window_batches for _, index in batch]
        window_weight = sum(example_weights[i] for i in indices)
        factor = learning_factor(step, planned, args.warmup_ratio)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = defaultdict(float)
        by_type: dict[str, list[float]] = defaultdict(list)
        tokens = correct = 0
        window_started = time.perf_counter()
        for batch_ids in window_batches:
            items = [train_items[index] for _, index in batch_ids]
            weights = torch.tensor(
                [example_weights[index] for _, index in batch_ids], device=device
            )
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in collate(items, pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
                terms = per_example_loss(
                    logits,
                    batch["labels"],
                    batch["candidate_mask"],
                    objective="ce_brier",
                    brier_weight=args.brier_weight,
                    teacher_probs=(
                        batch["teacher_probs"] if teacher is not None else None
                    ),
                    replay_mask=batch["replay_mask"] if teacher is not None else None,
                    replay_kl_weight=args.teacher_kl_weight,
                )
                loss = (terms["total"] * weights).sum() / window_weight
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            for name in ("total", "ce", "brier", "replay_kl"):
                sums[name] += terms[name].detach().sum().item()
            for item, value in zip(items, terms["ce"].detach().tolist()):
                by_type[item["task_type"]].append(value)
            correct += (logits.argmax(-1) == batch["labels"]).sum().item()
            tokens += batch["attention_mask"].sum().item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        torch.cuda.synchronize(device)
        step += 1
        count = len(indices)
        event = {
            "event": "train",
            "step": step,
            "examples": count,
            "loss": sums["total"] / count,
            "ce": sums["ce"] / count,
            "brier": sums["brier"] / count,
            "teacher_kl": sums["replay_kl"] / count,
            "ce_by_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
            "accuracy": correct / count,
            "tokens": tokens,
            "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
            "gradient_norm": gradient_norm.item(),
            "seconds": time.perf_counter() - window_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        if model.ordinal_score is not None:
            event["ordinal_gate"] = model.ordinal_score.gate.item()
        if model.layer_mix is not None:
            event["layer_mix_gate"] = model.layer_mix.gate.item()
        log(event)
        if step in save_steps:
            metrics = evaluate(
                model,
                select_items,
                pad_id=pad_id,
                batch_size=args.eval_batch,
                device=device,
                output=output,
                tag=f"select-step-{step:07d}",
            )
            log(
                {
                    "event": "select",
                    "step": step,
                    "metrics": {k: v for k, v in metrics.items() if k != "by_family"},
                }
            )
            save_checkpoint(metrics)
    if step != planned:
        raise RuntimeError("Run ended before its planned optimizer horizon")
    atomic_json(
        output / "COMPLETE.json",
        {
            "status": "complete",
            "step": step,
            "planned_updates": planned,
            "best": json.loads((output / "BEST.json").read_text())["checkpoint"],
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "calibration_status": "untouched",
        },
    )
    metrics_file.close()


if __name__ == "__main__":
    main()
