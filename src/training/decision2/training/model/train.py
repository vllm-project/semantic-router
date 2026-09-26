"""Single-GPU, resumable Decision 2.0 full/head/LoRA fine-tuning pilot.

Run as ``python -m training.model.train`` from the research repository root.
No benchmark gold input is accepted; select controls checkpoints, and cal is
validated for leakage but remains unopened by model evaluation here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from .data import check_partition_isolation, file_sha256, load_partition
from .decision_model import PROMPT_VERSION, DecisionModel, collate, encode
from .lora import adapter_parameters, attach_lora
from .loss import LOSS_VERSION, per_example_loss
from .plan import epoch_batches, planned_updates, replay_count, validate_resume_state
from .source import source_fingerprint

SOURCE_FILES = (
    "data.py",
    "decision_model.py",
    "loss.py",
    "plan.py",
    "source.py",
    "train.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def fsync_tree(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            descriptor = os.open(file, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
    by_name = {}
    for family, subset in sorted(by_family.items()):
        by_name[family] = {
            "n": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "accuracy": sum(row["correct"] for row in subset) / len(subset),
            "brier": sum(row["brier"] for row in subset) / len(subset),
            "nll": sum(row["nll"] for row in subset) / len(subset),
        }
    return {
        "n": len(records),
        "correct": sum(row["correct"] for row in records),
        "micro_accuracy": sum(row["correct"] for row in records) / len(records),
        "family_macro_accuracy": sum(value["accuracy"] for value in by_name.values())
        / len(by_name),
        "family_macro_brier": sum(value["brier"] for value in by_name.values())
        / len(by_name),
        "by_family": by_name,
    }


def _point(keys: list[str], probabilities: list[float]) -> int | None:
    maximum = max(probabilities)
    matches = [
        i for i, value in enumerate(probabilities) if abs(value - maximum) <= 1e-8
    ]
    return matches[0] if len(matches) == 1 else None


def evaluate(
    model: DecisionModel,
    encoded: list[dict[str, Any]],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    output: Path,
    tag: str,
) -> dict[str, Any]:
    was_training = model.training
    was_backbone_training = model.backbone.training
    model.eval()
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            items = encoded[start : start + batch_size]
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
            probabilities = logits.float().softmax(-1).cpu().tolist()
            for item, all_prob in zip(items, probabilities):
                keys = item["keys"]
                p = all_prob[: len(keys)]
                chosen = _point(keys, p)
                target = item["label"]
                answer: dict[str, Any]
                if item["task_type"] == "noul":
                    p_true = p[keys.index("true")]
                    chosen = (
                        None
                        if p_true == 0.5
                        else keys.index("true" if p_true > 0.5 else "false")
                    )
                    answer = {"type": "noul", "noul": p_true}
                elif item["task_type"] == "score":
                    expected = sum(
                        int(key) * probability for key, probability in zip(keys, p)
                    )
                    answer = {
                        "type": "score",
                        "score": expected,
                        "probabilities": dict(zip(keys, p)),
                    }
                else:
                    answer = {
                        "type": "choice",
                        "choice": keys[chosen] if chosen is not None else None,
                        "probabilities": dict(zip(keys, p)),
                    }
                records.append(
                    {
                        "id": item["id"],
                        "family": item["family"],
                        "task_type": item["task_type"],
                        "gold_key": keys[target],
                        "prediction_key": keys[chosen] if chosen is not None else None,
                        "correct": chosen == target,
                        "answer": answer,
                        "brier": sum(
                            (value - float(i == target)) ** 2
                            for i, value in enumerate(p)
                        )
                        / 2,
                        "nll": -math.log(max(p[target], 1e-12)),
                        "input_tokens": len(item["ids"]),
                        "prompt_sha256": item["prompt_sha256"],
                        "token_ids_sha256": item["token_ids_sha256"],
                    }
                )
    if was_training:
        model.train()
        model.backbone.train(was_backbone_training)
    summary = metric_summary(records)
    summary.update(
        {
            "tag": tag,
            "seconds": time.perf_counter() - started,
            "precision": "FP32 parameters; BF16 backbone compute; FP32 head",
        }
    )
    atomic_jsonl(output / f"{tag}-predictions.jsonl", records)
    atomic_json(output / f"{tag}-metrics.json", summary)
    return summary


def learning_factor(step: int, total: int, warmup_ratio: float) -> float:
    warmup = max(1, round(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        help="Local Qwen3.5-family base/posttrained, Decision 1.0, or Decision 2.0 directory",
    )
    parser.add_argument(
        "--init-kind",
        choices=("base", "posttrained", "decision1", "decision2"),
        default="base",
    )
    parser.add_argument(
        "--base-revision",
        help="Immutable Qwen source revision; required for fresh base/posttrained initialization",
    )
    parser.add_argument("--train", required=True)
    parser.add_argument(
        "--select", required=True, help="Labeled selection rows; dev checkpoints only"
    )
    parser.add_argument(
        "--cal",
        required=True,
        help="Calibration rows; lineage audit only; never evaluated in training",
    )
    parser.add_argument(
        "--replay",
        help="Separate train-split rows carrying teacher_probs keyed by option",
    )
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument("--replay-kl-weight", type=float, default=0.0)
    parser.add_argument("--objective", choices=("ce", "ce_brier"), default="ce")
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument(
        "--train-mode", choices=("full", "head", "lora"), default="full"
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument(
        "--source-path", help="Content-identical local source for exact LoRA resume"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Exact checkpoint-N directory within --output")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pilot is single-device; use one process and one GPU")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "A CUDA/ROCm device with BF16 support is required for this trainer"
        )
    if not args.resume and (
        not args.model_path
        or (args.init_kind in ("base", "posttrained") and not args.base_revision)
    ):
        raise ValueError(
            "Fresh initialization needs --model-path and Qwen source runs also need --base-revision"
        )
    if args.resume and args.model_path:
        raise ValueError(
            "Exact resume loads its model from --resume; omit --model-path"
        )
    if not args.resume and args.source_path:
        raise ValueError(
            "Fresh initialization uses --model-path; --source-path is only for LoRA resume"
        )
    if args.resume and args.train_mode == "lora" and not args.source_path:
        raise ValueError("Exact LoRA resume requires --source-path")
    if args.resume and args.train_mode != "lora" and args.source_path:
        raise ValueError("--source-path applies only to LoRA resume")
    for name in (
        "epochs",
        "microbatch",
        "accumulation",
        "eval_batch",
        "max_length",
        "head_dim",
        "save_every",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    for name in ("backbone_lr", "head_lr", "lora_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.weight_decay < 1:
        raise ValueError("Invalid warmup_ratio or weight_decay")
    if not 0 <= args.replay_fraction < 1 or not math.isfinite(args.replay_fraction):
        raise ValueError("replay_fraction must be finite and in [0,1)")
    if (args.replay and args.replay_fraction == 0) or (
        not args.replay and args.replay_fraction > 0
    ):
        raise ValueError(
            "--replay and a positive --replay-fraction must be supplied together"
        )
    if not math.isfinite(args.replay_kl_weight) or args.replay_kl_weight < 0:
        raise ValueError("replay_kl_weight must be finite and nonnegative")
    if not math.isfinite(args.brier_weight) or args.brier_weight < 0:
        raise ValueError("brier_weight must be finite and nonnegative")
    if (
        args.lora_rank < 1
        or args.lora_alpha < 1
        or not math.isfinite(args.lora_dropout)
        or not 0 <= args.lora_dropout < 1
    ):
        raise ValueError("Invalid LoRA rank, alpha, or dropout")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = Path(args.output)
    resume = Path(args.resume) if args.resume else None
    if resume:
        if (
            resume.parent.resolve() != output.resolve()
            or not (resume / "checkpoint.json").is_file()
        ):
            raise ValueError(
                "--resume must name a complete checkpoint directly inside --output"
            )
        if (output / "COMPLETE.json").exists():
            raise ValueError("This run is already complete")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run requires a new or empty output directory")

    train_rows = load_partition(args.train, "train")
    select_rows = load_partition(args.select, "select")
    cal_rows = load_partition(args.cal, "cal")
    replay_rows = (
        load_partition(args.replay, "train", replay=True) if args.replay else []
    )
    check_partition_isolation(
        {
            "train": train_rows,
            "replay": replay_rows,
            "select": select_rows,
            "cal": cal_rows,
        }
    )
    data_sha = {
        "train": file_sha256(args.train),
        "select": file_sha256(args.select),
        "cal": file_sha256(args.cal),
    }
    if args.replay:
        data_sha["replay"] = file_sha256(args.replay)
    code_files = (
        (*SOURCE_FILES, "lora.py") if args.train_mode == "lora" else SOURCE_FILES
    )
    source_code_sha = {
        name: file_sha256(Path(__file__).with_name(name)) for name in code_files
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    if resume:
        prior = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        source = prior["model_source"]
        model, tokenizer = DecisionModel.from_checkpoint(
            resume,
            source_path=args.source_path,
            trainable_adapter=args.train_mode == "lora",
        )
        if (
            args.train_mode == "lora"
            and model.metadata["lora"]["source_fingerprint"] != source
        ):
            raise ValueError(
                "Resume checkpoint source fingerprint differs from run provenance"
            )
        if args.train_mode == "lora" and model.metadata["lora"].get(
            "peft_version"
        ) != version("peft"):
            raise ValueError("Exact LoRA resume requires the original PEFT version")
    else:
        source = source_fingerprint(Path(args.model_path))
        if args.init_kind in ("base", "posttrained"):
            model, tokenizer = DecisionModel.from_base(
                args.model_path,
                args.base_revision,
                args.head_dim,
                source_stage=args.init_kind,
            )
        elif args.init_kind == "decision1":
            model, tokenizer = DecisionModel.from_decision1(
                args.model_path, args.head_dim
            )
        else:
            model, tokenizer = DecisionModel.from_checkpoint(args.model_path)
            if model.metadata["head_dim"] != args.head_dim:
                raise ValueError(
                    "--head-dim must match the Decision 2.0 initialization checkpoint"
                )
        if args.train_mode == "lora":
            attach_lora(
                model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                source_kind=args.init_kind,
                source_fingerprint=source,
            )
    model = model.float().to(device)
    if args.train_mode == "head":
        model.backbone.requires_grad_(False)
    elif args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.backbone.config.use_cache = False
    model.metadata.update(
        {"training_mode": args.train_mode, "loss_version": LOSS_VERSION}
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    train_items = [encode(row, tokenizer, args.max_length) for row in train_rows]
    replay_items = [encode(row, tokenizer, args.max_length) for row in replay_rows]
    select_items = [encode(row, tokenizer, args.max_length) for row in select_rows]
    # Cal data is parsed and hashed for split isolation, but never tokenized or evaluated here.
    train_lengths = [len(item["ids"]) for item in train_items]
    replay_lengths = [len(item["ids"]) for item in replay_items]
    planned = planned_updates(
        len(train_items),
        len(replay_items),
        args.replay_fraction,
        args.microbatch,
        args.accumulation,
        args.epochs,
        args.max_steps,
    )
    contract = {
        "prompt_version": PROMPT_VERSION,
        "loss_version": LOSS_VERSION,
        "model_source": source,
        "data_sha256": data_sha,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "microbatch": args.microbatch,
        "accumulation": args.accumulation,
        "eval_batch": args.eval_batch,
        "max_length": args.max_length,
        "head_dim": args.head_dim,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_every": args.save_every,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "init_kind": args.init_kind,
        "base_revision": args.base_revision,
        "objective": args.objective,
        "brier_weight": args.brier_weight,
        "replay_fraction": args.replay_fraction,
        "replay_kl_weight": args.replay_kl_weight,
        "train_mode": args.train_mode,
        "planned_updates": planned,
        "train_count": len(train_items),
        "replay_pool_count": len(replay_items),
    }
    if args.train_mode == "lora":
        contract["lora"] = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "lr": args.lora_lr,
            "target_modules": model.metadata["lora"]["target_modules"],
            "peft_version": model.metadata["lora"]["peft_version"],
        }
    if resume and contract != prior["contract"]:
        raise ValueError("Current arguments or data differ from original run contract")

    parameter_groups = []
    if args.train_mode == "full":
        parameter_groups.append(
            {
                "params": list(model.backbone.parameters()),
                "lr": args.backbone_lr,
                "peak_lr": args.backbone_lr,
                "name": "backbone",
            }
        )
    elif args.train_mode == "lora":
        parameter_groups.append(
            {
                "params": adapter_parameters(model),
                "lr": args.lora_lr,
                "peak_lr": args.lora_lr,
                "name": "lora",
            }
        )
    parameter_groups.append(
        {
            "params": list(model.head.parameters()),
            "lr": args.head_lr,
            "peak_lr": args.head_lr,
            "name": "head",
        }
    )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay, foreach=True
    )
    step = start_epoch = start_batch = 0
    if resume:
        state = torch.load(
            resume / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        validate_resume_state(state, contract, source_code_sha)
        optimizer.load_state_dict(state["optimizer"])
        step, start_epoch, start_batch = (
            state["step"],
            state["next_epoch"],
            state["next_batch"],
        )
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "provenance.json",
            {
                "created_utc": utc_now(),
                "contract": contract,
                "code_sha256": source_code_sha,
                "model_source": source,
                "train_examples": len(train_items),
                "replay_pool_examples": len(replay_items),
                "replay_examples_per_epoch": replay_count(
                    len(train_items), len(replay_items), args.replay_fraction
                ),
                "select_examples": len(select_items),
                "cal_examples_audited_only": len(cal_rows),
                "train_tokens": sum(train_lengths),
                "train_max_tokens": max(train_lengths),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "trainable_parameters_by_component": {
                    "backbone_or_adapter": sum(
                        p.numel()
                        for p in model.backbone.parameters()
                        if p.requires_grad
                    ),
                    "head": sum(
                        p.numel() for p in model.head.parameters() if p.requires_grad
                    ),
                },
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "precision": (
                    "FP32 frozen backbone; FP32 LoRA/head and Adam state; BF16 autocast backbone; FP32 head/loss"
                    if args.train_mode == "lora"
                    else "FP32 backbone/head parameters and Adam state; BF16 autocast backbone; FP32 head/loss"
                ),
                "calibration_policy": "Cal records are never fed to this trainer or checkpoint selector",
            },
        )
    metrics_path = output / "train-metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")
    if resume and step == planned:
        atomic_json(
            output / "COMPLETE.json",
            {
                "status": "complete",
                "step": step,
                "planned_updates": planned,
                "best": json.loads((output / "BEST.json").read_text())["checkpoint"],
                "completed_utc": utc_now(),
                "calibration_status": "untouched",
            },
        )
        metrics_file.close()
        return

    def log(event: dict[str, Any]) -> None:
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        metrics_file.write(text + "\n")
        metrics_file.flush()
        os.fsync(metrics_file.fileno())
        print(text, flush=True)

    if resume:
        log(
            {
                "event": "resume",
                "step": step,
                "checkpoint": resume.name,
                "next_epoch": start_epoch,
                "next_batch": start_batch,
                "note": "Later train log entries from an interrupted attempt may be superseded by this resume",
            }
        )

    def save_checkpoint(
        next_epoch: int, next_batch: int, dev_metrics: dict[str, Any]
    ) -> None:
        name = f"checkpoint-{step:07d}"
        destination = output / name
        pending = output / f"{name}.pending"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite complete checkpoint {destination}")
        if pending.exists():
            shutil.rmtree(pending)
        model.save(pending, tokenizer)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "contract": contract,
                "code_sha256": source_code_sha,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
            },
            pending / "trainer_state.pt",
        )
        atomic_json(
            pending / "checkpoint.json",
            {
                "step": step,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "dev_metrics": dev_metrics,
                "complete": True,
                "saved_utc": utc_now(),
            },
        )
        fsync_tree(pending)
        os.replace(pending, destination)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_json(output / "LATEST.json", {"checkpoint": name, "step": step})
        checkpoints = [
            path
            for path in output.glob("checkpoint-*")
            if path.is_dir() and not path.name.endswith(".pending")
        ]
        best = max(
            checkpoints,
            key=lambda path: (
                json.loads((path / "checkpoint.json").read_text())["dev_metrics"][
                    "family_macro_accuracy"
                ],
                -json.loads((path / "checkpoint.json").read_text())["dev_metrics"][
                    "family_macro_brier"
                ],
                -int(path.name.split("-")[-1]),
            ),
        )
        atomic_json(
            output / "BEST.json",
            {
                "checkpoint": best.name,
                "selection": "select family-macro accuracy descending, then normalized Brier ascending, then earliest step",
            },
        )
        log(
            {"event": "checkpoint", "step": step, "checkpoint": name, "best": best.name}
        )

    if not resume:
        baseline = evaluate(
            model,
            select_items,
            pad_id=pad_id,
            batch_size=args.eval_batch,
            device=device,
            output=output,
            tag="select-baseline",
        )
        log({"event": "baseline", "metrics": baseline})
    model.train()
    if args.train_mode == "head":
        model.backbone.eval()
    last_saved = step if resume else -1
    for epoch in range(start_epoch, args.epochs):
        batches = epoch_batches(
            train_lengths,
            replay_lengths,
            epoch=epoch,
            seed=args.seed,
            microbatch=args.microbatch,
            replay_fraction=args.replay_fraction,
        )
        first = start_batch if epoch == start_epoch else 0
        if first > len(batches) or (
            first % args.accumulation != 0 and first != len(batches)
        ):
            raise ValueError(
                "Saved batch cursor does not align with an optimizer boundary"
            )
        for window in range(first, len(batches), args.accumulation):
            window_end = min(window + args.accumulation, len(batches))
            window_batches = batches[window:window_end]
            window_count = sum(len(batch) for batch in window_batches)
            factor = learning_factor(step, planned, args.warmup_ratio)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            sums = dict.fromkeys(("total", "ce", "brier", "replay_kl"), 0.0)
            correct = tokens = replay_seen = 0
            started = time.perf_counter()
            for batch_ids in window_batches:
                items = [
                    (
                        train_items[index]
                        if source_name == "train"
                        else replay_items[index]
                    )
                    for source_name, index in batch_ids
                ]
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
                        objective=args.objective,
                        brier_weight=args.brier_weight,
                        teacher_probs=batch["teacher_probs"],
                        replay_mask=batch["replay_mask"],
                        replay_kl_weight=args.replay_kl_weight,
                    )
                    loss = terms["total"].sum() / window_count
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                for name in sums:
                    sums[name] += terms[name].detach().sum().item()
                correct += (logits.argmax(-1) == batch["labels"]).sum().item()
                tokens += batch["attention_mask"].sum().item()
                replay_seen += batch["replay_mask"].sum().item()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            torch.cuda.synchronize(device)
            step += 1
            elapsed = time.perf_counter() - started
            log(
                {
                    "event": "train",
                    "step": step,
                    "epoch": epoch,
                    "examples": window_count,
                    "replay_examples": replay_seen,
                    "loss": sums["total"] / window_count,
                    "ce": sums["ce"] / window_count,
                    "brier": sums["brier"] / window_count,
                    "replay_kl": sums["replay_kl"] / max(1, replay_seen),
                    "accuracy": correct / window_count,
                    "tokens": tokens,
                    "learning_rates": {
                        group["name"]: group["lr"] for group in optimizer.param_groups
                    },
                    "gradient_norm": gradient_norm.item(),
                    "seconds": elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                }
            )
            next_epoch = epoch + 1 if window_end == len(batches) else epoch
            next_batch = 0 if window_end == len(batches) else window_end
            if step % args.save_every == 0 or step == planned:
                dev_metrics = evaluate(
                    model,
                    select_items,
                    pad_id=pad_id,
                    batch_size=args.eval_batch,
                    device=device,
                    output=output,
                    tag=f"select-step-{step:07d}",
                )
                log({"event": "select", "step": step, "metrics": dev_metrics})
                save_checkpoint(next_epoch, next_batch, dev_metrics)
                last_saved = step
            if step >= planned:
                break
        if step >= planned:
            break
    if last_saved != step:
        raise RuntimeError("Final optimizer step has no durable checkpoint")
    atomic_json(
        output / "COMPLETE.json",
        {
            "status": "complete",
            "step": step,
            "planned_updates": planned,
            "best": json.loads((output / "BEST.json").read_text())["checkpoint"],
            "completed_utc": utc_now(),
            "calibration_status": "untouched",
        },
    )
    metrics_file.close()


if __name__ == "__main__":
    main()
