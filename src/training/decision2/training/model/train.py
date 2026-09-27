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
import re
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
from .infer import checkpoint_fingerprint
from .inline_replay import attach_inline_teacher
from .lora import LORA_FORMAT, adapter_parameters, attach_lora
from .loss import LOSS_VERSION, ORDINAL_LOSS_VERSION, per_example_loss
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
TYPED_HEAD_SOURCE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TYPED_HEAD_PARTITIONS = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
TYPED_HEAD_SOURCE_FILES = {
    "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
}


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
        help="Local official Qwen3/Qwen3.5 base or posttrained, Decision 1.0, or Decision 2.0 directory",
    )
    parser.add_argument(
        "--init-kind",
        choices=("base", "posttrained", "decision1", "decision2", "decision2-lora"),
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
    parser.add_argument(
        "--choice-source",
        action="append",
        default=[],
        help="Exact TRAIN source whose Choice rows receive the fixed source weight",
    )
    parser.add_argument("--choice-source-weight", type=float, default=1.0)
    parser.add_argument(
        "--inline-teacher",
        help="Private source probabilities for existing TRAIN rows; adds no samples",
    )
    parser.add_argument(
        "--inline-teacher-roster-sha256",
        help="Precommitted SHA-256 of ordered inline {id,input_sha256} roster",
    )
    parser.add_argument("--inline-teacher-control-sha256")
    parser.add_argument("--inline-teacher-parity-roster-sha256")
    parser.add_argument("--inline-teacher-source-model-sha256")
    parser.add_argument("--inline-teacher-source-receipt-sha256")
    parser.add_argument("--objective", choices=("ce", "ce_brier"), default="ce")
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument(
        "--ordinal-weight",
        type=float,
        default=0.0,
        help="Optional normalized ranked-probability loss on Score rows only",
    )
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
    parser.add_argument(
        "--initial-model-sha256",
        help="Required original adapter-plus-base fingerprint for direct LoRA continuation",
    )
    parser.add_argument(
        "--direct-lora-parity-receipt",
        help="Sealed PASS receipt for both zero-step native BF16 direct-LoRA starts",
    )
    parser.add_argument(
        "--direct-lora-parity-sha256",
        help="SHA-256 of the sealed direct-LoRA parity receipt",
    )
    parser.add_argument("--direct-lora-arm", choices=("A", "B", "C"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Exact checkpoint-N directory within --output")
    parser.add_argument(
        "--zero-step-only",
        action="store_true",
        help="Write native SELECT baseline and exit before any optimizer update",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--head-variant",
        choices=("shared", "type-separated", "candidate-interaction"),
        default="shared",
        help="Explicit experimental readout; existing checkpoints remain shared",
    )
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
    if args.resume and args.zero_step_only:
        raise ValueError("Zero-step-only probe must use fresh initialization")
    if not args.resume and args.source_path and args.init_kind != "decision2-lora":
        raise ValueError(
            "Fresh initialization uses --model-path; --source-path is only for direct LoRA continuation or resume"
        )
    if args.init_kind == "decision2-lora":
        if (
            args.train_mode != "lora"
            or not args.source_path
            or not isinstance(args.initial_model_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.initial_model_sha256)
            or args.base_revision is not None
            or not args.direct_lora_parity_receipt
            or not isinstance(args.direct_lora_parity_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.direct_lora_parity_sha256)
            or args.direct_lora_arm not in ("A", "B", "C")
        ):
            raise ValueError(
                "Direct LoRA continuation needs --train-mode lora, --source-path, "
                "a 64-character --initial-model-sha256, a sealed parity receipt/hash/arm, "
                "and no --base-revision"
            )
    elif any(
        value is not None
        for value in (
            args.initial_model_sha256,
            args.direct_lora_parity_receipt,
            args.direct_lora_parity_sha256,
            args.direct_lora_arm,
        )
    ):
        raise ValueError(
            "Direct LoRA source and parity flags apply only to decision2-lora"
        )
    if args.resume and args.train_mode == "lora" and not args.source_path:
        raise ValueError("Exact LoRA resume requires --source-path")
    if args.resume and args.train_mode != "lora" and args.source_path:
        raise ValueError("--source-path applies only to LoRA resume")
    if args.head_variant in ("type-separated", "candidate-interaction") and (
        args.init_kind != "base"
        or args.train_mode != "full"
        or args.base_revision != TYPED_HEAD_SOURCE_REVISION
        or args.head_dim != 256
        or args.seed != 20260926
        or args.replay
        or args.inline_teacher
        or args.choice_source
        or args.replay_fraction != 0
        or args.replay_kl_weight != 0
        or args.choice_source_weight != 1
        or args.objective != "ce_brier"
        or args.brier_weight != 0.5
        or args.ordinal_weight != 0
        or args.epochs != 1
        or args.max_steps != 466
        or args.microbatch != 1
        or args.accumulation != 16
        or args.eval_batch != 2
        or args.max_length != 8192
        or args.backbone_lr != 2e-5
        or args.head_lr != 2e-4
        or args.weight_decay != 0.01
        or args.warmup_ratio != 0.05
        or args.save_every != 64
        or not args.gradient_checkpointing
    ):
        raise ValueError(
            "The type-separated ablation must use its frozen official source, "
            "full 466-step matched schedule and objective without added data or replay"
        )
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
    if (
        not math.isfinite(args.choice_source_weight)
        or args.choice_source_weight < 1.0
        or len(set(args.choice_source)) != len(args.choice_source)
        or (args.choice_source_weight != 1.0) != bool(args.choice_source)
    ):
        raise ValueError("Weighted Choice needs unique source IDs and weight >= 1")
    if args.inline_teacher:
        if (
            args.replay
            or args.replay_fraction
            or not isinstance(args.inline_teacher_roster_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.inline_teacher_roster_sha256)
            or not isinstance(args.inline_teacher_control_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.inline_teacher_control_sha256)
            or not isinstance(args.inline_teacher_parity_roster_sha256, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", args.inline_teacher_parity_roster_sha256
            )
            or not isinstance(args.inline_teacher_source_model_sha256, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", args.inline_teacher_source_model_sha256
            )
            or not isinstance(args.inline_teacher_source_receipt_sha256, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", args.inline_teacher_source_receipt_sha256
            )
            or args.replay_kl_weight <= 0
        ):
            raise ValueError(
                "Inline teacher needs a frozen roster and positive KL weight, "
                "without appended replay rows"
            )
    elif any(
        value is not None
        for value in (
            args.inline_teacher_roster_sha256,
            args.inline_teacher_control_sha256,
            args.inline_teacher_parity_roster_sha256,
            args.inline_teacher_source_model_sha256,
            args.inline_teacher_source_receipt_sha256,
        )
    ):
        raise ValueError("Inline identity hashes need --inline-teacher")
    if not math.isfinite(args.brier_weight) or args.brier_weight < 0:
        raise ValueError("brier_weight must be finite and nonnegative")
    if not math.isfinite(args.ordinal_weight) or args.ordinal_weight < 0:
        raise ValueError("ordinal_weight must be finite and nonnegative")
    if args.ordinal_weight and args.objective != "ce_brier":
        raise ValueError("Ordinal Score loss requires ce_brier")
    if (
        args.lora_rank < 1
        or args.lora_alpha < 1
        or not math.isfinite(args.lora_dropout)
        or not 0 <= args.lora_dropout < 1
    ):
        raise ValueError("Invalid LoRA rank, alpha, or dropout")


def load_direct_lora_start(
    args: argparse.Namespace,
) -> tuple[DecisionModel, Any, dict[str, Any]]:
    """Reuse one byte-pinned PEFT adapter/head with a new optimizer later."""
    identity = checkpoint_fingerprint(Path(args.model_path), Path(args.source_path))
    if identity["model_sha256"] != args.initial_model_sha256:
        raise ValueError("Initial LoRA adapter-plus-base fingerprint mismatch")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.model_path, source_path=args.source_path, trainable_adapter=True
    )
    lora = model.metadata.get("lora")
    if (
        model.metadata.get("checkpoint_format") != LORA_FORMAT
        or not isinstance(lora, dict)
        or lora.get("rank") != args.lora_rank
        or lora.get("alpha") != args.lora_alpha
        or lora.get("dropout") != args.lora_dropout
        or lora.get("peft_version") != version("peft")
        or model.metadata.get("head_dim") != args.head_dim
        or not isinstance(lora.get("source_fingerprint"), dict)
    ):
        raise ValueError("Initial LoRA topology, PEFT version, head or source differs")
    active = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    ]
    if not active or any("lora_" not in name for name in active):
        raise ValueError(
            "Direct continuation needs only the existing trainable LoRA tensors"
        )
    if not all(parameter.requires_grad for parameter in model.head.parameters()):
        raise ValueError("Selected Decision head is not fully trainable")
    model.metadata["continuation_origin"] = {
        "initial_model_sha256": identity["model_sha256"],
        "initial_adapter_sha256": identity["files_sha256"][
            "checkpoint/adapter/adapter_model.safetensors"
        ],
        "initial_head_sha256": identity["files_sha256"][
            "checkpoint/decision_head.safetensors"
        ],
    }
    return model, tokenizer, identity


def direct_lora_contract_fields(args: argparse.Namespace) -> dict[str, str]:
    """Keep preexisting run contracts byte-compatible outside the new mode."""
    return (
        {
            "initial_model_sha256": args.initial_model_sha256,
            "direct_lora_parity_sha256": args.direct_lora_parity_sha256,
            "direct_lora_arm": args.direct_lora_arm,
        }
        if args.init_kind == "decision2-lora"
        else {}
    )


def verify_direct_lora_parity_gate(args: argparse.Namespace) -> None:
    """Reject a direct-LoRA optimizer start without both frozen BF16 starts."""
    if args.init_kind != "decision2-lora":
        return
    path = Path(args.direct_lora_parity_receipt)
    if file_sha256(path) != args.direct_lora_parity_sha256:
        raise ValueError("Direct LoRA parity receipt hash differs")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    arm = args.direct_lora_arm
    if receipt.get("schema_version") == "decision2-score-v7p-direct-lora-start/2":
        expected_source = (
            "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
        )
        arms = receipt.get("arms")
        if (
            receipt.get("status") != "PASS"
            or args.initial_model_sha256 != expected_source
            or receipt.get("source_model_sha256") != expected_source
            or not isinstance(receipt.get("roster_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["roster_sha256"])
            or receipt.get("roster_items") != 32
            or receipt.get("tolerance") != 1e-4
            or not isinstance(receipt.get("source_files_sha256"), dict)
            or not receipt["source_files_sha256"]
            or not isinstance(receipt.get("container_image_id"), str)
            or not receipt["container_image_id"].startswith("sha256:")
            or not isinstance(arms, dict)
            or set(arms) != {"A", "B", "C"}
            or arm not in arms
        ):
            raise ValueError("Direct LoRA v7p receipt or source differs")
        for name, result in arms.items():
            train_sha = result.get("train_sha256")
            pred_sha = result.get("prediction_sha256")
            drift = result.get("max_absolute_option_probability_drift")
            if (
                result.get("status") != "PASS"
                or not isinstance(train_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", train_sha)
                or not isinstance(pred_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", pred_sha)
                or result.get("same_argmax") != 32
                or type(drift) not in (int, float)
                or not math.isfinite(drift)
                or drift > 1e-4
            ):
                raise ValueError(f"Direct LoRA v7p zero-step arm {name} failed parity")
        if arms["C"]["train_sha256"] != arms["A"]["train_sha256"]:
            raise ValueError("Direct LoRA v7p objective arm changed TRAIN data")
        if file_sha256(args.train) != arms[arm]["train_sha256"]:
            raise ValueError("Direct LoRA v7p frozen arm TRAIN differs")
        return
    expected_train_sha = {
        "A": "6a6ef7d3f2eac2a63cdd61cd806275e67c0aa772e78b45bf0953200f2a776235",
        "B": "1c705c9a8271ce2e526b6bc91affe18d463a52bb86ce99b6b1bcb5007d467b41",
    }
    if (
        receipt.get("schema_version") != "decision2-score-en-direct-lora-start-parity/1"
        or receipt.get("status") != "PASS"
        or args.initial_model_sha256
        != "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
        or receipt.get("source_model_sha256") != args.initial_model_sha256
        or receipt.get("roster_sha256")
        != "193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5"
        or receipt.get("tolerance") != 1e-4
        or file_sha256(args.train) != expected_train_sha[arm]
        or set(receipt.get("arms", {})) != {"A", "B"}
    ):
        raise ValueError("Direct LoRA parity receipt or frozen arm differs")
    for name, expected in expected_train_sha.items():
        result = receipt["arms"][name]
        if (
            result.get("status") != "PASS"
            or result.get("train_sha256") != expected
            or result.get("same_argmax") != 32
            or type(result.get("max_absolute_option_probability_drift"))
            not in (int, float)
            or not math.isfinite(result["max_absolute_option_probability_drift"])
            or result["max_absolute_option_probability_drift"] > 1e-4
        ):
            raise ValueError(f"Direct LoRA zero-step arm {name} failed parity")


def main() -> None:
    args = parse_args()
    validate_args(args)
    verify_direct_lora_parity_gate(args)
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
    weighted_train_ids = {
        row["id"]
        for row in train_rows
        if row["task_type"] == "choice" and row["source"] in args.choice_source
    }
    if args.choice_source and (
        not weighted_train_ids
        or set(args.choice_source) - {row["source"] for row in train_rows}
    ):
        raise ValueError("Choice source roster is empty or contains absent source")
    train_weights = [
        args.choice_source_weight if row["id"] in weighted_train_ids else 1.0
        for row in train_rows
    ]
    data_sha = {
        "train": file_sha256(args.train),
        "select": file_sha256(args.select),
        "cal": file_sha256(args.cal),
    }
    if args.head_variant in ("type-separated", "candidate-interaction"):
        if data_sha != TYPED_HEAD_PARTITIONS:
            raise ValueError("Experimental-head ablation data differs")
        if (
            not resume
            and {
                name: file_sha256(Path(args.model_path) / name)
                for name in TYPED_HEAD_SOURCE_FILES
            }
            != TYPED_HEAD_SOURCE_FILES
        ):
            raise ValueError("Experimental-head ablation official source differs")
        if (len(train_rows), len(select_rows), len(cal_rows)) != (7455, 700, 700):
            raise ValueError("Experimental-head ablation partition counts differ")
    if args.replay:
        data_sha["replay"] = file_sha256(args.replay)
    if args.inline_teacher:
        data_sha["inline_teacher"] = file_sha256(args.inline_teacher)
    code_files = (
        (*SOURCE_FILES, "lora.py") if args.train_mode == "lora" else SOURCE_FILES
    )
    if args.head_variant == "type-separated":
        code_files = (*code_files, "type_separated_head.py")
    elif args.head_variant == "candidate-interaction":
        code_files = (*code_files, "candidate_interaction_head.py")
    if args.init_kind == "decision2-lora":
        code_files = (*code_files, "infer.py")
    if args.inline_teacher:
        code_files = (*code_files, "inline_replay.py")
    source_code_sha = {
        name: file_sha256(Path(__file__).with_name(name)) for name in code_files
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    initial_model_identity = None
    if resume:
        prior = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        source = prior["model_source"]
        model, tokenizer = DecisionModel.from_checkpoint(
            resume,
            source_path=args.source_path,
            trainable_adapter=args.train_mode == "lora",
        )
        if model.metadata.get("head_variant", "shared") != args.head_variant:
            raise ValueError("Resume checkpoint head variant differs from the run")
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
        if (
            args.init_kind == "decision2-lora"
            and model.metadata.get("continuation_origin", {}).get(
                "initial_model_sha256"
            )
            != args.initial_model_sha256
        ):
            raise ValueError("Direct LoRA resume lost its original adapter identity")
    else:
        if args.init_kind == "decision2-lora":
            model, tokenizer, initial_model_identity = load_direct_lora_start(args)
            source = model.metadata["lora"]["source_fingerprint"]
        elif args.init_kind in ("base", "posttrained"):
            source = source_fingerprint(Path(args.model_path))
            model, tokenizer = DecisionModel.from_base(
                args.model_path,
                args.base_revision,
                args.head_dim,
                source_stage=args.init_kind,
                head_variant=args.head_variant,
            )
        elif args.init_kind == "decision1":
            source = source_fingerprint(Path(args.model_path))
            model, tokenizer = DecisionModel.from_decision1(
                args.model_path, args.head_dim
            )
        else:
            source = source_fingerprint(Path(args.model_path))
            model, tokenizer = DecisionModel.from_checkpoint(args.model_path)
            if model.metadata.get("head_variant", "shared") != args.head_variant:
                raise ValueError("Initialization checkpoint head variant differs")
            if model.metadata["head_dim"] != args.head_dim:
                raise ValueError(
                    "--head-dim must match the Decision 2.0 initialization checkpoint"
                )
        if args.train_mode == "lora" and args.init_kind != "decision2-lora":
            attach_lora(
                model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                source_kind=args.init_kind,
                source_fingerprint=source,
            )
    inline_teacher_count = 0
    if args.inline_teacher:
        inline_teacher_count = attach_inline_teacher(
            args.inline_teacher,
            train_rows,
            train_sha256=data_sha["train"],
            source_files_sha256=source["files_sha256"],
            expected_source_model_sha256=args.inline_teacher_source_model_sha256,
            expected_materialization_receipt_sha256=args.inline_teacher_source_receipt_sha256,
            expected_roster_sha256=args.inline_teacher_roster_sha256,
            expected_control_baseline_sha256=args.inline_teacher_control_sha256,
            expected_parity_roster_sha256=args.inline_teacher_parity_roster_sha256,
        )
    model = model.float().to(device)
    if args.train_mode == "head":
        model.backbone.requires_grad_(False)
    elif args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.backbone.config.use_cache = False
    loss_version = ORDINAL_LOSS_VERSION if args.ordinal_weight else LOSS_VERSION
    model.metadata.update(
        {"training_mode": args.train_mode, "loss_version": loss_version}
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
    if (
        args.head_variant in ("type-separated", "candidate-interaction")
        and sum(train_lengths) != 4_094_489
    ):
        raise ValueError("Experimental-head ablation tokenizer exposure differs")
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
        "loss_version": loss_version,
        "model_source": source,
        "data_sha256": data_sha,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "zero_step_only": args.zero_step_only,
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
        **direct_lora_contract_fields(args),
        "base_revision": args.base_revision,
        "objective": args.objective,
        "brier_weight": args.brier_weight,
        **({"ordinal_weight": args.ordinal_weight} if args.ordinal_weight else {}),
        "replay_fraction": args.replay_fraction,
        "replay_kl_weight": args.replay_kl_weight,
        "choice_source": args.choice_source,
        "choice_source_weight": args.choice_source_weight,
        "weighted_choice_count": len(weighted_train_ids),
        **(
            {
                "inline_teacher_roster_sha256": args.inline_teacher_roster_sha256,
                "inline_teacher_control_sha256": args.inline_teacher_control_sha256,
                "inline_teacher_parity_roster_sha256": args.inline_teacher_parity_roster_sha256,
                "inline_teacher_source_model_sha256": args.inline_teacher_source_model_sha256,
                "inline_teacher_source_receipt_sha256": args.inline_teacher_source_receipt_sha256,
                "inline_teacher_count": inline_teacher_count,
            }
            if args.inline_teacher
            else {}
        ),
        "train_mode": args.train_mode,
        "planned_updates": planned,
        "train_count": len(train_items),
        "replay_pool_count": len(replay_items),
    }
    if args.head_variant != "shared":
        contract["head_variant"] = args.head_variant
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
                **(
                    {"initial_model_identity": initial_model_identity}
                    if initial_model_identity is not None
                    else {}
                ),
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
    if args.zero_step_only:
        log({"event": "zero_step_only", "step": 0})
        metrics_file.close()
        return
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
            window_weight = sum(
                train_weights[index] if source_name == "train" else 1.0
                for batch_ids in window_batches
                for source_name, index in batch_ids
            )
            factor = learning_factor(step, planned, args.warmup_ratio)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            sums = dict.fromkeys(("total", "ce", "brier", "replay_kl", "ordinal"), 0.0)
            weighted_total = 0.0
            correct = tokens = replay_seen = 0
            started = time.perf_counter()
            for batch_ids in window_batches:
                item_weights = torch.tensor(
                    [
                        train_weights[index] if source_name == "train" else 1.0
                        for source_name, index in batch_ids
                    ],
                    device=device,
                    dtype=torch.float32,
                )
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
                        task_type_ids=batch["task_type_ids"],
                        score_level_indices=batch["score_level_indices"],
                        ordinal_weight=args.ordinal_weight,
                    )
                    loss = (terms["total"] * item_weights).sum() / window_weight
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                weighted_total += (terms["total"].detach() * item_weights).sum().item()
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
                    "weighted_choice_examples": sum(
                        source_name == "train" and train_weights[index] > 1.0
                        for batch_ids in window_batches
                        for source_name, index in batch_ids
                    ),
                    "replay_examples": replay_seen,
                    "loss": weighted_total / window_weight,
                    "unweighted_loss": sums["total"] / window_count,
                    "ce": sums["ce"] / window_count,
                    "brier": sums["brier"] / window_count,
                    "replay_kl": sums["replay_kl"] / max(1, replay_seen),
                    "ordinal": sums["ordinal"] / window_count,
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
