"""One projected update, native zero-step parity and package reload gate.

This opt-in diagnostic is bounded to one GPU update under the frozen 466-step
schedule. It writes a private gold-free receipt and one reloadable checkpoint;
it does not run SELECT scoring or a full projected-gradient treatment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import digest, file_sha256
from .decision_model import DecisionModel
from .gradient_conflict_preflight import SEED, SOURCE_REVISION
from .gradient_projection import PROJECTION_VERSION, TaskGradientAccumulator
from .gradient_projection_parity import (
    PLANNED_UPDATES,
    SELECT_PROBES,
    _input_manifest,
    _loss,
    _predictions,
    validate_arm,
    validate_plan,
)
from .loss import LOSS_VERSION
from .source import source_fingerprint
from .train import TYPED_HEAD_PARTITIONS, learning_factor

SCHEMA = "decision2-06b-projected-one-update-preflight/1"
MAX_PROBABILITY_DRIFT = 1e-5
CODE_FILES = (
    "gradient_projection_preflight.py",
    "gradient_projection.py",
    "gradient_projection_parity.py",
    "decision_model.py",
    "data.py",
    "loss.py",
    "plan.py",
    "source.py",
    "train.py",
    "train_gradient_projection.py",
)
CONTROL_TRAIN_SHA256 = (
    "0111bbfd0e6a372661a88c44e0c716c18a94b78f651c0662fa2b18bbe59ad96d"
)


def _vector_diff(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(left) != SELECT_PROBES or len(right) != SELECT_PROBES:
        raise ValueError("Native parity requires the fixed 32 SELECT inputs")
    changed = 0
    maximum = 0.0
    for a, b in zip(left, right):
        if a["token_ids_sha256"] != b["token_ids_sha256"] or len(
            a["probabilities"]
        ) != len(b["probabilities"]):
            raise ValueError("Native parity prompt roster or option width differs")
        changed += a["prediction_index"] != b["prediction_index"]
        maximum = max(
            maximum,
            *(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"])),
        )
    return {"categorical_changes": changed, "max_probability_diff": maximum}


def run(
    *,
    train: Path,
    select: Path,
    cal: Path,
    source: Path,
    plan_path: Path,
    ordinary_path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    import torch

    if (
        checkpoint.exists()
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Need a fresh checkpoint path and one BF16 GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Projected one-update preflight is single-device")
    if file_sha256(cal) != TYPED_HEAD_PARTITIONS["cal"]:
        raise ValueError("Frozen CAL hash differs; no CAL contents are read")
    saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(saved_plan)
    ordinary = json.loads(ordinary_path.read_text(encoding="utf-8"))
    validate_arm(ordinary)
    if ordinary["arm"] != "ordinary" or ordinary["plan_sha256"] != file_sha256(
        plan_path
    ):
        raise ValueError("Need the independent ordinary arm from this sealed plan")
    plan, items, probes, tokenizer = _input_manifest(train, select, source)
    if saved_plan["code_sha256"]["train.py"] != CONTROL_TRAIN_SHA256:
        raise ValueError("Ordinary control was not the frozen trainer source")
    if plan["code_sha256"]["train.py"] != CONTROL_TRAIN_SHA256:
        raise ValueError("Current ordinary trainer differs from the frozen source")
    unchanged_files = set(saved_plan["code_sha256"]) - {"train.py"}
    if any(
        saved_plan["code_sha256"][name] != plan["code_sha256"][name]
        for name in unchanged_files
    ):
        raise ValueError("Non-trainer parity code changed after ordinary arm")
    if {k: v for k, v in plan.items() if k not in ("created_utc", "code_sha256")} != {
        k: v for k, v in saved_plan.items() if k not in ("created_utc", "code_sha256")
    }:
        raise ValueError("Frozen source, data or schedule changed after parity")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    device = torch.device("cuda:0")
    model, reloaded_tokenizer = DecisionModel.from_base(
        source, SOURCE_REVISION, 256, source_stage="base", head_variant="shared"
    )
    if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Tokenizer vocabulary differs from sealed plan")
    model = model.float().to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.metadata.update(
        {
            "training_mode": "full",
            "loss_version": LOSS_VERSION,
            "gradient_projection_version": PROJECTION_VERSION,
        }
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no padding or EOS token")
    zero_step = _predictions(model, probes, pad_id, device)
    zero_vs_ordinary = _vector_diff(ordinary["zero_step"], zero_step)
    if (
        zero_vs_ordinary["categorical_changes"]
        or zero_vs_ordinary["max_probability_diff"] > MAX_PROBABILITY_DRIFT
    ):
        raise ValueError("Projected source differs from independent zero-step control")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": 2e-5,
                "peak_lr": 2e-5,
                "name": "backbone",
            },
            {
                "params": list(model.head.parameters()),
                "lr": 2e-4,
                "peak_lr": 2e-4,
                "name": "head",
            },
        ],
        weight_decay=0.01,
        foreach=True,
    )
    factor = learning_factor(0, PLANNED_UPDATES, 0.05)
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * factor
    accumulator = TaskGradientAccumulator(model)
    model.train()
    for item in items:
        optimizer.zero_grad(set_to_none=True)
        _loss(model, item, pad_id, device, len(items)).backward()
        accumulator.capture(item["task_type"])
    optimizer.zero_grad(set_to_none=True)
    projection = accumulator.finalize(enabled=True)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise ValueError("Nonfinite projected global gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    post_step = _predictions(model, probes, pad_id, device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint, tokenizer)
    checkpoint_files = source_fingerprint(checkpoint)["files_sha256"]
    del accumulator, optimizer, model
    torch.cuda.empty_cache()
    model, reload_tokenizer = DecisionModel.from_checkpoint(checkpoint)
    if reload_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Saved checkpoint tokenizer differs")
    model = model.float().to(device)
    reload_vectors = _predictions(model, probes, pad_id, device)
    reload_diff = _vector_diff(post_step, reload_vectors)
    if (
        reload_diff["categorical_changes"]
        or reload_diff["max_probability_diff"] > MAX_PROBABILITY_DRIFT
    ):
        raise ValueError("Projected one-step package reload drift exceeds gate")
    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASS_PROJECTED_ONE_UPDATE_PREFLIGHT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": SOURCE_REVISION,
        "train_sha256": saved_plan["train_sha256"],
        "select_sha256": saved_plan["select_sha256"],
        "cal_sha256": TYPED_HEAD_PARTITIONS["cal"],
        "select32_sha256": saved_plan["select32_sha256"],
        "plan_sha256": file_sha256(plan_path),
        "ordinary_receipt_sha256": file_sha256(ordinary_path),
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name)) for name in CODE_FILES
        },
        "ordinary_train_code_sha256": CONTROL_TRAIN_SHA256,
        "projection_version": PROJECTION_VERSION,
        "planned_updates": PLANNED_UPDATES,
        "optimizer_steps": 1,
        "zero_vs_ordinary": zero_vs_ordinary,
        "post_vs_reload": reload_diff,
        "projection": projection,
        "global_gradient_norm": float(gradient_norm.item()),
        "checkpoint_files_sha256": checkpoint_files,
        "checkpoint_files_digest": digest(checkpoint_files),
        "device_seconds": elapsed,
        "device_gpu_hours": elapsed / 3600,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any]) -> None:
    fields = {
        "schema_version",
        "status",
        "created_utc",
        "source_revision",
        "train_sha256",
        "select_sha256",
        "cal_sha256",
        "select32_sha256",
        "plan_sha256",
        "ordinary_receipt_sha256",
        "code_sha256",
        "ordinary_train_code_sha256",
        "projection_version",
        "planned_updates",
        "optimizer_steps",
        "zero_vs_ordinary",
        "post_vs_reload",
        "projection",
        "global_gradient_norm",
        "checkpoint_files_sha256",
        "checkpoint_files_digest",
        "device_seconds",
        "device_gpu_hours",
    }
    if (
        set(receipt) != fields
        or receipt["schema_version"] != SCHEMA
        or receipt["status"] != "PASS_PROJECTED_ONE_UPDATE_PREFLIGHT"
    ):
        raise ValueError("Projected one-step receipt has unexpected fields")
    if (
        receipt["source_revision"] != SOURCE_REVISION
        or receipt["cal_sha256"] != TYPED_HEAD_PARTITIONS["cal"]
        or receipt["projection_version"] != PROJECTION_VERSION
        or receipt["ordinary_train_code_sha256"] != CONTROL_TRAIN_SHA256
        or receipt["planned_updates"] != PLANNED_UPDATES
        or receipt["optimizer_steps"] != 1
        or any(
            not isinstance(receipt[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt[key])
            for key in (
                "train_sha256",
                "select_sha256",
                "select32_sha256",
                "plan_sha256",
                "ordinary_receipt_sha256",
                "checkpoint_files_digest",
            )
        )
        or set(receipt["code_sha256"]) != set(CODE_FILES)
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in receipt["code_sha256"].values()
        )
        or not isinstance(receipt["checkpoint_files_sha256"], dict)
        or not receipt["checkpoint_files_sha256"]
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
            or ".." in name
            or name.startswith("/")
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in receipt["checkpoint_files_sha256"].items()
        )
        or digest(receipt["checkpoint_files_sha256"])
        != receipt["checkpoint_files_digest"]
    ):
        raise ValueError("Projected one-step receipt differs from frozen contract")
    for comparison in ("zero_vs_ordinary", "post_vs_reload"):
        value = receipt[comparison]
        if (
            set(value) != {"categorical_changes", "max_probability_diff"}
            or value["categorical_changes"] != 0
            or type(value["max_probability_diff"]) not in (int, float)
            or not math.isfinite(value["max_probability_diff"])
            or value["max_probability_diff"] > MAX_PROBABILITY_DRIFT
        ):
            raise ValueError("Projected one-step native parity gate failed")
    for key in ("global_gradient_norm", "device_seconds", "device_gpu_hours"):
        if (
            type(receipt[key]) not in (int, float)
            or not math.isfinite(receipt[key])
            or receipt[key] < 0
        ):
            raise ValueError("Projected one-step numeric receipt is invalid")
    summary = receipt["projection"]
    if (
        set(summary)
        != {
            "projection_version",
            "enabled",
            "task_counts",
            "task_norms",
            "pairwise_cosines",
            "projected_pairs",
            "ordinary_backbone_norm",
            "pre_match_backbone_norm",
            "norm_scale",
            "final_backbone_norm",
        }
        or summary["projection_version"] != PROJECTION_VERSION
        or summary["enabled"] is not True
        or set(summary["task_counts"]) != {"choice", "noul", "score"}
        or any(
            type(count) is not int or count < 0
            for count in summary["task_counts"].values()
        )
        or sum(summary["task_counts"].values()) != 16
        or set(summary["task_norms"]) != {"choice", "noul", "score"}
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in summary["task_norms"].values()
        )
        or set(summary["pairwise_cosines"])
        != {"choice_noul", "choice_score", "noul_score"}
        or any(
            value is not None
            and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not -1 <= value <= 1
            )
            for value in summary["pairwise_cosines"].values()
        )
        or summary["projected_pairs"] < 0
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (
                summary["ordinary_backbone_norm"],
                summary["pre_match_backbone_norm"],
                summary["norm_scale"],
                summary["final_backbone_norm"],
            )
        )
    ):
        raise ValueError("Projected one-step summary is invalid")
    if abs(summary["final_backbone_norm"] - summary["ordinary_backbone_norm"]) > max(
        1e-5, summary["ordinary_backbone_norm"] * 1e-5
    ):
        raise ValueError("Projected one-step norm matching failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--select", required=True, type=Path)
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--ordinary", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint.exists():
        parser.error("One-step output already exists; preserve the previous result")
    receipt = run(
        train=args.train,
        select=args.select,
        cal=args.cal,
        source=args.model_path,
        plan_path=args.plan,
        ordinary_path=args.ordinary,
        checkpoint=args.checkpoint,
    )
    pending = args.output.with_name(args.output.name + ".pending")
    if pending.exists():
        parser.error("Pending one-step output exists; preserve it")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)


if __name__ == "__main__":
    main()
