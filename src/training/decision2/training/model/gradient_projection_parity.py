"""Opt-in, one-update parity gate for the frozen 0.6B gradient arm.

The ordinary arm independently reproduces the trainer's first optimizer
window with the full 466-update LR schedule. The grouped arm only collects
backbone gradients by task type, then sums them without projection. Neither
arm reads formal/public evaluation data or scores SELECT labels. The output
contains hashes and gold-free probability vectors for exactly 32 SELECT
prompts; ``compare`` emits only aggregate parity statistics.
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

from .data import (
    MAX_OPTIONS,
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)
from .decision_model import PROMPT_VERSION, TASK_TYPES, DecisionModel, collate, encode
from .gradient_conflict_preflight import (
    ACCUMULATION,
    MAX_LENGTH,
    SEED,
    SOURCE_FILES,
    SOURCE_REVISION,
    TRAIN_COUNT,
    TRAIN_SHA256,
    TRAIN_TOKENS,
)
from .loss import per_example_loss
from .plan import epoch_batches, planned_updates
from .source import source_fingerprint
from .train import learning_factor

SCHEMA = "decision2-06b-gradient-projection-parity/1"
SELECT_SHA256 = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
SELECT_COUNT = 700
SELECT_PROBES = 32
PLANNED_UPDATES = 466
MAX_PROBABILITY_DRIFT = 1e-5
CODE_FILES = (
    "gradient_projection_parity.py",
    "data.py",
    "decision_model.py",
    "loss.py",
    "plan.py",
    "source.py",
    "train.py",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.with_name(path.name + ".pending").exists():
        raise ValueError("Existing receipt must be preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _code_hashes() -> dict[str, str]:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in CODE_FILES}


def _input_manifest(
    train_path: Path, select_path: Path, source: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Any]:
    """Freeze all training exposure before constructing a model or optimizer."""
    from transformers import AutoTokenizer

    if file_sha256(train_path) != TRAIN_SHA256:
        raise ValueError("Frozen TRAIN hash differs")
    if file_sha256(select_path) != SELECT_SHA256:
        raise ValueError("Frozen SELECT hash differs")
    if {name: file_sha256(source / name) for name in SOURCE_FILES} != SOURCE_FILES:
        raise ValueError("Official Qwen source files differ")
    all_source_hashes = source_fingerprint(source)["files_sha256"]
    train_rows = load_partition(train_path, "train")
    select_rows = load_partition(select_path, "select")
    check_partition_isolation({"train": train_rows, "select": select_rows})
    if len(train_rows) != TRAIN_COUNT or len(select_rows) != SELECT_COUNT:
        raise ValueError("Frozen partition row count differs")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    train = [encode(row, tokenizer, MAX_LENGTH) for row in train_rows]
    select = [encode(row, tokenizer, MAX_LENGTH) for row in select_rows[:SELECT_PROBES]]
    lengths = [len(item["ids"]) for item in train]
    if sum(lengths) != TRAIN_TOKENS or max(lengths) > MAX_LENGTH:
        raise ValueError("Frozen TRAIN token exposure differs")
    if (
        planned_updates(TRAIN_COUNT, 0, 0.0, 1, ACCUMULATION, 1, None)
        != PLANNED_UPDATES
    ):
        raise ValueError("Full control update schedule differs")
    batches = epoch_batches(
        lengths, [], epoch=0, seed=SEED, microbatch=1, replay_fraction=0.0
    )
    order = [
        index
        for batch in batches
        for source_name, index in batch
        if source_name == "train"
    ]
    if len(order) != TRAIN_COUNT or set(order) != set(range(TRAIN_COUNT)):
        raise ValueError("Trainer schedule omitted or duplicated TRAIN rows")
    first = [index for batch in batches[:ACCUMULATION] for _, index in batch]
    if len(first) != ACCUMULATION or any(
        source_name != "train"
        for batch in batches[:ACCUMULATION]
        for source_name, _ in batch
    ):
        raise ValueError("First optimizer window differs from trainer")
    first_items = [train[index] for index in first]
    counts = {
        kind: sum(item["task_type"] == kind for item in first_items)
        for kind in TASK_TYPES
    }
    # All SELECT gold labels are replaced before the inference path is entered.
    for item in select:
        item["label"] = 0
    plan = {
        "schema_version": SCHEMA,
        "status": "PLAN_ONLY",
        "created_utc": _utc_now(),
        "source_revision": SOURCE_REVISION,
        "source_files_sha256": SOURCE_FILES,
        "source_all_files_sha256": all_source_hashes,
        "code_sha256": _code_hashes(),
        "train_sha256": TRAIN_SHA256,
        "select_sha256": SELECT_SHA256,
        "train_rows": TRAIN_COUNT,
        "train_native_tokens": TRAIN_TOKENS,
        "select_probes": SELECT_PROBES,
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "accumulation": ACCUMULATION,
        "planned_updates": PLANNED_UPDATES,
        "first_step_learning_factor": learning_factor(0, PLANNED_UPDATES, 0.05),
        "schedule_sha256": digest(order),
        "first_window_sha256": digest(
            [(item["token_ids_sha256"], item["task_type"]) for item in first_items]
        ),
        "first_window_type_counts": counts,
        "select32_sha256": digest([item["token_ids_sha256"] for item in select]),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    validate_plan(plan)
    return plan, first_items, select, tokenizer


def validate_plan(plan: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "created_utc",
        "source_revision",
        "source_files_sha256",
        "source_all_files_sha256",
        "code_sha256",
        "train_sha256",
        "select_sha256",
        "train_rows",
        "train_native_tokens",
        "select_probes",
        "prompt_version",
        "seed",
        "accumulation",
        "planned_updates",
        "first_step_learning_factor",
        "schedule_sha256",
        "first_window_sha256",
        "first_window_type_counts",
        "select32_sha256",
        "optimizer_constructed",
        "optimizer_steps",
    }
    if set(plan) != expected or any(
        key in plan for key in ("label", "gold", "text", "prompt", "id", "path")
    ):
        raise ValueError("Plan receipt has unexpected fields")
    if (
        plan["schema_version"] != SCHEMA
        or plan["status"] != "PLAN_ONLY"
        or plan["source_revision"] != SOURCE_REVISION
        or plan["source_files_sha256"] != SOURCE_FILES
        or not isinstance(plan["source_all_files_sha256"], dict)
        or not plan["source_all_files_sha256"]
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            or not isinstance(value, str)
            or not HEX64.fullmatch(value)
            for name, value in plan["source_all_files_sha256"].items()
        )
        or any(
            plan["source_all_files_sha256"].get(name) != value
            for name, value in SOURCE_FILES.items()
        )
        or plan["train_sha256"] != TRAIN_SHA256
        or plan["select_sha256"] != SELECT_SHA256
        or plan["train_rows"] != TRAIN_COUNT
        or plan["train_native_tokens"] != TRAIN_TOKENS
        or plan["select_probes"] != SELECT_PROBES
        or plan["prompt_version"] != PROMPT_VERSION
        or plan["seed"] != SEED
        or plan["accumulation"] != ACCUMULATION
        or plan["planned_updates"] != PLANNED_UPDATES
        or plan["first_step_learning_factor"]
        != learning_factor(0, PLANNED_UPDATES, 0.05)
        or plan["optimizer_constructed"] is not False
        or plan["optimizer_steps"] != 0
        or not isinstance(plan["created_utc"], str)
        or not isinstance(plan["code_sha256"], dict)
        or set(plan["code_sha256"]) != set(CODE_FILES)
        or any(
            not isinstance(value, str) or not HEX64.fullmatch(value)
            for value in plan["code_sha256"].values()
        )
        or any(
            not isinstance(plan.get(key), str) or not HEX64.fullmatch(plan[key])
            for key in ("schedule_sha256", "first_window_sha256", "select32_sha256")
        )
        or set(plan["first_window_type_counts"]) != set(TASK_TYPES)
        or any(
            type(value) is not int or value < 0
            for value in plan["first_window_type_counts"].values()
        )
        or sum(plan["first_window_type_counts"].values()) != ACCUMULATION
    ):
        raise ValueError("Plan receipt differs from the frozen control")


def _loss(
    model: Any, item: dict[str, Any], pad_id: int, device: Any, divisor: int
) -> Any:
    import torch

    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in collate([item], pad_id).items()
    }
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        logits = model(**batch)
        terms = per_example_loss(
            logits,
            batch["labels"],
            batch["candidate_mask"],
            objective="ce_brier",
            brier_weight=0.5,
            teacher_probs=batch["teacher_probs"],
            replay_mask=batch["replay_mask"],
            replay_kl_weight=0.0,
            task_type_ids=batch["task_type_ids"],
            score_level_indices=batch["score_level_indices"],
            ordinal_weight=0.0,
        )
        loss = terms["total"].sum() / divisor
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Nonfinite one-update loss")
    return loss


def _require_finite_gradients(parameters: list[Any]) -> None:
    import torch

    if any(
        param.grad is not None and not bool(torch.isfinite(param.grad).all().item())
        for param in parameters
    ):
        raise ValueError("Nonfinite one-update gradient")


def ordinary_one_update(
    model: Any, optimizer: Any, items: list[dict[str, Any]], pad_id: int, device: Any
) -> float:
    """Independent ordinary-gradient control; same operations as train.py."""
    import torch

    optimizer.zero_grad(set_to_none=True)
    for item in items:
        _loss(model, item, pad_id, device, len(items)).backward()
    parameters = list(model.parameters())
    _require_finite_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise ValueError("Nonfinite ordinary clipped gradient norm")
    optimizer.step()
    return float(norm.item())


def grouped_one_update(
    model: Any, optimizer: Any, items: list[dict[str, Any]], pad_id: int, device: Any
) -> float:
    """Projection-disabled per-type accumulation; head retains ordinary sum."""
    import torch

    backbone = list(model.backbone.parameters())
    head = list(model.head.parameters())
    if (
        not backbone
        or not head
        or any(param.dtype != torch.float32 for param in backbone + head)
    ):
        raise ValueError("Frozen full-training path requires FP32 backbone and head")
    typed = {
        kind: [torch.zeros_like(param) for param in backbone] for kind in TASK_TYPES
    }
    head_sum = [torch.zeros_like(param) for param in head]
    backbone_used = [False] * len(backbone)
    head_used = [False] * len(head)
    for item in items:
        optimizer.zero_grad(set_to_none=True)
        _loss(model, item, pad_id, device, len(items)).backward()
        _require_finite_gradients(backbone + head)
        for position, param in enumerate(backbone):
            if param.grad is not None:
                typed[item["task_type"]][position].add_(param.grad)
                backbone_used[position] = True
        for position, param in enumerate(head):
            if param.grad is not None:
                head_sum[position].add_(param.grad)
                head_used[position] = True
    optimizer.zero_grad(set_to_none=True)
    for position, param in enumerate(backbone):
        if backbone_used[position]:
            result = typed[TASK_TYPES[0]][position]
            for kind in TASK_TYPES[1:]:
                result.add_(typed[kind][position])
            param.grad = result
    for position, param in enumerate(head):
        if head_used[position]:
            param.grad = head_sum[position]
    parameters = backbone + head
    _require_finite_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise ValueError("Nonfinite grouped clipped gradient norm")
    optimizer.step()
    return float(norm.item())


def _predictions(
    model: Any, items: list[dict[str, Any]], pad_id: int, device: Any
) -> list[dict[str, Any]]:
    import torch

    was_training = model.training
    was_backbone_training = model.backbone.training
    model.eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(items), 2):
            batch_items = items[offset : offset + 2]
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in collate(batch_items, pad_id).items()
            }
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(**batch)
            vectors = logits.float().softmax(-1).cpu().tolist()
            for item, all_values in zip(batch_items, vectors):
                probs = all_values[: len(item["keys"])]
                if any(not math.isfinite(value) for value in probs):
                    raise ValueError("Nonfinite SELECT probability")
                output.append(
                    {
                        "token_ids_sha256": item["token_ids_sha256"],
                        "prediction_index": max(
                            range(len(probs)), key=probs.__getitem__
                        ),
                        "probabilities": probs,
                    }
                )
    if was_training:
        model.train()
        model.backbone.train(was_backbone_training)
    return output


def run_arm(
    arm: str, plan_path: Path, train: Path, select: Path, source: Path
) -> dict[str, Any]:
    import torch

    if arm not in ("ordinary", "grouped"):
        raise ValueError("Parity arm must be ordinary or grouped")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("One BF16 CUDA/ROCm device is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One-update parity must be single-device")
    saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(saved_plan)
    current_plan, items, probes, tokenizer = _input_manifest(train, select, source)
    if {k: v for k, v in saved_plan.items() if k != "created_utc"} != {
        k: v for k, v in current_plan.items() if k != "created_utc"
    }:
        raise ValueError(
            "Current source/data/code/schedule differs from sealed CPU plan"
        )
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    started = time.monotonic()
    model, reloaded_tokenizer = DecisionModel.from_base(
        source, SOURCE_REVISION, 256, source_stage="base", head_variant="shared"
    )
    if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Official tokenizer vocabulary differs")
    model = model.float().to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no padding or EOS token")
    zero_step = _predictions(model, probes, pad_id, device)
    model.train()
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
    update = ordinary_one_update if arm == "ordinary" else grouped_one_update
    norm = update(model, optimizer, items, pad_id, device)
    torch.cuda.synchronize(device)
    post_step = _predictions(model, probes, pad_id, device)
    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": SCHEMA,
        "status": "ARM_COMPLETE",
        "created_utc": _utc_now(),
        "arm": arm,
        "plan_sha256": file_sha256(plan_path),
        "code_sha256": _code_hashes(),
        "source_revision": SOURCE_REVISION,
        "train_sha256": TRAIN_SHA256,
        "select_sha256": SELECT_SHA256,
        "select32_sha256": saved_plan["select32_sha256"],
        "planned_updates": PLANNED_UPDATES,
        "learning_factor": factor,
        "optimizer_steps": 1,
        "unclipped_gradient_norm": norm,
        "finite_gradients": True,
        "zero_step": zero_step,
        "post_step": post_step,
        "device_seconds": elapsed,
        "device_gpu_hours": elapsed / 3600,
    }
    validate_arm(receipt)
    return receipt


def validate_arm(receipt: dict[str, Any]) -> None:
    fields = {
        "schema_version",
        "status",
        "created_utc",
        "arm",
        "plan_sha256",
        "code_sha256",
        "source_revision",
        "train_sha256",
        "select_sha256",
        "select32_sha256",
        "planned_updates",
        "learning_factor",
        "optimizer_steps",
        "unclipped_gradient_norm",
        "finite_gradients",
        "zero_step",
        "post_step",
        "device_seconds",
        "device_gpu_hours",
    }
    if (
        set(receipt) != fields
        or receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "ARM_COMPLETE"
    ):
        raise ValueError("Arm receipt has unexpected fields")
    if (
        receipt["arm"] not in ("ordinary", "grouped")
        or any(
            not isinstance(receipt[key], str) or not HEX64.fullmatch(receipt[key])
            for key in (
                "plan_sha256",
                "train_sha256",
                "select_sha256",
                "select32_sha256",
            )
        )
        or receipt["source_revision"] != SOURCE_REVISION
        or receipt["train_sha256"] != TRAIN_SHA256
        or receipt["select_sha256"] != SELECT_SHA256
        or receipt["planned_updates"] != PLANNED_UPDATES
        or receipt["learning_factor"] != learning_factor(0, PLANNED_UPDATES, 0.05)
        or receipt["optimizer_steps"] != 1
        or receipt["finite_gradients"] is not True
        or not isinstance(receipt["code_sha256"], dict)
        or set(receipt["code_sha256"]) != set(CODE_FILES)
        or any(
            not isinstance(v, str) or not HEX64.fullmatch(v)
            for v in receipt["code_sha256"].values()
        )
        or any(
            type(receipt[key]) not in (int, float)
            or not math.isfinite(receipt[key])
            or receipt[key] < 0
            for key in ("unclipped_gradient_norm", "device_seconds", "device_gpu_hours")
        )
    ):
        raise ValueError("Arm receipt differs from frozen one-update contract")
    for phase in ("zero_step", "post_step"):
        vectors = receipt[phase]
        if not isinstance(vectors, list) or len(vectors) != SELECT_PROBES:
            raise ValueError("Arm receipt lacks the fixed 32 probes")
        for row in vectors:
            if set(row) != {"token_ids_sha256", "prediction_index", "probabilities"}:
                raise ValueError("Arm receipt contains raw or gold data")
            probs = row["probabilities"]
            if (
                not isinstance(row["token_ids_sha256"], str)
                or not HEX64.fullmatch(row["token_ids_sha256"])
                or not isinstance(probs, list)
                or not 2 <= len(probs) <= MAX_OPTIONS
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in probs
                )
                or abs(sum(probs) - 1) > 1e-5
                or type(row["prediction_index"]) is not int
                or not 0 <= row["prediction_index"] < len(probs)
                or row["prediction_index"]
                != max(range(len(probs)), key=probs.__getitem__)
            ):
                raise ValueError("Arm receipt contains invalid probability vectors")
        if (
            digest([row["token_ids_sha256"] for row in vectors])
            != receipt["select32_sha256"]
        ):
            raise ValueError("Arm receipt SELECT probe roster differs")


def compare_arms(ordinary: dict[str, Any], grouped: dict[str, Any]) -> dict[str, Any]:
    validate_arm(ordinary)
    validate_arm(grouped)
    if ordinary["arm"] != "ordinary" or grouped["arm"] != "grouped":
        raise ValueError("Comparison requires independent ordinary and grouped arms")
    for field in (
        "plan_sha256",
        "code_sha256",
        "source_revision",
        "train_sha256",
        "select_sha256",
        "select32_sha256",
        "planned_updates",
        "learning_factor",
    ):
        if ordinary[field] != grouped[field]:
            raise ValueError("Independent parity arms used different inputs or code")
    differences = {}
    for phase in ("zero_step", "post_step"):
        changed = 0
        maximum = 0.0
        for control, treatment in zip(ordinary[phase], grouped[phase]):
            if control["token_ids_sha256"] != treatment["token_ids_sha256"] or len(
                control["probabilities"]
            ) != len(treatment["probabilities"]):
                raise ValueError("Independent parity arms have different SELECT probes")
            changed += control["prediction_index"] != treatment["prediction_index"]
            maximum = max(
                maximum,
                *(
                    abs(a - b)
                    for a, b in zip(
                        control["probabilities"], treatment["probabilities"]
                    )
                ),
            )
        differences[phase] = {
            "categorical_changes": changed,
            "max_probability_diff": maximum,
        }
    result = {
        "schema_version": SCHEMA,
        "status": (
            "PASS_ONE_UPDATE_PARITY"
            if all(
                value["categorical_changes"] == 0
                and value["max_probability_diff"] <= MAX_PROBABILITY_DRIFT
                for value in differences.values()
            )
            else "HOLD_ONE_UPDATE_PARITY"
        ),
        "created_utc": _utc_now(),
        "ordinary_receipt_sha256": digest(ordinary),
        "grouped_receipt_sha256": digest(grouped),
        "source_revision": SOURCE_REVISION,
        "train_sha256": TRAIN_SHA256,
        "select_sha256": SELECT_SHA256,
        "select32_sha256": ordinary["select32_sha256"],
        "planned_updates": PLANNED_UPDATES,
        "optimizer_steps_per_arm": 1,
        "max_probability_drift_gate": MAX_PROBABILITY_DRIFT,
        "zero_step": differences["zero_step"],
        "post_step": differences["post_step"],
        "gpu_hours": ordinary["device_gpu_hours"] + grouped["device_gpu_hours"],
    }
    validate_comparison(result)
    return result


def validate_comparison(receipt: dict[str, Any]) -> None:
    fields = {
        "schema_version",
        "status",
        "created_utc",
        "ordinary_receipt_sha256",
        "grouped_receipt_sha256",
        "source_revision",
        "train_sha256",
        "select_sha256",
        "select32_sha256",
        "planned_updates",
        "optimizer_steps_per_arm",
        "max_probability_drift_gate",
        "zero_step",
        "post_step",
        "gpu_hours",
    }
    if set(receipt) != fields or receipt["schema_version"] != SCHEMA:
        raise ValueError("Comparison receipt has unexpected fields")
    if receipt["status"] not in ("PASS_ONE_UPDATE_PARITY", "HOLD_ONE_UPDATE_PARITY"):
        raise ValueError("Comparison status is invalid")
    if (
        any(
            not isinstance(receipt[key], str) or not HEX64.fullmatch(receipt[key])
            for key in (
                "ordinary_receipt_sha256",
                "grouped_receipt_sha256",
                "train_sha256",
                "select_sha256",
                "select32_sha256",
            )
        )
        or receipt["source_revision"] != SOURCE_REVISION
        or receipt["train_sha256"] != TRAIN_SHA256
        or receipt["select_sha256"] != SELECT_SHA256
        or receipt["planned_updates"] != PLANNED_UPDATES
        or receipt["optimizer_steps_per_arm"] != 1
        or receipt["max_probability_drift_gate"] != MAX_PROBABILITY_DRIFT
        or type(receipt["gpu_hours"]) not in (int, float)
        or not math.isfinite(receipt["gpu_hours"])
        or receipt["gpu_hours"] < 0
    ):
        raise ValueError("Comparison differs from frozen contract")
    for phase in ("zero_step", "post_step"):
        result = receipt[phase]
        if set(result) != {"categorical_changes", "max_probability_diff"} or (
            type(result["categorical_changes"]) is not int
            or not 0 <= result["categorical_changes"] <= SELECT_PROBES
            or type(result["max_probability_diff"]) not in (int, float)
            or not math.isfinite(result["max_probability_diff"])
            or result["max_probability_diff"] < 0
        ):
            raise ValueError("Comparison lacks finite aggregate parity statistics")
    expected_pass = all(
        receipt[phase]["categorical_changes"] == 0
        and receipt[phase]["max_probability_diff"] <= MAX_PROBABILITY_DRIFT
        for phase in ("zero_step", "post_step")
    )
    if (receipt["status"] == "PASS_ONE_UPDATE_PARITY") != expected_pass:
        raise ValueError("Comparison status disagrees with frozen gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("plan", "ordinary", "grouped", "compare"), default="plan"
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--select", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ordinary", type=Path)
    parser.add_argument("--grouped", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".pending").exists()
    ):
        parser.error("Existing output must be preserved")
    if args.phase == "compare":
        if not args.ordinary or not args.grouped:
            parser.error("Compare requires both independent arm receipts")
        result = compare_arms(
            json.loads(args.ordinary.read_text(encoding="utf-8")),
            json.loads(args.grouped.read_text(encoding="utf-8")),
        )
        # Use exact file hashes, including their distinct creation timestamps.
        result["ordinary_receipt_sha256"] = file_sha256(args.ordinary)
        result["grouped_receipt_sha256"] = file_sha256(args.grouped)
        validate_comparison(result)
    else:
        if not args.train or not args.select or not args.model_path:
            parser.error("Plan and arm phases require frozen TRAIN, SELECT and source")
        if args.phase == "plan":
            result, _, _, _ = _input_manifest(args.train, args.select, args.model_path)
        else:
            if not args.plan:
                parser.error("Each GPU arm requires the sealed CPU plan")
            result = run_arm(
                args.phase, args.plan, args.train, args.select, args.model_path
            )
    _write_new(args.output, result)


if __name__ == "__main__":
    main()
