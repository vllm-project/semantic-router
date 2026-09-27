"""Bounded native zero-step and one-update gate for official Qwen3-0.6B.

This is a source-initialization contrast with the completed official Base
shared-head arm. No benchmark gold or calibration labels are scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from training.model.data import canonical, file_sha256, load_partition
from training.model.decision_model import DecisionModel, collate, encode
from training.model.loss import per_example_loss
from training.model.plan import epoch_batches
from training.model.source import source_fingerprint
from training.model.train import learning_factor

from .preflight_qwen06_type_head import (
    _compare,
    _phase_status,
    _predict,
    _roster,
    _write_once,
)

SCHEMA = "decision2-qwen06-official-posttrained-technical-gate/1"
ROOT = Path(__file__).resolve().parents[1]


def _lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown posttrained source gate")
    for relative, expected in value["code_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen source differs: {relative}")
    return value


def _source_files(path: Path, expected: dict[str, str]) -> dict[str, Any]:
    for name, digest in expected.items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Official source file differs: {name}")
    return source_fingerprint(path)


def _run(
    lock_path: Path,
    audit_path: Path,
    audit_sha256: str,
    base: Path,
    posttrained: Path,
    train: Path,
    select: Path,
    cal: Path,
    output: Path,
) -> dict[str, Any]:
    lock = _lock(lock_path)
    if file_sha256(audit_path) != audit_sha256:
        raise ValueError("CPU exposure audit differs from reviewed receipt")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit.get("schema") != lock["audit_schema"]:
        raise ValueError("CPU exposure audit did not pass")
    if audit["posttrained_source"] != lock["posttrained_source"]:
        raise ValueError("Official source revision differs")
    if audit["base_source"] != lock["base_source"]:
        raise ValueError("Control source revision differs")
    if audit["input_mismatch_rows"] != {"train": 0, "select": 0, "cal": 0}:
        raise ValueError("Tokenizer input mismatch")
    for role in ("train", "select", "cal"):
        if (
            audit["input_token_lengths"][role]["base"]
            != audit["input_token_lengths"][role]["posttrained"]
        ):
            raise ValueError("Input token exposure differs")
        if (
            audit["input_token_ids_sha256"][role]["base"]
            != audit["input_token_ids_sha256"][role]["posttrained"]
        ):
            raise ValueError("Native token ID digest differs")
    if audit["input_token_lengths"]["train"]["posttrained"] != lock["train_tokens"]:
        raise ValueError("TRAIN token exposure changed")
    base_fingerprint = _source_files(base, lock["base_files_sha256"])
    post_fingerprint = _source_files(posttrained, lock["posttrained_files_sha256"])
    for role, path in (("train", train), ("select", select), ("cal", cal)):
        if file_sha256(path) != lock["partition_sha256"][role]:
            raise ValueError(f"Frozen {role} partition differs")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    if len(train_rows) != 7455 or len(select_rows) != 700:
        raise ValueError("Frozen row counts differ")
    for role, rows in (("train", train_rows), ("select", select_rows)):
        identity = hashlib.sha256(
            canonical([row["id"] for row in rows]).encode()
        ).hexdigest()
        if identity != audit["row_order_sha256"][role]:
            raise ValueError(f"Frozen {role} row order differs")
    selected = _roster(select_rows, lock["select32_by_type"])
    zero_rows = _roster(select_rows, dict.fromkeys(lock["select32_by_type"], 1))
    started = time.monotonic()
    torch.manual_seed(lock["seed"])
    torch.cuda.manual_seed_all(lock["seed"])
    control, _ = DecisionModel.from_base(
        base, lock["base_revision"], source_stage="base"
    )
    shared_head_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in control.head.state_dict().items()
    }
    del control

    torch.manual_seed(lock["seed"])
    torch.cuda.manual_seed_all(lock["seed"])
    treatment, tokenizer = DecisionModel.from_base(
        posttrained, lock["posttrained_revision"], source_stage="posttrained"
    )
    if (
        treatment.metadata["head_variant"] != "shared"
        or treatment.metadata["source_stage"] != "posttrained"
    ):
        raise ValueError("Wrong native source/readout path")
    if any(
        not torch.equal(treatment.head.state_dict()[name].cpu(), weight)
        for name, weight in shared_head_state.items()
    ):
        raise ValueError("Shared head initialization differs from Base control")
    if treatment.metadata["text_parameter_count"] != lock["loaded_backbone_parameters"]:
        raise ValueError("Loaded backbone parameter count differs")
    treatment = treatment.float().to("cuda:0")
    first = _predict(treatment, tokenizer, zero_rows, lock["max_length"])
    second = _predict(treatment, tokenizer, zero_rows, lock["max_length"])
    repeat = _compare(first, second)
    if any(abs(sum(row["probabilities"]) - 1) > 1e-5 for row in first):
        raise ValueError("Native decision probabilities do not sum to one")
    zero_seconds = time.monotonic() - started
    zero_status = _phase_status(
        repeat,
        expected_rows=3,
        max_drift=1e-4,
        elapsed_seconds=zero_seconds,
        seconds_cap=lock["zero_step_seconds_cap"],
    )
    zero_path = output / "zero-step-result.json"
    _write_once(
        zero_path,
        {
            "schema": SCHEMA + "/zero-step-result",
            "status": zero_status,
            "lock_sha256": file_sha256(lock_path),
            "source_audit_sha256": audit_sha256,
            "base_fingerprint": base_fingerprint,
            "posttrained_fingerprint": post_fingerprint,
            "shared_head_exact_match": True,
            "native_three_type_repeat": repeat,
            "gpu_elapsed_seconds": zero_seconds,
        },
    )
    if zero_status != "PASS":
        raise ValueError("Zero-step gate failed")

    one_started = time.monotonic()
    train_lengths = [
        len(encode(row, tokenizer, lock["max_length"])["ids"]) for row in train_rows
    ]
    if sum(train_lengths) != lock["train_tokens"]:
        raise ValueError("TRAIN length changed after source audit")
    first_window = epoch_batches(
        train_lengths,
        [],
        epoch=0,
        seed=lock["seed"],
        microbatch=1,
        replay_fraction=0.0,
    )[:16]
    chosen = [train_rows[index] for group in first_window for _, index in group]
    if len(chosen) != 16:
        raise ValueError("First control optimizer window differs")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad/EOS")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(treatment.backbone.parameters()),
                "lr": 2e-5 * learning_factor(0, 466, 0.05),
            },
            {
                "params": list(treatment.head.parameters()),
                "lr": 2e-4 * learning_factor(0, 466, 0.05),
            },
        ],
        weight_decay=0.01,
        foreach=True,
    )
    treatment.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    treatment.train()
    gradient_types = {}
    for kind in ("choice", "noul", "score"):
        row = next(row for row in train_rows if row["task_type"] == kind)
        item = encode(row, tokenizer, lock["max_length"])
        batch = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in collate([item], pad_id).items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = treatment(**batch)
            loss = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )["total"].mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite {kind} loss")
        loss.backward()
        gradients = [parameter.grad for parameter in treatment.head.parameters()]
        finite = all(
            gradient is None or torch.isfinite(gradient).all() for gradient in gradients
        )
        nonzero = any(
            gradient is not None and gradient.abs().sum() > 0 for gradient in gradients
        )
        if not finite or not nonzero:
            raise ValueError(f"Invalid {kind} shared-head gradient")
        gradient_types[kind] = {"finite": True, "nonzero": True}
    optimizer.zero_grad(set_to_none=True)

    losses = []
    for row in chosen:
        item = encode(row, tokenizer, lock["max_length"])
        batch = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in collate([item], pad_id).items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = treatment(**batch)
            loss = (
                per_example_loss(
                    logits,
                    batch["labels"],
                    batch["candidate_mask"],
                    objective="ce_brier",
                    brier_weight=0.5,
                )["total"].mean()
                / 16
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite first-window loss")
        loss.backward()
        losses.append(float(loss.detach().item()) * 16)
    gradient_norm = torch.nn.utils.clip_grad_norm_(treatment.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise ValueError("Nonfinite first-window gradient")
    optimizer.step()
    torch.cuda.synchronize()
    before = _predict(treatment, tokenizer, selected, lock["max_length"])
    checkpoint = output / "checkpoint-0000001"
    treatment.save(checkpoint, tokenizer)
    del treatment
    torch.cuda.empty_cache()
    restored, tokenizer = DecisionModel.from_checkpoint(checkpoint)
    restored = restored.float().to("cuda:0")
    after = _predict(restored, tokenizer, selected, lock["max_length"])
    comparison = _compare(before, after)
    one_seconds = time.monotonic() - one_started
    one_status = _phase_status(
        comparison,
        expected_rows=32,
        max_drift=lock["reload_max_drift"],
        elapsed_seconds=one_seconds,
        seconds_cap=lock["one_step_seconds_cap"],
    )
    one_path = output / "one-step-result.json"
    _write_once(
        one_path,
        {
            "schema": SCHEMA + "/one-step-result",
            "status": one_status,
            "lock_sha256": file_sha256(lock_path),
            "source_audit_sha256": audit_sha256,
            "gradient_types": gradient_types,
            "first_update": {
                "examples": 16,
                "mean_loss": sum(losses) / 16,
                "gradient_norm": float(gradient_norm),
            },
            "reload32": comparison,
            "gpu_elapsed_seconds": one_seconds,
        },
    )
    if one_status != "PASS":
        raise ValueError("One-step gate failed")
    result = {
        "schema": SCHEMA + "/gpu-result",
        "status": "PASS",
        "lock_sha256": file_sha256(lock_path),
        "source_audit_sha256": audit_sha256,
        "source_revision": lock["posttrained_revision"],
        "architecture": restored.metadata["architecture"],
        "loaded_parameters": sum(
            parameter.numel() for parameter in restored.parameters()
        ),
        "zero_step_receipt_sha256": file_sha256(zero_path),
        "one_step_receipt_sha256": file_sha256(one_path),
        "total_gpu_elapsed_seconds": time.monotonic() - started,
    }
    _write_once(output / "technical-gate-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-audit-sha256", required=True)
    for name in ("base", "posttrained", "train", "select", "cal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("GPU gate output must be absent")
    args.output.mkdir(parents=True, mode=0o700)
    try:
        result = _run(
            args.lock,
            args.source_audit,
            args.source_audit_sha256,
            args.base,
            args.posttrained,
            args.train,
            args.select,
            args.cal,
            args.output,
        )
    except Exception as exc:
        _write_once(
            args.output / "technical-gate-result.json",
            {
                "schema": SCHEMA + "/gpu-result",
                "status": "HOLD",
                "failure_class": type(exc).__name__,
                "lock_sha256": file_sha256(args.lock),
            },
        )
        raise
    print(json.dumps({"schema": result["schema"], "status": result["status"]}))


if __name__ == "__main__":
    main()
