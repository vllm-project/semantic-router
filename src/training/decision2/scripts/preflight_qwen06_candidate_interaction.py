"""Frozen CPU audit and bounded real-source gate for the 0.6B set head.

This script never runs a full training arm or reads development/formal labels.
Run its CPU audit first, review the immutable receipt, then give the GPU stage
the receipt SHA-256 in an isolated one-GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from training.model.data import (
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import (
    QWEN3_INTERACTION_ARCHITECTURE,
    DecisionModel,
    collate,
    encode,
)
from training.model.loss import per_example_loss
from training.model.plan import epoch_batches
from training.model.source import source_fingerprint
from training.model.train import learning_factor
from transformers import AutoTokenizer

from .preflight_qwen06_type_head import (
    _compare,
    _phase_status,
    _predict,
    _roster,
    _write_once,
)

SCHEMA = "decision2-qwen06-candidate-interaction-technical-gate/1"
ROOT = Path(__file__).resolve().parents[1]


def _lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown candidate-interaction technical gate")
    for relative, expected in value["code_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen source differs: {relative}")
    return value


def _inputs(
    lock: dict[str, Any], source: Path, train: Path, select: Path, cal: Path
) -> dict[str, Any]:
    for name, expected in lock["source_file_sha256"].items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Official Qwen source differs: {name}")
    for role, path in (("train", train), ("select", select), ("cal", cal)):
        if file_sha256(path) != lock["partition_sha256"][role]:
            raise ValueError(f"Frozen {role} partition differs")
    return source_fingerprint(source)


def audit(
    lock_path: Path,
    source: Path,
    train: Path,
    select: Path,
    cal: Path,
    output: Path,
) -> dict[str, Any]:
    lock = _lock(lock_path)
    fingerprint = _inputs(lock, source, train, select, cal)
    partitions = {
        role: load_partition(path, role)
        for role, path in (("train", train), ("select", select), ("cal", cal))
    }
    check_partition_isolation(partitions)
    if {role: len(rows) for role, rows in partitions.items()} != lock["partition_rows"]:
        raise ValueError("Frozen partition counts differ")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    train_lengths = [
        len(encode(row, tokenizer, lock["max_length"])["ids"])
        for row in partitions["train"]
    ]
    if sum(train_lengths) != lock["train_tokens"]:
        raise ValueError("Frozen tokenizer exposure differs")
    for role in ("select", "cal"):
        for row in partitions[role]:
            encode(row, tokenizer, lock["max_length"])
    selected = _roster(partitions["select"], lock["select32_by_type"])
    receipt = {
        "schema": SCHEMA + "/cpu-audit",
        "status": "PASS",
        "lock_sha256": file_sha256(lock_path),
        "source_fingerprint": fingerprint,
        "partition_sha256": lock["partition_sha256"],
        "partition_rows": lock["partition_rows"],
        "train_tokens": sum(train_lengths),
        "train_lengths": train_lengths,
        "train_id_order_sha256": hashlib.sha256(
            canonical([row["id"] for row in partitions["train"]]).encode()
        ).hexdigest(),
        "select32_ids_sha256": hashlib.sha256(
            canonical([row["id"] for row in selected]).encode()
        ).hexdigest(),
    }
    _write_once(output, receipt)
    return receipt


def _logits(
    model: DecisionModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
) -> torch.Tensor:
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad/EOS")
    items = [encode(row, tokenizer, max_length) for row in rows]
    batch = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in collate(items, pad_id).items()
    }
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return model(**batch).detach().cpu()


def run(
    lock_path: Path,
    audit_path: Path,
    audit_sha256: str,
    source: Path,
    train: Path,
    select: Path,
    cal: Path,
    output: Path,
) -> dict[str, Any]:
    lock = _lock(lock_path)
    if file_sha256(audit_path) != audit_sha256:
        raise ValueError("CPU audit receipt differs from reviewed hash")
    receipt = json.loads(audit_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("lock_sha256") != file_sha256(
        lock_path
    ):
        raise ValueError("CPU audit does not bind this gate")
    if _inputs(lock, source, train, select, cal) != receipt["source_fingerprint"]:
        raise ValueError("Official source changed after audit")
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    train_order = hashlib.sha256(
        canonical([row["id"] for row in train_rows]).encode()
    ).hexdigest()
    if train_order != receipt["train_id_order_sha256"]:
        raise ValueError("TRAIN order changed after audit")
    selected = _roster(select_rows, lock["select32_by_type"])
    selected_hash = hashlib.sha256(
        canonical([row["id"] for row in selected]).encode()
    ).hexdigest()
    if selected_hash != receipt["select32_ids_sha256"]:
        raise ValueError("SELECT roster changed after audit")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")

    seed = lock["seed"]
    zero_rows = _roster(select_rows, dict.fromkeys(lock["select32_by_type"], 1))
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    control, tokenizer = DecisionModel.from_base(source, lock["source_revision"])
    control = control.float().to("cuda:0")
    shared_state = {
        key: value.detach().cpu().clone()
        for key, value in control.head.state_dict().items()
    }
    zero_shared = _logits(control, tokenizer, zero_rows, lock["max_length"])
    del control
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    treatment, tokenizer = DecisionModel.from_base(
        source, lock["source_revision"], head_variant="candidate-interaction"
    )
    if treatment.metadata["architecture"] != QWEN3_INTERACTION_ARCHITECTURE:
        raise ValueError("Loaded the wrong decision architecture")
    if any(
        not torch.equal(treatment.head.state_dict()[key].cpu(), value)
        for key, value in shared_state.items()
    ):
        raise ValueError("Source-matched shared head was not initialized identically")
    treatment = treatment.float().to("cuda:0")
    zero_treatment = _logits(treatment, tokenizer, zero_rows, lock["max_length"])
    zero_exact = torch.equal(zero_shared, zero_treatment)
    del shared_state
    if not zero_exact:
        raise ValueError("Candidate residual changed shared-head zero-step logits")

    zero_first = _predict(treatment, tokenizer, zero_rows, lock["max_length"])
    zero_second = _predict(treatment, tokenizer, zero_rows, lock["max_length"])
    zero_repeat = _compare(zero_first, zero_second)
    zero_seconds = time.monotonic() - started
    zero_status = _phase_status(
        zero_repeat,
        expected_rows=3,
        max_drift=1e-4,
        elapsed_seconds=zero_seconds,
        seconds_cap=lock["zero_step_gpu_seconds_cap"],
    )
    zero_path = output / "zero-step-result.json"
    _write_once(
        zero_path,
        {
            "schema": SCHEMA + "/zero-step-result",
            "status": zero_status,
            "lock_sha256": file_sha256(lock_path),
            "cpu_audit_sha256": audit_sha256,
            "source_revision": lock["source_revision"],
            "exact_control_logits": zero_exact,
            "repeat": zero_repeat,
            "gpu_elapsed_seconds": zero_seconds,
            "gpu_seconds_cap": lock["zero_step_gpu_seconds_cap"],
        },
    )
    if zero_status != "PASS":
        raise ValueError("Zero-step runtime budget or repeat failed")

    one_started = time.monotonic()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad/EOS")
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
        gradient = treatment.head.interaction_out.weight.grad
        if kind == "noul" and gradient is not None:
            raise ValueError("Noul unexpectedly trained interaction parameters")
        if kind != "noul" and (
            gradient is None
            or not torch.isfinite(gradient).all()
            or gradient.abs().sum().item() == 0
        ):
            raise ValueError(f"Missing {kind} interaction gradient")
    optimizer.zero_grad(set_to_none=True)

    first_window = epoch_batches(
        receipt["train_lengths"],
        [],
        epoch=0,
        seed=seed,
        microbatch=1,
        replay_fraction=0.0,
    )[:16]
    chosen = [train_rows[index] for group in first_window for _, index in group]
    if len(chosen) != 16:
        raise ValueError("Control's first optimizer window was not reproduced")
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
        max_drift=lock["parity_max_drift"],
        elapsed_seconds=one_seconds,
        seconds_cap=lock["one_step_gpu_seconds_cap"],
    )
    one_path = output / "one-step-result.json"
    _write_once(
        one_path,
        {
            "schema": SCHEMA + "/one-step-result",
            "status": one_status,
            "lock_sha256": file_sha256(lock_path),
            "cpu_audit_sha256": audit_sha256,
            "first_update": {
                "examples": 16,
                "mean_loss": sum(losses) / 16,
                "gradient_norm": float(gradient_norm),
            },
            "reload32": comparison,
            "gpu_elapsed_seconds": one_seconds,
            "gpu_seconds_cap": lock["one_step_gpu_seconds_cap"],
        },
    )
    result = {
        "schema": SCHEMA + "/gpu-result",
        "status": one_status,
        "lock_sha256": file_sha256(lock_path),
        "cpu_audit_sha256": audit_sha256,
        "source_revision": lock["source_revision"],
        "architecture": restored.metadata["architecture"],
        "head_parameter_count": sum(p.numel() for p in restored.head.parameters()),
        "zero_step_receipt_sha256": file_sha256(zero_path),
        "one_step_receipt_sha256": file_sha256(one_path),
        "total_gpu_elapsed_seconds": time.monotonic() - started,
    }
    _write_once(output / "technical-gate-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "run"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256")
    args = parser.parse_args()
    if args.stage == "audit":
        if args.audit is not None or args.audit_sha256 is not None:
            parser.error("CPU audit does not accept a previous receipt")
        result = audit(
            args.lock, args.source, args.train, args.select, args.cal, args.output
        )
    else:
        if args.audit is None or args.audit_sha256 is None:
            parser.error("GPU run needs a reviewed CPU audit receipt and hash")
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            result = run(
                args.lock,
                args.audit,
                args.audit_sha256,
                args.source,
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
    print(
        json.dumps({key: result[key] for key in ("schema", "status")}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
