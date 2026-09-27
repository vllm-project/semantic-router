"""CPU audit, then one bounded GPU update and native typed-head reload gate.

This is a technical admission test, not a model-selection or evaluation run.
The private audit receipt contains only row identities and token lengths, no
TRAIN/CAL text. A separate root review is required before the GPU stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    QWEN3_TYPED_ARCHITECTURE,
    DecisionModel,
    collate,
    encode,
)
from training.model.loss import per_example_loss
from training.model.plan import epoch_batches
from training.model.source import source_fingerprint
from training.model.train import learning_factor
from transformers import AutoTokenizer

SCHEMA = "decision2-qwen06-type-head-technical-gate/1"
RESEARCH_ROOT = Path(__file__).resolve().parents[1]


def _write_once(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != SCHEMA:
        raise ValueError("Unknown typed-head gate schema")
    for relative, expected in lock["code_sha256"].items():
        if file_sha256(RESEARCH_ROOT / relative) != expected:
            raise ValueError(f"Frozen gate source changed: {relative}")
    return lock


def _verify_inputs(
    lock: dict[str, Any], source: Path, train: Path, select: Path, cal: Path
) -> dict[str, Any]:
    for relative, expected in lock["source_file_sha256"].items():
        if file_sha256(source / relative) != expected:
            raise ValueError(f"Pinned source file changed: {relative}")
    for role, path in (("train", train), ("select", select), ("cal", cal)):
        if file_sha256(path) != lock["partition_sha256"][role]:
            raise ValueError(f"Frozen {role} data changed")
    return source_fingerprint(source)


def _roster(rows: list[dict[str, Any]], counts: dict[str, int]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for kind, count in counts.items():
        matching = [row for row in rows if row["task_type"] == kind]
        if len(matching) < count:
            raise ValueError(f"Too few {kind} rows for technical gate")
        chosen.extend(matching[:count])
    indexed = {row["id"]: index for index, row in enumerate(rows)}
    return sorted(chosen, key=lambda row: indexed[row["id"]])


def audit(
    lock_path: Path,
    source: Path,
    train: Path,
    select: Path,
    cal: Path,
    output: Path,
) -> dict[str, Any]:
    lock = _read_lock(lock_path)
    fingerprint = _verify_inputs(lock, source, train, select, cal)
    partitions = {
        role: load_partition(path, role)
        for role, path in (("train", train), ("select", select), ("cal", cal))
    }
    check_partition_isolation(partitions)
    for role, rows in partitions.items():
        if len(rows) != lock["partition_rows"][role]:
            raise ValueError(f"Frozen {role} row count changed")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    lengths = {}
    for role, rows in partitions.items():
        lengths[role] = [
            len(encode(row, tokenizer, lock["max_length"])["ids"]) for row in rows
        ]
    if sum(lengths["train"]) != lock["train_tokens"]:
        raise ValueError("Frozen TRAIN tokenizer exposure changed")
    selected = _roster(partitions["select"], lock["select32_by_type"])
    receipt = {
        "schema": SCHEMA + "/cpu-audit",
        "lock_sha256": file_sha256(lock_path),
        "source_fingerprint": fingerprint,
        "partition_sha256": lock["partition_sha256"],
        "partition_rows": lock["partition_rows"],
        "train_tokens": sum(lengths["train"]),
        "max_tokens_by_role": {key: max(values) for key, values in lengths.items()},
        "train_lengths": lengths["train"],
        "train_id_order_sha256": hashlib.sha256(
            canonical([row["id"] for row in partitions["train"]]).encode()
        ).hexdigest(),
        "select32_ids_sha256": hashlib.sha256(
            canonical([row["id"] for row in selected]).encode()
        ).hexdigest(),
        "status": "PASS",
    }
    _write_once(output, receipt)
    return receipt


def _predict(
    model: DecisionModel, tokenizer: Any, rows: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise ValueError("Tokenizer needs pad or EOS")
    items = [encode(row, tokenizer, max_length) for row in rows]
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), 2):
            pair = items[start : start + 2]
            batch = {
                key: value.to("cuda:0") if torch.is_tensor(value) else value
                for key, value in collate(pair, pad).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            for item, values in zip(pair, logits):
                probs = values[: len(item["keys"])].float().softmax(-1).cpu().tolist()
                if any(not math.isfinite(value) for value in probs):
                    raise ValueError("Nonfinite valid-option probability")
                result.append(
                    {
                        "id": item["id"],
                        "tokens_sha256": item["token_ids_sha256"],
                        "probabilities": probs,
                        "winner": max(range(len(probs)), key=probs.__getitem__),
                    }
                )
    return result


def _compare(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Native reload changed row coverage")
    changed, drift = 0, 0.0
    for before, after in zip(left, right):
        if (
            before["id"] != after["id"]
            or before["tokens_sha256"] != after["tokens_sha256"]
            or len(before["probabilities"]) != len(after["probabilities"])
        ):
            raise ValueError("Native reload changed input or candidate identity")
        changed += before["winner"] != after["winner"]
        drift = max(
            drift,
            *(
                abs(a - b)
                for a, b in zip(before["probabilities"], after["probabilities"])
            ),
        )
    return {
        "rows": len(left),
        "category_changes": changed,
        "max_probability_drift": drift,
    }


def _phase_status(
    comparison: dict[str, Any],
    *,
    expected_rows: int,
    max_drift: float,
    elapsed_seconds: float,
    seconds_cap: int,
) -> str:
    return (
        "PASS"
        if comparison["rows"] == expected_rows
        and comparison["category_changes"] == 0
        and comparison["max_probability_drift"] <= max_drift
        and math.isfinite(elapsed_seconds)
        and elapsed_seconds <= seconds_cap
        else "HOLD"
    )


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
    lock = _read_lock(lock_path)
    if file_sha256(audit_path) != audit_sha256:
        raise ValueError("CPU audit receipt differs from reviewed hash")
    receipt = json.loads(audit_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("lock_sha256") != file_sha256(
        lock_path
    ):
        raise ValueError("CPU audit does not bind this frozen protocol")
    fingerprint = _verify_inputs(lock, source, train, select, cal)
    if fingerprint != receipt.get("source_fingerprint"):
        raise ValueError("Official source changed after CPU audit")
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    if (
        hashlib.sha256(canonical([r["id"] for r in train_rows]).encode()).hexdigest()
        != receipt["train_id_order_sha256"]
    ):
        raise ValueError("TRAIN row order differs from CPU audit")
    selected = _roster(select_rows, lock["select32_by_type"])
    if (
        hashlib.sha256(canonical([r["id"] for r in selected]).encode()).hexdigest()
        != receipt["select32_ids_sha256"]
    ):
        raise ValueError("SELECT parity roster differs from CPU audit")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")

    # Exclude read-only input verification from the separately capped GPU work.
    started = time.monotonic()
    seed = lock["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    first, tokenizer = DecisionModel.from_base(
        source, lock["source_revision"], head_variant="type-separated"
    )
    first = first.float().to("cuda:0")
    zero_rows = _roster(select_rows, dict.fromkeys(lock["select32_by_type"], 1))
    zero_first = _predict(first, tokenizer, zero_rows, lock["max_length"])
    del first
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, tokenizer = DecisionModel.from_base(
        source, lock["source_revision"], head_variant="type-separated"
    )
    if model.metadata["architecture"] != QWEN3_TYPED_ARCHITECTURE:
        raise ValueError("GPU preflight loaded the wrong decision architecture")
    model = model.float().to("cuda:0")
    zero_repeat = _compare(
        zero_first, _predict(model, tokenizer, zero_rows, lock["max_length"])
    )
    zero_step_seconds = time.monotonic() - started
    zero_status = _phase_status(
        zero_repeat,
        expected_rows=3,
        max_drift=1e-4,
        elapsed_seconds=zero_step_seconds,
        seconds_cap=lock["zero_step_gpu_seconds_cap"],
    )
    zero_receipt = output / "zero-step-result.json"
    _write_once(
        zero_receipt,
        {
            "schema": SCHEMA + "/zero-step-result",
            "status": zero_status,
            "lock_sha256": file_sha256(lock_path),
            "cpu_audit_sha256": audit_sha256,
            "source_revision": lock["source_revision"],
            "repeat": zero_repeat,
            "gpu_elapsed_seconds": zero_step_seconds,
            "gpu_seconds_cap": lock["zero_step_gpu_seconds_cap"],
        },
    )
    if zero_status != "PASS":
        raise ValueError("Fresh official-source typed starts failed their gate")
    one_step_started = time.monotonic()

    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise ValueError("Tokenizer needs pad or EOS")
    # The optimizer update uses the exact first 16 TRAIN rows selected by the
    # completed control's epoch_batches implementation and 466-step schedule.
    batches = epoch_batches(
        receipt["train_lengths"],
        [],
        epoch=0,
        seed=seed,
        microbatch=1,
        replay_fraction=0.0,
    )
    chosen = [train_rows[index] for batch in batches[:16] for _, index in batch]
    if len(chosen) != 16:
        raise ValueError("First accumulation window is not 16 rows")
    factor = learning_factor(0, 466, 0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": 2e-5 * factor},
            {"params": list(model.head.parameters()), "lr": 2e-4 * factor},
        ],
        weight_decay=0.01,
        foreach=True,
    )
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    # Independently probe every readout's gradient without changing weights.
    for kind in ("choice", "noul", "score"):
        row = next(row for row in train_rows if row["task_type"] == kind)
        item = encode(row, tokenizer, lock["max_length"])
        batch = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in collate([item], pad).items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch)
            loss = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )["total"].mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite {kind} technical loss")
        loss.backward()
        parameters = model.head.heads[
            ("choice", "noul", "score").index(kind)
        ].parameters()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f"Missing or nonfinite {kind} readout gradient")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for row in chosen:
        item = encode(row, tokenizer, lock["max_length"])
        batch = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in collate([item], pad).items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch)
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
            raise ValueError("Nonfinite first-window technical loss")
        loss.backward()
        losses.append(float(loss.detach().item()) * 16)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise ValueError("Nonfinite first-window gradient norm")
    optimizer.step()
    torch.cuda.synchronize()
    before = _predict(model, tokenizer, selected, lock["max_length"])
    checkpoint = output / "checkpoint-0000001"
    model.save(checkpoint, tokenizer)
    del model
    torch.cuda.empty_cache()
    reloaded, tokenizer = DecisionModel.from_checkpoint(checkpoint)
    reloaded = reloaded.float().to("cuda:0")
    after = _predict(reloaded, tokenizer, selected, lock["max_length"])
    parity = _compare(before, after)
    one_step_seconds = time.monotonic() - one_step_started
    one_status = _phase_status(
        parity,
        expected_rows=32,
        max_drift=lock["parity_max_drift"],
        elapsed_seconds=one_step_seconds,
        seconds_cap=lock["one_step_gpu_seconds_cap"],
    )
    one_receipt = output / "one-step-result.json"
    _write_once(
        one_receipt,
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
            "reload32": parity,
            "gpu_elapsed_seconds": one_step_seconds,
            "gpu_seconds_cap": lock["one_step_gpu_seconds_cap"],
        },
    )
    result = {
        "schema": SCHEMA + "/gpu-result",
        "status": one_status,
        "lock_sha256": file_sha256(lock_path),
        "cpu_audit_sha256": audit_sha256,
        "source_revision": lock["source_revision"],
        "architecture": reloaded.metadata["architecture"],
        "head_parameter_count": sum(p.numel() for p in reloaded.head.parameters()),
        "zero_step_receipt_sha256": file_sha256(zero_receipt),
        "one_step_receipt_sha256": file_sha256(one_receipt),
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
            parser.error("GPU run needs the reviewed CPU audit and SHA-256")
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
