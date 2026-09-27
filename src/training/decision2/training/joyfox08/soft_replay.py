"""One preregistered Joyfox 0.8B source-probability replay experiment.

This module uses only the frozen rights-clean TRAIN sample and SELECT. Its
source cache and numeric gates are private artifacts. No sealed benchmark
labels, Jev API outputs, or teacher corpus are read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from inference.joyfox import MODEL_REVISION, verify_release
from training.model.data import file_sha256, load_partition

from . import pilot, train

CONTRACT = "joyfox08-native1024-cleanv2-soft-replay64-v1"
SAMPLE_SHA256 = "ecb50a755c351c903a72a73a285d54622b141fce3e09809bf79d609ae8d2e532"
MANIFEST_SHA256 = "6d7d22309b0271b8e430f38f75bb4af65c16899d0dabcd278910bae14e79a0d4"
IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
SMOKE_QUOTAS = {"choice": 12, "noul": 10, "score": 10}
SMOKE_SEED = "joyfox08-replay-smoke-v1"
TEACHER_TEMPERATURE = 2.0
REPLAY_WEIGHT = 0.2
MAX_PROBABILITY_DRIFT = 1e-6


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _source_identity(
    model_path: Path, source_path: Path, sample_path: Path, manifest_path: Path
) -> dict[str, Any]:
    if (
        file_sha256(sample_path) != SAMPLE_SHA256
        or file_sha256(manifest_path) != MANIFEST_SHA256
    ):
        raise ValueError("Frozen 512-row sample or manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("selected_count") != 512
        or manifest.get("selected_sha256") != SAMPLE_SHA256
    ):
        raise ValueError("Frozen 512-row manifest contents changed")
    return verify_release(
        model_path.resolve(strict=True),
        source_path.resolve(strict=True),
        MODEL_REVISION,
    )


def _native_modules(source_path: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str((source_path / "src").resolve(strict=True)))
    from jev_inference import DecisionEngine
    from jev_inference.model import encode, question_options

    return DecisionEngine, encode, question_options


def _encode_sample(
    rows: list[dict[str, Any]], tokenizer: Any, encode: Any, question_options: Any
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        record = pilot.to_record(row)
        native = encode(tokenizer, record, pilot.MAX_TOKENS)
        keys, _ = question_options(record["questions"]["decision"])
        if pilot.target_key(row) not in keys:
            raise ValueError(f"{row['id']}: native target absent")
        token_count = len(native["ids"])
        effective_length = max(
            native["segments"].count(0) + native["segments"].count(segment)
            for segment in set(native["segments"]) - {0}
        )
        if effective_length > pilot.MAX_TOKENS:
            raise ValueError(f"{row['id']}: native context overflow")
        prepared.append(
            {
                "row": row,
                "native": native,
                "keys": keys,
                "token_count": token_count,
                "effective_length": effective_length,
            }
        )
    if len(prepared) != 512:
        raise ValueError("Expected all 512 frozen training rows")
    return prepared


def smoke_ids(prepared: list[dict[str, Any]]) -> list[str]:
    selected = []
    for kind, quota in SMOKE_QUOTAS.items():
        members = [entry for entry in prepared if entry["row"]["task_type"] == kind]

        def rank(entry: dict[str, Any]) -> str:
            return hashlib.sha256(
                f"{SMOKE_SEED}:{entry['row']['id']}".encode()
            ).hexdigest()

        ranked = sorted(members, key=rank)
        if len(ranked) < quota:
            raise ValueError(f"Insufficient {kind} rows for smoke")
        group = ranked[:quota]
        longest = sorted(
            members, key=lambda entry: (-entry["effective_length"], rank(entry))
        )[0]
        if longest not in group:
            group[-1] = longest
        selected.extend(entry["row"]["id"] for entry in group)
    if len(selected) != 32 or len(set(selected)) != 32:
        raise ValueError("Smoke selection must have 32 distinct rows")
    return selected


def _source_logits(engine: Any, prepared: list[dict[str, Any]]) -> list[list[float]]:
    import torch

    results = []
    with torch.inference_mode():
        for entry in prepared:
            logits = engine.model([entry["native"]], engine.tokenizer.pad_token_id)[
                0
            ].float()
            if logits.numel() != len(entry["keys"]) or not torch.isfinite(logits).all():
                raise ValueError(f"{entry['row']['id']}: invalid source logits")
            results.append(logits.cpu().tolist())
    torch.cuda.synchronize()
    return results


def source_pass(
    *,
    mode: str,
    model_path: Path,
    source_path: Path,
    sample_path: Path,
    manifest_path: Path,
    cache_path: Path | None,
    smoke_path: Path,
    receipt_path: Path,
    runtime_image_id: str,
) -> dict[str, Any]:
    import torch
    import transformers
    import peft

    if mode not in {"cache", "repeat"} or runtime_image_id != IMAGE_ID:
        raise ValueError("Source pass mode or pinned image ID mismatch")
    if (
        smoke_path.exists()
        or receipt_path.exists()
        or (cache_path is not None and cache_path.exists())
    ):
        raise FileExistsError("Source-pass evidence already exists")
    source = _source_identity(model_path, source_path, sample_path, manifest_path)
    rows = load_partition(sample_path, "train")
    DecisionEngine, encode, question_options = _native_modules(source_path)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pinned BF16 GPU unavailable")
    engine = DecisionEngine.load(
        model_path, device="cuda:0", dtype="bfloat16", cutoff_len=pilot.MAX_TOKENS
    )
    prepared = _encode_sample(rows, engine.tokenizer, encode, question_options)
    ids = smoke_ids(prepared)
    wanted = (
        prepared
        if mode == "cache"
        else [entry for entry in prepared if entry["row"]["id"] in ids]
    )
    started = time.perf_counter()
    logits = _source_logits(engine, wanted)
    elapsed = time.perf_counter() - started
    records = [
        {
            "id": entry["row"]["id"],
            "input_sha256": entry["row"]["input_sha256"],
            "keys": entry["keys"],
            "logits": values,
            "token_count": entry["token_count"],
            "effective_length": entry["effective_length"],
        }
        for entry, values in zip(wanted, logits, strict=True)
    ]
    smoke = [record for record in records if record["id"] in ids]
    if len(smoke) != 32:
        raise ValueError("Source smoke missing rows")
    if mode == "cache":
        if cache_path is None:
            raise ValueError("Cache path required for first source pass")
        _write_new(cache_path, records)
    elif cache_path is not None:
        raise ValueError("Repeat process must not rewrite source cache")
    _write_new(smoke_path, smoke)
    receipt = {
        "contract": CONTRACT,
        "mode": mode,
        "source_code_sha256": {
            "soft_replay.py": file_sha256(__file__),
            **train.source_hashes(),
        },
        "source": source,
        "runtime_image_id": runtime_image_id,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "sample_sha256": SAMPLE_SHA256,
        "sample_manifest_sha256": MANIFEST_SHA256,
        "source_head_sha256": source["model_head_sha256"],
        "source_backbone_sha256": "af4cfee3e639739d2a5d19f76923555ba017c801b65bdcdc1f3f8a686bbc9992",
        "source_tokenizer_sha256": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
        "token_count": sum(entry["token_count"] for entry in prepared),
        "smoke_ids": ids,
        "smoke_ids_sha256": _hash_json(ids),
        "smoke_sha256": file_sha256(smoke_path),
        "cache_sha256": file_sha256(cache_path) if cache_path is not None else None,
        "measured_gpu_seconds": elapsed,
        "measured_gpu_hours": elapsed / 3600,
        "records": len(records),
    }
    _write_new(receipt_path, receipt)
    return receipt


def compare_source_passes(
    *,
    first_smoke: Path,
    second_smoke: Path,
    first_receipt: Path,
    second_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    first = json.loads(first_receipt.read_text(encoding="utf-8"))
    second = json.loads(second_receipt.read_text(encoding="utf-8"))
    if first.get("mode") != "cache" or second.get("mode") != "repeat":
        raise ValueError("Expected cache and independent repeat process")
    for key in (
        "contract",
        "source",
        "source_code_sha256",
        "runtime_image_id",
        "torch",
        "hip",
        "transformers",
        "peft",
        "sample_sha256",
        "sample_manifest_sha256",
        "token_count",
        "smoke_ids",
        "smoke_ids_sha256",
    ):
        if first.get(key) != second.get(key):
            raise ValueError(f"Source pass mismatch: {key}")
    if first.get("smoke_sha256") != file_sha256(first_smoke) or second.get(
        "smoke_sha256"
    ) != file_sha256(second_smoke):
        raise ValueError("Source smoke bytes changed")
    records1 = json.loads(first_smoke.read_text(encoding="utf-8"))
    records2 = json.loads(second_smoke.read_text(encoding="utf-8"))
    if len(records1) != len(records2) or len(records1) != 32:
        raise ValueError("Expected two full 32-row source smokes")
    by_id2 = {row["id"]: row for row in records2}
    if len(by_id2) != 32 or set(by_id2) != set(first["smoke_ids"]):
        raise ValueError("Source smoke IDs changed")
    max_probability_drift = 0.0
    category_changes = 0
    for row1 in records1:
        row2 = by_id2[row1["id"]]
        for key in ("input_sha256", "keys", "token_count", "effective_length"):
            if row1[key] != row2[key]:
                raise ValueError(f"{row1['id']}: source cache input changed")
        if len(row1["logits"]) != len(row2["logits"]):
            raise ValueError(f"{row1['id']}: option count changed")
        probs1 = _softmax(row1["logits"])
        probs2 = _softmax(row2["logits"])
        category_changes += probs1.index(max(probs1)) != probs2.index(max(probs2))
        max_probability_drift = max(
            max_probability_drift,
            *(abs(a - b) for a, b in zip(probs1, probs2, strict=True)),
        )
    passed = category_changes == 0 and max_probability_drift <= MAX_PROBABILITY_DRIFT
    result = {
        "contract": CONTRACT,
        "first_receipt_sha256": file_sha256(first_receipt),
        "second_receipt_sha256": file_sha256(second_receipt),
        "first_smoke_sha256": file_sha256(first_smoke),
        "second_smoke_sha256": file_sha256(second_smoke),
        "cache_sha256": first["cache_sha256"],
        "smoke_ids_sha256": first["smoke_ids_sha256"],
        "rows": 32,
        "category_changes": category_changes,
        "max_probability_drift": max_probability_drift,
        "numeric_gate_passed": passed,
    }
    _write_new(output, result)
    return result


def _softmax(values: list[float]) -> list[float]:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite or empty source logits")
    pivot = max(values)
    weights = [math.exp(value - pivot) for value in values]
    norm = sum(weights)
    return [weight / norm for weight in weights]


def replay_loss(values: Any, target: int, source_logits: Any, torch: Any) -> Any:
    from torch.nn import functional as F

    base = train.loss_for(values, target, torch)
    source = source_logits.float().detach()
    student = values.float()
    if (
        source.shape != student.shape
        or not torch.isfinite(source).all()
        or not torch.isfinite(student).all()
    ):
        raise ValueError("Source/student logits invalid for native offered options")
    divergence = F.kl_div(
        F.log_softmax(student / TEACHER_TEMPERATURE, dim=-1),
        F.softmax(source / TEACHER_TEMPERATURE, dim=-1),
        reduction="sum",
    )
    return base + REPLAY_WEIGHT * TEACHER_TEMPERATURE**2 * divergence


def _load_frozen_cache(
    cache_path: Path,
    source_receipt_path: Path,
    compare_path: Path,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    comparison = json.loads(compare_path.read_text(encoding="utf-8"))
    if (
        source_receipt.get("cache_sha256") != file_sha256(cache_path)
        or comparison.get("cache_sha256") != file_sha256(cache_path)
        or comparison.get("numeric_gate_passed") is not True
        or comparison.get("first_receipt_sha256") != file_sha256(source_receipt_path)
    ):
        raise ValueError("Source cache or two-process numeric gate is not frozen")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in cache}
    if len(cache) != len(by_id) or len(cache) != 512:
        raise ValueError("Source soft cache must have exactly 512 distinct rows")
    if set(by_id) != {entry["row"]["id"] for entry in prepared}:
        raise ValueError("Source soft cache IDs differ from TRAIN sample")
    for entry in prepared:
        row = entry["row"]
        cached = by_id[row["id"]]
        if (
            cached.get("input_sha256") != row["input_sha256"]
            or cached.get("keys") != entry["keys"]
            or cached.get("token_count") != entry["token_count"]
            or cached.get("effective_length") != entry["effective_length"]
            or len(cached.get("logits", [])) != len(entry["keys"])
        ):
            raise ValueError(f"{row['id']}: source soft cache/input differs")
        _softmax(cached["logits"])
    return by_id, source_receipt


def run_train(
    *,
    model_path: Path,
    source_path: Path,
    sample_path: Path,
    manifest_path: Path,
    select_path: Path,
    cache_path: Path,
    source_receipt_path: Path,
    compare_path: Path,
    output: Path,
    runtime_image_id: str,
) -> dict[str, Any]:
    import torch

    if runtime_image_id != IMAGE_ID or output.exists():
        raise ValueError("Pinned runtime image changed or output already exists")
    source = _source_identity(model_path, source_path, sample_path, manifest_path)
    if file_sha256(select_path) != pilot.HASHES["select"]:
        raise ValueError("Frozen SELECT changed")
    rows = load_partition(sample_path, "train")
    select = load_partition(select_path, "select")
    if len(select) != 700:
        raise ValueError("SELECT row count changed")
    DecisionEngine, encode, question_options = _native_modules(source_path)
    # A read-only source load first validates the entire 512-row native encoding.
    source_engine = DecisionEngine.load(
        model_path, device="cuda:0", dtype="bfloat16", cutoff_len=pilot.MAX_TOKENS
    )
    prepared = _encode_sample(rows, source_engine.tokenizer, encode, question_options)
    by_id, source_receipt = _load_frozen_cache(
        cache_path, source_receipt_path, compare_path, prepared
    )
    if (
        source_receipt.get("source") != source
        or source_receipt.get("runtime_image_id") != runtime_image_id
        or source_receipt.get("source_code_sha256")
        != {
            "soft_replay.py": file_sha256(__file__),
            **train.source_hashes(),
        }
    ):
        raise ValueError("Cached source revision/runtime differs")
    del source_engine
    torch.cuda.empty_cache()
    engine, targets, preflight = train.load_and_preflight(model_path, source_path, rows)
    post_control_rng = torch.random.get_rng_state().clone()
    post_control_cuda_rng = torch.cuda.get_rng_state().clone()
    train_rows, train_invalid = train.encode_rows(
        rows, engine.tokenizer, encode, question_options
    )
    select_rows, select_invalid = train.encode_rows(
        select, engine.tokenizer, encode, question_options
    )
    if train_invalid or len(train_rows) != 512:
        raise ValueError("Frozen TRAIN sample not native admissible")
    model = engine.model
    model.eval()
    smoke_ids_set = set(source_receipt["smoke_ids"])
    max_abs_zero_step = 0.0
    with torch.inference_mode():
        for row, encoded, _ in train_rows:
            if row["id"] not in smoke_ids_set:
                continue
            student = model([encoded], engine.tokenizer.pad_token_id)[0].float()
            teacher = torch.tensor(
                by_id[row["id"]]["logits"], device=student.device, dtype=torch.float32
            )
            max_abs_zero_step = max(
                max_abs_zero_step, float((student - teacher).abs().max().item())
            )
    if max_abs_zero_step > MAX_PROBABILITY_DRIFT or len(smoke_ids_set) != 32:
        raise RuntimeError(
            f"Zero-step source/LoRA numeric gate failed: {max_abs_zero_step}"
        )
    model.train()
    first_row, first_encoded, target = train_rows[0]
    values = model([first_encoded], engine.tokenizer.pad_token_id)[0]
    teacher = torch.tensor(by_id[first_row["id"]]["logits"], device=values.device)
    probe_loss = replay_loss(values, target, teacher, torch)
    if not torch.isfinite(probe_loss):
        raise RuntimeError("Zero-step replay loss is nonfinite")
    probe_loss.backward()
    grads = [
        p.grad
        for p in model.backbone.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads or not any(
        torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads
    ):
        raise RuntimeError("Zero-step replay LoRA gradient missing/nonfinite")
    model.zero_grad(set_to_none=True)
    # The hard-label control did not perform this extra probe. Restore its
    # post-preflight Torch RNG so dropout draws remain a matched comparison.
    torch.random.set_rng_state(post_control_rng)
    torch.cuda.set_rng_state(post_control_cuda_rng)
    ordered = list(train_rows)
    random.Random(train.SEED).shuffle(ordered)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train.LR,
        weight_decay=train.WEIGHT_DECAY,
        eps=train.ADAM_EPS,
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    select0 = train.evaluate_select(
        model, select_rows, select_invalid, engine.tokenizer.pad_token_id, torch
    )
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for index, (row, encoded, target) in enumerate(ordered):
        model.train()
        values = model([encoded], engine.tokenizer.pad_token_id)[0]
        teacher = torch.tensor(
            by_id[row["id"]]["logits"], device=values.device, dtype=torch.float32
        )
        loss = replay_loss(values, target, teacher, torch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite replay loss at sample {index}")
        (loss / train.ACCUMULATION).backward()
        losses.append(float(loss.detach().item()))
        if (index + 1) % train.ACCUMULATION:
            continue
        step = (index + 1) // train.ACCUMULATION
        norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradient norm at step {step}")
        optimizer.param_groups[0]["lr"] = train.schedule(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step in (32, 64):
            model.backbone.save_pretrained(output / f"step-{step:04d}")
    select64 = train.evaluate_select(
        model, select_rows, select_invalid, engine.tokenizer.pad_token_id, torch
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    receipt = {
        "contract": CONTRACT,
        "source": source,
        "source_code_sha256": {
            "soft_replay.py": file_sha256(__file__),
            **train.source_hashes(),
        },
        "runtime_image_id": runtime_image_id,
        "sample_sha256": SAMPLE_SHA256,
        "sample_manifest_sha256": MANIFEST_SHA256,
        "select_sha256": file_sha256(select_path),
        "cache_sha256": file_sha256(cache_path),
        "source_receipt_sha256": file_sha256(source_receipt_path),
        "compare_sha256": file_sha256(compare_path),
        "max_abs_zero_step_logits": max_abs_zero_step,
        "zero_step_replay_loss": float(probe_loss.detach().item()),
        "source_preflight": preflight,
        "lora_target_sha256": _hash_json(targets),
        "token_count": source_receipt["token_count"],
        "updates": 64,
        "processed_rows": len(ordered),
        "selected_step": 64,
        "optimizer": {
            "accumulation": train.ACCUMULATION,
            "rank": train.RANK,
            "alpha": train.ALPHA,
            "dropout": train.DROPOUT,
            "lr": train.LR,
            "floor": train.LR_FLOOR,
            "warmup": train.WARMUP,
            "weight_decay": train.WEIGHT_DECAY,
            "eps": train.ADAM_EPS,
            "brier_weight": train.BRIER_WEIGHT,
            "replay_weight": REPLAY_WEIGHT,
            "replay_temperature": TEACHER_TEMPERATURE,
            "seed": train.SEED,
        },
        "mean_train_loss": sum(losses) / len(losses),
        "select": {"0": select0, "64": select64},
        "checkpoint_sha256": {
            "32": file_sha256(output / "step-0032" / "adapter_model.safetensors"),
            "64": file_sha256(output / "step-0064" / "adapter_model.safetensors"),
        },
        "measured_gpu_seconds": elapsed,
        "measured_gpu_hours": elapsed / 3600,
    }
    _write_new(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "repeat", "train"):
        child = sub.add_parser(name)
        for option in (
            "model-path",
            "source-path",
            "sample-path",
            "manifest-path",
            "runtime-image-id",
        ):
            child.add_argument(
                f"--{option}",
                required=True,
                type=Path if option.endswith("path") else str,
            )
        if name in {"cache", "repeat"}:
            child.add_argument("--smoke-path", required=True, type=Path)
            child.add_argument("--receipt-path", required=True, type=Path)
            if name == "cache":
                child.add_argument("--cache-path", required=True, type=Path)
        else:
            for option in (
                "select-path",
                "cache-path",
                "source-receipt-path",
                "compare-path",
                "output",
            ):
                child.add_argument(f"--{option}", required=True, type=Path)
    compare = sub.add_parser("compare")
    for option in (
        "first-smoke",
        "second-smoke",
        "first-receipt",
        "second-receipt",
        "output",
    ):
        compare.add_argument(f"--{option}", required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command in {"cache", "repeat"}:
        if command == "repeat":
            args["cache_path"] = None
        result = source_pass(mode=command, **args)
    elif command == "compare":
        result = compare_source_passes(**args)
    else:
        result = run_train(**args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
