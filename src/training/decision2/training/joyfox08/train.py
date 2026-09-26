"""One fixed-budget LoRA continuation of the pinned native Joyfox 0.8B model.

Runs only on an authorized GPU experiment host, reading an audited TRAIN sample
and disjoint SELECT. A gradient/source parity preflight occurs before step one.
It never reads benchmark gold, calibration labels, or teacher distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from inference.joyfox import MODEL_REVISION, verify_release

from training.model.data import file_sha256, load_partition
from training.model.lora import select_target_modules

from . import pilot

SEED = 20260927
UPDATES = 64
ACCUMULATION = 8
RANK = 8
ALPHA = 16
DROPOUT = 0.05
LR = 1e-5
LR_FLOOR = 1e-6
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
WARMUP = 8
BRIER_WEIGHT = 0.25


def source_hashes() -> dict[str, str]:
    return {
        path.name: file_sha256(path) for path in (Path(__file__), Path(pilot.__file__))
    }


def schedule(step: int) -> float:
    if not 1 <= step <= UPDATES:
        raise ValueError("Optimizer step outside frozen budget")
    if step <= WARMUP:
        return LR * step / WARMUP
    progress = (step - WARMUP) / (UPDATES - WARMUP)
    return LR_FLOOR + (LR - LR_FLOOR) * (1 + math.cos(math.pi * progress)) / 2


def loss_for(logits: Any, target: int, torch: Any) -> Any:
    from torch.nn import functional as F

    values = logits.float()
    label = torch.tensor([target], dtype=torch.long, device=values.device)
    probabilities = values.softmax(-1)
    one_hot = F.one_hot(label, num_classes=len(values)).float()[0]
    return (
        F.cross_entropy(values[None, :], label)
        + BRIER_WEIGHT * (probabilities - one_hot).square().sum()
    )


def encode_rows(
    rows: list[dict[str, Any]], tokenizer: Any, encode: Any, question_options: Any
):
    result = []
    invalid = []
    for row in rows:
        record = pilot.to_record(row)
        question = record["questions"]["decision"]
        try:
            encoded = encode(tokenizer, record, pilot.MAX_TOKENS)
            keys, _ = question_options(question)
        except ValueError as exc:
            if "input requires" not in str(exc):
                raise
            invalid.append(row)
            continue
        key = pilot.target_key(row)
        if key not in keys:
            raise ValueError(f"{row['id']}: target absent from native options")
        result.append((row, encoded, keys.index(key)))
    return result, invalid


def metric_summary(
    by_family: dict[str, list[tuple[bool, float]]],
    by_type: dict[str, list[bool]],
    invalid: int,
):
    family_acc = [
        sum(correct for correct, _ in values) / len(values)
        for values in by_family.values()
    ]
    family_brier = [
        sum(brier for _, brier in values) / len(values) for values in by_family.values()
    ]
    return {
        "correct": sum(sum(values) for values in by_type.values()),
        "total": sum(map(len, by_type.values())),
        "invalid": invalid,
        "family_macro_accuracy": sum(family_acc) / len(family_acc),
        "family_macro_brier": sum(family_brier) / len(family_brier),
        "by_type": {
            kind: {"correct": sum(values), "total": len(values)}
            for kind, values in sorted(by_type.items())
        },
        "by_family": {
            family: {
                "correct": sum(correct for correct, _ in values),
                "total": len(values),
            }
            for family, values in sorted(by_family.items())
        },
    }


def evaluate_select(
    model: Any,
    encoded_rows: Any,
    invalid: list[dict[str, Any]],
    pad_id: int,
    torch: Any,
):
    was_training = model.training
    model.eval()
    by_family: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    with torch.inference_mode():
        for row, encoded, target in encoded_rows:
            logits = model([encoded], pad_id)[0].float()
            probs = logits.softmax(-1)
            best = int(probs.argmax().item())
            correct = best == target and int((probs == probs.max()).sum().item()) == 1
            one_hot = torch.zeros_like(probs)
            one_hot[target] = 1
            brier = float((probs - one_hot).square().sum().item()) / 2
            by_family[row["family"]].append((correct, brier))
            by_type[row["task_type"]].append(correct)
    if was_training:
        model.train()
    for row in invalid:
        by_family[row["family"]].append((False, 1.0))
        by_type[row["task_type"]].append(False)
    return metric_summary(by_family, by_type, len(invalid))


def load_and_preflight(
    model_path: Path, source_path: Path, sample: list[dict[str, Any]]
):
    import torch
    from jev_inference import DecisionEngine
    from jev_inference.model import encode, question_options
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU is required for the native source")
    torch.manual_seed(SEED)
    random.seed(SEED)
    engine = DecisionEngine.load(
        model_path, device="cuda:0", dtype="bfloat16", cutoff_len=pilot.MAX_TOKENS
    )
    if next(engine.model.parameters()).device.type != "cuda":
        raise RuntimeError("Native model fell back to CPU")
    model = engine.model
    model.head.requires_grad_(False)
    encoded, rejected = encode_rows(
        sample[:1], engine.tokenizer, encode, question_options
    )
    if rejected or len(encoded) != 1:
        raise ValueError("Selected TRAIN preflight row is not native-admitted")
    before = model([encoded[0][1]], engine.tokenizer.pad_token_id)[0].detach().float()
    targets = select_target_modules(model.backbone)
    config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    model.backbone = get_peft_model(model.backbone, config)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.eval()
    after = model([encoded[0][1]], engine.tokenizer.pad_token_id)[0].detach().float()
    if not torch.allclose(before, after, rtol=1e-3, atol=1e-3):
        raise RuntimeError(
            "Source predictions changed when zero-initialized LoRA was attached"
        )
    model.train()
    value = model([encoded[0][1]], engine.tokenizer.pad_token_id)[0]
    loss = loss_for(value, encoded[0][2], torch)
    if not torch.isfinite(loss):
        raise RuntimeError("Native preflight loss is non-finite")
    loss.backward()
    gradients = [
        p.grad
        for p in model.backbone.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not gradients or not any(
        torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gradients
    ):
        raise RuntimeError("No finite nonzero LoRA gradient reached the optimizer")
    model.zero_grad(set_to_none=True)
    return (
        engine,
        targets,
        {
            "source_logits_parity_max_abs": float((before - after).abs().max()),
            "preflight_loss": float(loss.detach()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "lora_target_count": len(targets),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
    )


def run(
    model_path: Path,
    source_path: Path,
    sample_path: Path,
    manifest_path: Path,
    select_path: Path,
    output: Path | None,
) -> dict[str, Any]:
    import torch

    if output is not None and output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != pilot.CONTRACT
        or manifest.get("source_data_sha256") != pilot.HASHES
        or manifest.get("selected_sha256") != file_sha256(sample_path)
        or manifest.get("selected_count") != UPDATES * ACCUMULATION
        or file_sha256(select_path) != pilot.HASHES["select"]
    ):
        raise ValueError("Frozen Joyfox pilot data/manifest mismatch")
    source = verify_release(model_path, source_path, MODEL_REVISION)
    if manifest.get("source") != source:
        raise ValueError("Frozen source identity changed")
    sample = load_partition(sample_path, "train")
    select = load_partition(select_path, "select")
    if len(sample) != UPDATES * ACCUMULATION or len(select) != 700:
        raise ValueError("Pilot counts changed")
    sys.path.insert(0, str((source_path / "src").resolve(strict=True)))
    from jev_inference.model import encode, question_options

    engine, targets, preflight = load_and_preflight(model_path, source_path, sample)
    if output is None:
        return {
            "contract": pilot.CONTRACT,
            "source": source,
            "selected_sha256": file_sha256(sample_path),
            "preflight": preflight,
        }
    train_rows, train_invalid = encode_rows(
        sample, engine.tokenizer, encode, question_options
    )
    select_rows, select_invalid = encode_rows(
        select, engine.tokenizer, encode, question_options
    )
    if train_invalid or len(train_rows) != len(sample):
        raise ValueError("TRAIN sample has a native-inadmissible item")
    ordered = list(train_rows)
    random.Random(SEED).shuffle(ordered)
    model = engine.model
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPS,
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_metrics = {}
    selected_metrics[0] = evaluate_select(
        model, select_rows, select_invalid, engine.tokenizer.pad_token_id, torch
    )
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for index, (row, encoded, target) in enumerate(ordered):
        model.train()
        values = model([encoded], engine.tokenizer.pad_token_id)[0]
        loss = loss_for(values, target, torch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite native loss at sample {index}")
        (loss / ACCUMULATION).backward()
        losses.append(float(loss.detach()))
        if (index + 1) % ACCUMULATION:
            continue
        step = (index + 1) // ACCUMULATION
        norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradient norm at step {step}")
        optimizer.param_groups[0]["lr"] = schedule(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step in (32, UPDATES):
            checkpoint = output / f"step-{step:04d}"
            model.backbone.save_pretrained(checkpoint)
            selected_metrics[step] = evaluate_select(
                model, select_rows, select_invalid, engine.tokenizer.pad_token_id, torch
            )
    chosen = max(
        selected_metrics,
        key=lambda step: (
            selected_metrics[step]["family_macro_accuracy"],
            -selected_metrics[step]["family_macro_brier"],
            -step,
        ),
    )
    source_metrics = selected_metrics[0]
    candidate_metrics = selected_metrics[chosen]
    by_type_pass = all(
        candidate_metrics["by_type"].get(kind, {}).get("correct", 0)
        / candidate_metrics["by_type"].get(kind, {}).get("total", 1)
        >= source_metrics["by_type"].get(kind, {}).get("correct", 0)
        / source_metrics["by_type"].get(kind, {}).get("total", 1)
        - 0.01
        for kind in ("choice", "noul", "score")
    )
    passed = (
        chosen != 0
        and candidate_metrics["family_macro_accuracy"]
        >= source_metrics["family_macro_accuracy"] + 0.015
        and by_type_pass
        and candidate_metrics["invalid"] <= source_metrics["invalid"]
    )
    receipt = {
        "contract": pilot.CONTRACT,
        "source": source,
        "source_sha256": source_hashes(),
        "sample_sha256": file_sha256(sample_path),
        "select_sha256": file_sha256(select_path),
        "preflight": preflight,
        "lora_target_sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest(),
        "optimizer": {
            "updates": UPDATES,
            "accumulation": ACCUMULATION,
            "rank": RANK,
            "alpha": ALPHA,
            "dropout": DROPOUT,
            "lr": LR,
            "floor": LR_FLOOR,
            "warmup": WARMUP,
            "weight_decay": WEIGHT_DECAY,
            "eps": ADAM_EPS,
            "brier_weight": BRIER_WEIGHT,
            "seed": SEED,
        },
        "selected_step": chosen,
        "select_gate_passed": passed,
        "select": {str(step): metrics for step, metrics in selected_metrics.items()},
        "mean_train_loss": sum(losses) / len(losses),
        "checkpoint_sha256": {
            str(step): file_sha256(
                output / f"step-{step:04d}" / "adapter_model.safetensors"
            )
            for step in (32, UPDATES)
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-path",
        "source-path",
        "sample-path",
        "manifest-path",
        "select-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
