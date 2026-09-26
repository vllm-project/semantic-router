"""One-epoch pinned Eikos LoRA pilot with its unmodified letter-logit readout.

Only isolated TRAIN rows produce gradients. SELECT chooses checkpoints. CAL is
parsed solely to detect partition overlap; its labels are never used here.
Run this module on one GPU with the qualified ROCm/PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from inference.eikos import REVISION, verify_release

from training.eikos.data import (
    MAX_ONE_PASS,
    PROMPT_VERSION,
    collate,
    encode,
    one_pass_quarantine,
)
from training.eikos.rights import verify_clean_files
from training.model.data import (
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)
from training.model.lora import select_target_modules
from training.model.loss import LOSS_VERSION, per_example_loss
from training.model.plan import epoch_batches


def atomic_json(path: Path, payload: Any) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
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


def letter_logits(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    """Same text backbone, last-token position, and LM head as LetterAdapter.

    Running the backbone directly avoids materializing all-sequence vocabulary
    logits. PEFT has already inserted the LoRA projections into that backbone.
    """
    base = model.get_base_model()
    hidden = base.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=not model.training,
    ).last_hidden_state
    last = hidden[
        torch.arange(len(hidden), device=hidden.device), batch["last_positions"]
    ]
    vocab_logits = base.lm_head(last)
    selected = vocab_logits.gather(1, batch["letter_ids"])
    return selected.float().masked_fill(~batch["candidate_mask"], -float("inf"))


def evaluate(
    model: Any,
    items: list[dict[str, Any]],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    output: Path,
    tag: str,
) -> dict[str, Any]:
    model.eval()
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            portion = items[offset : offset + batch_size]
            batch = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in collate(portion, pad_id).items()
            }
            logits = letter_logits(model, batch)
            probabilities = torch.softmax(logits, -1).cpu().tolist()
            for item, p_full in zip(portion, probabilities):
                probabilities_valid = p_full[: len(item["keys"])]
                pred = max(
                    range(len(probabilities_valid)), key=probabilities_valid.__getitem__
                )
                target = item["label"]
                row = {
                    "id": item["id"],
                    "family": item["family"],
                    "task_type": item["task_type"],
                    "source_input_sha256": item["source_input_sha256"],
                    "keys": item["keys"],
                    "gold_key": item["keys"][target],
                    "prediction_key": item["keys"][pred],
                    "probabilities": dict(zip(item["keys"], probabilities_valid)),
                    "correct": pred == target,
                    "brier": sum(
                        (value - float(index == target)) ** 2
                        for index, value in enumerate(probabilities_valid)
                    )
                    / 2,
                    "nll": -math.log(max(probabilities_valid[target], 1e-12)),
                    "input_tokens": len(item["ids"]),
                }
                records.append(row)
                by_family[item["family"]].append(row)
    families = {
        name: {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "brier": sum(r["brier"] for r in rows) / len(rows),
        }
        for name, rows in sorted(by_family.items())
    }
    summary = {
        "tag": tag,
        "n": len(records),
        "correct": sum(row["correct"] for row in records),
        "micro_accuracy": sum(row["correct"] for row in records) / len(records),
        "family_macro_accuracy": sum(v["accuracy"] for v in families.values())
        / len(families),
        "family_macro_brier": sum(v["brier"] for v in families.values())
        / len(families),
        "by_family": families,
        "seconds": time.perf_counter() - started,
        "readout": "native Eikos last-token letter logits; T=1",
    }
    atomic_jsonl(output / f"{tag}-predictions.jsonl", records)
    atomic_json(output / f"{tag}-metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=REVISION)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        required=True,
        help="Rights-cleared TRAIN/SELECT/CAL receipt; publication-bound runs only",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument(
        "--select-limit", type=int, help="Preflight only; full pilot leaves this unset"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--brier-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PROMPT_STYLE", "semif") != "semif":
        raise ValueError("Eikos-4B requires the released semif prompt style")
    if args.model_revision != REVISION:
        raise ValueError("Only the pinned Eikos-4B source revision is supported")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Eikos pilot: {args.output}"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One BF16 ROCm/CUDA device is required")
    if (
        args.max_length < 1
        or args.microbatch < 1
        or args.accumulation < 1
        or args.eval_batch < 1
        or args.save_every < 1
        or (args.select_limit is not None and args.select_limit < 1)
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        raise ValueError(
            "Batch, context, step and checkpoint settings must be positive"
        )
    if not 0 <= args.lora_dropout < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid LoRA optimizer settings")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pilot supports one process and one GPU")

    release = verify_release(args.model_path, args.model_revision)
    train_rows_all = load_partition(args.train, "train")
    select_rows = load_partition(args.select, "select")
    cal_rows = load_partition(args.cal, "cal")
    check_partition_isolation(
        {"train": train_rows_all, "select": select_rows, "cal": cal_rows}
    )
    rights = verify_clean_files(
        args.data_manifest,
        args.train,
        args.select,
        args.cal,
        (len(train_rows_all), len(select_rows), len(cal_rows)),
    )
    select_max_options = max(len(row["options"]) for row in select_rows)
    cal_max_options = max(len(row["options"]) for row in cal_rows)
    if select_max_options > MAX_ONE_PASS or cal_max_options > MAX_ONE_PASS:
        raise ValueError(
            "SELECT/CAL exceed native Eikos one-pass limit; rework evaluator before proceeding"
        )
    train_rows, quarantined = one_pass_quarantine(train_rows_all)
    if args.select_limit is not None:
        select_rows = select_rows[: args.select_limit]
    if not select_rows:
        raise ValueError("SELECT partition is empty")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    sys.path.insert(0, str(args.model_path.resolve()))
    import decision_core as core
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if core.PROMPT_VERSION != PROMPT_VERSION:
        raise ValueError("Source Eikos prompt contract changed")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer lacks PAD and EOS IDs")
    train_items = [
        encode(row, tokenizer, core, max_length=args.max_length, shuffle_seed=args.seed)
        for row in train_rows
    ]
    select_items = [
        encode(row, tokenizer, core, max_length=args.max_length) for row in select_rows
    ]
    data_hashes = {
        "train": file_sha256(args.train),
        "select": file_sha256(args.select),
        "cal_audited_only": file_sha256(args.cal),
    }
    code_hashes = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("data.py", "rights.py", "train.py")
    }
    source_hashes = {
        name: file_sha256(args.model_path / name)
        for name in ("decision_core.py", "letter_adapter.py", "serve.py", "calib.json")
    }
    max_options = max(len(item["keys"]) for item in train_items)
    if max_options > MAX_ONE_PASS:
        raise AssertionError("Quarantine did not enforce the native one-pass policy")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    model.config.use_cache = False
    targets = select_target_modules(model.model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type=None,
        ),
    )
    model.get_base_model().gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.get_base_model().enable_input_require_grads()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        parameter.dtype != torch.float32 for parameter in trainable
    ):
        raise RuntimeError("Eikos LoRA must expose FP32 trainable adapters only")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=True
    )
    device = torch.device("cuda:0")
    lengths = [len(item["ids"]) for item in train_items]
    batches = epoch_batches(
        lengths,
        [],
        epoch=0,
        seed=args.seed,
        microbatch=args.microbatch,
        replay_fraction=0.0,
    )
    planned = math.ceil(len(batches) / args.accumulation)
    if args.max_steps is not None:
        planned = min(planned, args.max_steps)
    args.output.mkdir(parents=True)
    atomic_jsonl(args.output / "quarantine.jsonl", quarantined)
    quarantine_hash = file_sha256(args.output / "quarantine.jsonl")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "caiovicentino1/Eikos-4B",
        "model_revision": REVISION,
        "source_release": release,
        "source_code_sha256": source_hashes,
        "prompt_version": PROMPT_VERSION,
        "readout": "native last-token letter logits",
        "source_calibration": "T=1",
        "training_objective": "CE plus categorical Brier sum",
        "loss_implementation": LOSS_VERSION,
        "data_sha256": data_hashes,
        "code_sha256": code_hashes,
        "rights_manifest_sha256": file_sha256(args.data_manifest),
        "rights_schema_version": rights["schema_version"],
        "publication_eligible": rights["publication_eligible"],
        "publication_scope": rights["publication_scope"],
        "train_input_examples": len(train_rows_all),
        "train_effective_examples": len(train_items),
        "train_quarantined_examples": len(quarantined),
        "quarantine_sha256": quarantine_hash,
        "effective_train_payload_sha256": digest(train_rows),
        "select_examples": len(select_items),
        "cal_examples_audited_only": len(cal_rows),
        "select_max_options": select_max_options,
        "cal_max_options": cal_max_options,
        "train_tokens": sum(lengths),
        "train_max_tokens": max(lengths),
        "max_options": max_options,
        "native_max_one_pass_options": MAX_ONE_PASS,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": targets,
            "trainable_parameters": sum(p.numel() for p in trainable),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "brier_weight": args.brier_weight,
            "microbatch": args.microbatch,
            "accumulation": args.accumulation,
            "effective_batch": args.microbatch * args.accumulation,
            "epochs": 1,
            "planned_steps": planned,
            "save_every": args.save_every,
            "seed": args.seed,
            "max_length": args.max_length,
        },
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "selection_policy": "SELECT family-macro accuracy, then Brier, then earliest step",
        "calibration_policy": "CAL labels never produce gradients or select checkpoints",
    }
    atomic_json(args.output / "provenance.json", provenance)

    def emit(event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(line, flush=True)

    baseline = evaluate(
        model,
        select_items,
        pad_id=pad_id,
        batch_size=args.eval_batch,
        device=device,
        output=args.output,
        tag="select-baseline",
    )
    emit({"event": "baseline", "metrics": baseline})
    best_key = (baseline["family_macro_accuracy"], -baseline["family_macro_brier"], 0)
    atomic_json(
        args.output / "BEST.json", {"checkpoint": "source", "metrics": baseline}
    )
    model.train()
    for step in range(planned):
        window = batches[step * args.accumulation : (step + 1) * args.accumulation]
        examples = sum(len(batch) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        total = ce_total = brier_total = 0.0
        tokens = correct = 0
        started = time.perf_counter()
        for indices in window:
            items = [
                train_items[index] for source, index in indices if source == "train"
            ]
            batch = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in collate(items, pad_id).items()
            }
            logits = letter_logits(model, batch)
            terms = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=args.brier_weight,
            )
            loss = terms["total"].sum() / examples
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite Eikos loss at step {step + 1}")
            loss.backward()
            total += terms["total"].detach().sum().item()
            ce_total += terms["ce"].detach().sum().item()
            brier_total += terms["brier"].detach().sum().item()
            correct += (logits.argmax(-1) == batch["labels"]).sum().item()
            tokens += batch["attention_mask"].sum().item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Nonfinite Eikos gradient at step {step + 1}")
        optimizer.step()
        torch.cuda.synchronize(device)
        emit(
            {
                "event": "train",
                "step": step + 1,
                "examples": examples,
                "tokens": tokens,
                "loss": total / examples,
                "ce": ce_total / examples,
                "brier_sum": brier_total / examples,
                "accuracy": correct / examples,
                "gradient_norm": gradient_norm.item(),
                "seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
        count = step + 1
        if count % args.save_every != 0 and count != planned:
            continue
        metric = evaluate(
            model,
            select_items,
            pad_id=pad_id,
            batch_size=args.eval_batch,
            device=device,
            output=args.output,
            tag=f"select-step-{count:04d}",
        )
        emit({"event": "select", "step": count, "metrics": metric})
        checkpoint = args.output / f"checkpoint-{count:04d}"
        pending = args.output / f"checkpoint-{count:04d}.pending"
        if pending.exists() or checkpoint.exists():
            raise FileExistsError(checkpoint)
        pending.mkdir()
        model.save_pretrained(pending / "adapter", safe_serialization=True)
        atomic_json(
            pending / "checkpoint.json",
            {
                "step": count,
                "metrics": metric,
                "source_release": release,
                "prompt_version": PROMPT_VERSION,
                "readout": "native Eikos letter logits",
            },
        )
        os.replace(pending, checkpoint)
        candidate_key = (
            metric["family_macro_accuracy"],
            -metric["family_macro_brier"],
            -count,
        )
        if candidate_key > best_key:
            best_key = candidate_key
            atomic_json(
                args.output / "BEST.json",
                {"checkpoint": checkpoint.name, "metrics": metric},
            )
        model.train()
    atomic_json(
        args.output / "COMPLETE.json",
        {
            "status": "complete",
            "steps": planned,
            "best": json.loads((args.output / "BEST.json").read_text())["checkpoint"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_status": "unopened_for_model_selection",
        },
    )
    emit({"event": "complete", "steps": planned})


if __name__ == "__main__":
    main()
