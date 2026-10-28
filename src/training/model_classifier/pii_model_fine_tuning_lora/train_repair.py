# Heavy inference dependencies are lazy so data-contract tests stay dependency-light.
# ruff: noqa: PLC0415
"""Train a PII adapter or full token classifier with explicit runtime budgets."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from evaluate import evaluate_records, load_model
from full_training import (
    initial_artifact_receipt,
    optimizer_groups,
    save_full_checkpoint,
    tensor_receipt,
    validate_method,
)
from span_data import align_record, read_jsonl
from token_loss import document_mean_loss, entity_document_mean_loss, pad_entity_ids


def checkpoint_score(metrics, selection):
    if selection == "micro-f1":
        return metrics["micro"]["f1"]
    if selection == "length-macro-f1":
        groups = metrics["breakdowns"]["length_bucket"].values()
        scores = [group["micro"]["f1"] for group in groups]
        if not scores:
            raise ValueError("Length-aware selection needs development length groups")
        return sum(scores) / len(scores)
    raise ValueError(f"Unknown checkpoint selection metric: {selection}")


def prepare_records(
    records, tokenizer, label_to_id, max_length, *, include_entity_ids=False
):
    accepted, rejected = [], []
    for record in records:
        try:
            encoded = align_record(
                record,
                tokenizer,
                label_to_id,
                max_length,
                include_entity_ids=include_entity_ids,
            )
        except ValueError as error:
            if "budget" not in str(error) and "boundary crosses a token" not in str(
                error
            ):
                raise ValueError(
                    f"Invalid training record {record.get('id')}: {error}"
                ) from error
            rejected.append({"id": record.get("id"), "reason": str(error)})
            continue
        accepted.append((record, encoded))
    return accepted, rejected


def save_adapter(model, tokenizer, directory, base_id, base_revision, label_to_id):
    directory.mkdir(parents=True, exist_ok=True)
    config = model.peft_config["default"]
    config.base_model_name_or_path = base_id
    config.revision = base_revision
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    (directory / "label_mapping.json").write_text(
        json.dumps(
            {
                "label_to_id": label_to_id,
                "id_to_label": {
                    str(index): label for label, index in label_to_id.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


def main():
    import torch
    from transformers import DataCollatorForTokenClassification

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--fresh-head", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--replay")
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument(
        "--loss-normalization",
        choices=["token_mean", "document_mean", "entity_document_mean"],
        default="token_mean",
        help="Token mean, document mean, or equal entity/O mass per document",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--evaluation-dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--replay-probability", type=float, default=0.4)
    parser.add_argument("--replay-balance-entities", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--selection-metric",
        choices=["micro-f1", "length-macro-f1"],
        default="micro-f1",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--evaluate-initial", action="store_true")
    args = parser.parse_args()
    validate_method(args.method, args.adapter, args.fresh_head)
    if args.method == "lora" and args.head_learning_rate is not None:
        raise ValueError("A separate head learning rate requires full training")
    if (
        min(
            args.steps,
            args.batch_size,
            args.accumulate,
            args.max_length,
            args.eval_every,
        )
        <= 0
    ):
        raise ValueError("Step, batch and context budgets must be positive")
    if not 0 <= args.replay_probability <= 1:
        raise ValueError("Invalid replay probability")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a training run")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    initial_artifact = (
        initial_artifact_receipt(args.base) if args.method == "full" else None
    )
    model, tokenizer, label_to_id, id_to_label = load_model(
        args.base, args.config, args.adapter, trainable=True, fresh_head=args.fresh_head
    )
    if (
        initial_artifact is not None
        and initial_artifact_receipt(args.base) != initial_artifact
    ):
        raise ValueError("Base files changed during full-model initialization")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Budget exceeds checkpoint capacity")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to(args.device)
    model.train()
    train, rejected = prepare_records(
        read_jsonl(args.train),
        tokenizer,
        label_to_id,
        args.max_length,
        include_entity_ids=args.loss_normalization == "entity_document_mean",
    )
    replay, replay_rejected = (
        prepare_records(
            read_jsonl(args.replay),
            tokenizer,
            label_to_id,
            args.max_length,
            include_entity_ids=args.loss_normalization == "entity_document_mean",
        )
        if args.replay
        else ([], [])
    )
    dev = read_jsonl(args.dev)
    replay_by_type = defaultdict(list)
    for item in replay:
        for entity_type in {span["entity_type"] for span in item[0]["spans"]} or {
            "negative"
        }:
            replay_by_type[entity_type].append(item)
    replay_pools = list(replay_by_type.values())
    if not train or not dev:
        raise ValueError("Training and development sets must both be nonempty")
    groups = (
        optimizer_groups(
            model, args.learning_rate, args.head_learning_rate or args.learning_rate
        )
        if args.method == "full"
        else None
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        groups or parameters, lr=args.learning_rate, weight_decay=0.01
    )
    collator = DataCollatorForTokenClassification(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8 if args.max_length % 8 == 0 else None,
        return_tensors="pt",
    )
    metadata = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "method": args.method,
        "task": "token-classification",
        "architecture": type(model).__name__,
        "fresh_head": args.fresh_head,
        "initial_artifact": initial_artifact,
        "initial_tensors": tensor_receipt(model) if args.method == "full" else None,
        "contract_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "seed": args.seed,
        "label2id": label_to_id,
        "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(Path(args.dev).read_bytes()).hexdigest(),
        "replay_sha256": (
            hashlib.sha256(Path(args.replay).read_bytes()).hexdigest()
            if args.replay
            else None
        ),
        "train_rows": len(train),
        "replay_rows": len(replay),
        "rejections": rejected + replay_rejected,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "optimizer_groups": [
            {
                "name": group.get("name", "adapter"),
                "learning_rate": group["lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ],
        "attention": "sdpa",
        "checkpointing": "non-reentrant",
        "dtype": (
            "FP32 parameters and BF16 autocast" if args.device == "cuda" else "float32"
        ),
        "evaluation_dtype": args.evaluation_dtype,
        "device_name": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
        ),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulate,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate or args.learning_rate,
        "loss_normalization": args.loss_normalization,
        "loss_reduction": (
            "mean over nonignored token labels in each microbatch"
            if args.loss_normalization == "token_mean"
            else (
                "mean attended nonignored token CE per document, then mean over logical documents"
                if args.loss_normalization == "document_mean"
                else "half mean entity CE and half mean O CE per document; absent groups omitted; then mean over logical documents"
            )
        ),
        "replay_probability": args.replay_probability,
        "replay_balance_entities": args.replay_balance_entities,
        "probe_only": args.probe_only,
        "selection": args.selection_metric,
        "test_used": False,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def save_checkpoint(kind):
        if args.method == "full":
            save_full_checkpoint(
                model, tokenizer, args.output / f"{kind}-model", metadata
            )
        else:
            save_adapter(
                model,
                tokenizer,
                args.output / f"{kind}-adapter",
                args.base_id,
                args.base_revision,
                label_to_id,
            )

    print(
        json.dumps(
            {
                "event": "ready",
                **{
                    key: value
                    for key, value in metadata.items()
                    if key not in ("label2id", "rejections", "initial_tensors")
                },
                "rejected_rows": len(rejected) + len(replay_rejected),
            }
        ),
        flush=True,
    )
    best = -1.0
    if args.evaluate_initial and not args.probe_only:
        metrics, _predictions = evaluate_records(
            model, tokenizer, dev, id_to_label, dtype=args.evaluation_dtype
        )
        best = checkpoint_score(metrics, args.selection_metric)
        (args.output / "dev-step-0.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        save_checkpoint("best")
        (args.output / "selection.json").write_text(
            json.dumps(
                {
                    "step": 0,
                    "development_f1": best,
                    "metric": args.selection_metric,
                    "test_used": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"event": "initial_dev", "selection_score": best}), flush=True)
    log = args.output / "steps.jsonl"
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        step_loss = 0.0
        tokens = []
        for _ in range(args.accumulate):
            selected = []
            for _ in range(args.batch_size):
                pool = (
                    (
                        rng.choice(replay_pools)
                        if args.replay_balance_entities
                        else replay
                    )
                    if replay and rng.random() < args.replay_probability
                    else train
                )
                selected.append(rng.choice(pool)[1])
            tokens.extend(len(item["input_ids"]) for item in selected)
            batch = collator(
                [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"offset_mapping", "entity_ids"}
                    }
                    for item in selected
                ]
            )
            batch = {key: value.to(args.device) for key, value in batch.items()}
            entity_ids = (
                pad_entity_ids(
                    selected, batch["labels"].shape[1], tokenizer.padding_side
                ).to(args.device)
                if args.loss_normalization == "entity_document_mean"
                else None
            )
            with torch.autocast(
                args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"
            ):
                if args.loss_normalization == "token_mean":
                    outputs = model(**batch)
                    loss = outputs.loss / args.accumulate
                else:
                    outputs = model(
                        **{
                            key: value
                            for key, value in batch.items()
                            if key != "labels"
                        }
                    )
                    loss = (
                        document_mean_loss(
                            outputs.logits, batch["labels"], batch["attention_mask"]
                        )
                        if args.loss_normalization == "document_mean"
                        else entity_document_mean_loss(
                            outputs.logits,
                            batch["labels"],
                            batch["attention_mask"],
                            entity_ids,
                            o_label_id=label_to_id["O"],
                        )
                    )
                    loss = loss / args.accumulate
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Non-finite loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
            del outputs, loss, batch
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if (
            (args.method == "full" and len(gradients) != len(parameters))
            or not gradients
            or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        ):
            raise ValueError(f"Missing or non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        warmup = max(1, round(args.steps * 0.06))
        rate = (
            step / warmup
            if step <= warmup
            else 0.5
            * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
        )
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", args.learning_rate) * rate
        optimizer.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        row = {
            "step": step,
            "loss": step_loss,
            "grad_norm": float(grad_norm),
            "finite": True,
            "parameters_with_gradient": len(gradients),
            "max_input_tokens": max(tokens),
            "mean_input_tokens": sum(tokens) / len(tokens),
            "step_seconds": time.perf_counter() - started,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if args.device == "cuda"
                else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
                if args.device == "cuda"
                else None
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "group_learning_rates": {
                group.get("name", "adapter"): group["lr"]
                for group in optimizer.param_groups
            },
        }
        with log.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if step == 1 or step % 10 == 0 or args.probe_only:
            print(json.dumps(row), flush=True)
        if args.probe_only:
            continue
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _predictions = evaluate_records(
                model, tokenizer, dev, id_to_label, dtype=args.evaluation_dtype
            )
            score = checkpoint_score(metrics, args.selection_metric)
            (args.output / f"dev-step-{step}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "event": "dev",
                        "step": step,
                        "micro": metrics["micro"],
                        "complete_email": metrics["complete_email"],
                        "negative_documents": metrics["negative_documents"],
                    }
                ),
                flush=True,
            )
            if score > best:
                best = score
                save_checkpoint("best")
                (args.output / "selection.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "development_f1": best,
                            "metric": args.selection_metric,
                            "test_used": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    if not args.probe_only:
        save_checkpoint("last")
    print(
        json.dumps(
            {
                "event": "complete",
                "steps": args.steps,
                "best_dev_f1": best if not args.probe_only else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
