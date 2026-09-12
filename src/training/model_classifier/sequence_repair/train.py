"""Explicit single-GPU LoRA continuation with development-only selection."""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from .data import assert_disjoint, file_receipts, label_counts, read_records
from .evaluate import evaluate_records
from .model import load_model, save_adapter


def microbatch_loss(logits, labels, accumulation_steps):
    from torch.nn import functional

    if accumulation_steps <= 0:
        raise ValueError("Gradient accumulation must be positive")
    # Each microbatch has the same number of examples. HF Trainer loss-kwargs
    # normalization is intentionally not involved in this explicit loop.
    return (
        functional.cross_entropy(logits.float(), labels, reduction="mean")
        / accumulation_steps
    )


def selection_score(metrics, selection):
    if selection == "macro-f1":
        return metrics["macro_f1"]
    if selection in {"length-macro-f1", "source-macro-f1"}:
        field = "length_bucket" if selection == "length-macro-f1" else "source"
        groups = list(metrics["breakdowns"][field].values())
        if not groups:
            raise ValueError("Grouped selection requires measured groups")
        return sum(group["macro_f1"] for group in groups) / len(groups)
    raise ValueError("Unknown development selection metric")


def length_sampling_pools(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        buckets[str(row.get("length_bucket", "unspecified"))][row["label"]].append(
            index
        )
    return [list(labels.values()) for labels in buckets.values()]


def sample_length_balanced(pools, rng, balance_labels):
    bucket = rng.choice(pools)
    pool = (
        rng.choice(bucket)
        if balance_labels
        else [index for label in bucket for index in label]
    )
    return rng.choice(pool)


def source_sampling_pools(rows, balance_lengths):
    sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        if not row.get("source"):
            raise ValueError("Source-balanced sampling requires explicit sources")
        length = (
            str(row.get("length_bucket", "unspecified")) if balance_lengths else "all"
        )
        sources[row["source"]][length][row["label"]].append(index)
    return [
        [list(labels.values()) for labels in lengths.values()]
        for lengths in sources.values()
    ]


def sample_source_balanced(pools, rng, balance_labels):
    return sample_length_balanced(rng.choice(pools), rng, balance_labels)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", default="llm-semantic-router/mmbert-32k-yarn")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--selection",
        choices=["macro-f1", "length-macro-f1", "source-macro-f1"],
        default="macro-f1",
    )
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--length-balanced-sampling", action="store_true")
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.batch_size,
            args.accumulate,
            args.max_length,
            args.eval_every,
        )
        <= 0
        or args.learning_rate <= 0
    ):
        raise ValueError("Training budgets and learning rate must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a training run")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, tokenizer, label_to_id, id_to_label = load_model(
        args.base, args.contract, args.adapter, trainable=True
    )
    if model.config.problem_type != "single_label_classification":
        raise ValueError("This training loop requires single-label targets")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Training budget exceeds checkpoint position capacity")
    rows, dev = read_records(args.train, label_to_id), read_records(
        args.dev, label_to_id
    )
    assert_disjoint(rows, dev)
    train, rejected, pools = [], [], defaultdict(list)
    for row in rows:
        encoded = tokenizer(row["text"], truncation=False, padding=False)
        if len(encoded["input_ids"]) > args.max_length:
            rejected.append(
                {
                    "id": row["id"],
                    "reason": "explicit context budget",
                    "tokens": len(encoded["input_ids"]),
                }
            )
            continue
        index = len(train)
        train.append((row, encoded))
        pools[row["label"]].append(index)
    if not train:
        raise ValueError("No training rows fit the explicit budget")
    pool_values = list(pools.values())
    length_pools = length_sampling_pools([row for row, _encoded in train])
    source_pools = (
        source_sampling_pools(
            [row for row, _encoded in train], args.length_balanced_sampling
        )
        if args.source_balanced_sampling
        else None
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to("cuda")
    model.train()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "initial_adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "train_files": file_receipts(args.train),
        "dev_files": file_receipts(args.dev),
        "label2id": label_to_id,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_label_counts": label_counts([row for row, _ in train]),
        "rejected": rejected,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accumulate": args.accumulate,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "balanced_sampling": args.balanced_sampling,
        "length_balanced_sampling": args.length_balanced_sampling,
        "source_balanced_sampling": args.source_balanced_sampling,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "selection": args.selection,
        "test_used": False,
        "precision": "FP32 parameters and explicit FP32 mean CE / BF16 autocast",
        "attention": "sdpa",
        "checkpointing": "non-reentrant",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "ready",
                "train_rows": len(train),
                "rejected_rows": len(rejected),
                "trainable_parameters": metadata["trainable_parameters"],
            }
        ),
        flush=True,
    )
    best = -1.0

    def evaluate_checkpoint(step):
        nonlocal best
        metrics, _predictions = evaluate_records(
            model, tokenizer, dev, label_to_id, id_to_label, args.max_length
        )
        score = selection_score(metrics, args.selection)
        (args.output / f"dev-step-{step}.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "event": "dev",
                    "step": step,
                    "score": score,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            ),
            flush=True,
        )
        if score > best:
            best = score
            save_adapter(
                model,
                tokenizer,
                args.output / "best-adapter",
                args.base_id,
                args.base_revision,
            )
            (args.output / "selection.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "score": score,
                        "metric": args.selection,
                        "test_used": False,
                    },
                    indent=2,
                )
                + "\n"
            )

    if not args.probe_only:
        evaluate_checkpoint(0)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start, step_loss, lengths = time.perf_counter(), 0.0, []
        for _ in range(args.accumulate):
            indices = [
                (
                    sample_source_balanced(source_pools, rng, args.balanced_sampling)
                    if args.source_balanced_sampling
                    else (
                        sample_length_balanced(
                            length_pools, rng, args.balanced_sampling
                        )
                        if args.length_balanced_sampling
                        else (
                            rng.choice(rng.choice(pool_values))
                            if args.balanced_sampling
                            else rng.randrange(len(train))
                        )
                    )
                )
                for _ in range(args.batch_size)
            ]
            selected = [train[index] for index in indices]
            lengths.extend(len(encoded["input_ids"]) for _, encoded in selected)
            batch = tokenizer.pad(
                [encoded for _, encoded in selected], padding=True, return_tensors="pt"
            )
            batch = {key: value.to("cuda") for key, value in batch.items()}
            gold = torch.tensor(
                [label_to_id[row["label"]] for row, _ in selected], device="cuda"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                loss = microbatch_loss(logits, gold, args.accumulate)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Non-finite loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
            del logits, loss, batch
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ):
            raise ValueError(f"Missing or non-finite gradients at step {step}")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        warmup = max(1, round(args.steps * 0.06))
        rate = (
            step / warmup
            if step <= warmup
            else 0.5
            * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * rate
        optimizer.step()
        torch.cuda.synchronize()
        log = {
            "step": step,
            "loss": step_loss,
            "grad_norm": float(norm),
            "finite": True,
            "gradient_tensors": len(gradients),
            "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "step_seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        with (args.output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(log) + "\n")
        if step == 1 or step % 10 == 0 or args.probe_only:
            print(json.dumps(log), flush=True)
        if not args.probe_only and (step % args.eval_every == 0 or step == args.steps):
            evaluate_checkpoint(step)
    if not args.probe_only:
        save_adapter(
            model,
            tokenizer,
            args.output / "last-adapter",
            args.base_id,
            args.base_revision,
        )
    print(
        json.dumps(
            {
                "event": "complete",
                "best_dev_score": best if not args.probe_only else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
