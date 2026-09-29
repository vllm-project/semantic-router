"""Explicit single-GPU classifier training with optional development selection."""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .data import assert_disjoint, file_receipts, label_counts, read_records
from .evaluate import evaluate_records
from .optimization import (
    assert_frozen_parameters_preserved,
    configure_trainable_tail,
    frozen_parameter_receipt,
    initial_artifact_receipt,
    load_trainable_model,
    optimizer_groups,
    save_training_checkpoint,
    validate_method,
    verify_full_checkpoint,
)
from .retention import RetentionCache, masked_teacher_kl, tokenizer_file_receipt
from .selection import (
    BINARY_FP_SELECTION,
    binary_checkpoint_key,
    binary_selection_support,
    fit_binary_operating_point,
    validate_selection_options,
)
from .task_head import task_head_scope
from .training_order import load_training_order


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


def global_batch_loss(logits, labels, examples_per_step):
    """Normalize variable microbatches by the same sampled optimizer-step size."""
    from torch.nn import functional

    if examples_per_step <= 0:
        raise ValueError("Optimizer-step example count must be positive")
    return (
        functional.cross_entropy(logits.float(), labels, reduction="sum")
        / examples_per_step
    )


def token_budget_microbatches(indices, lengths, token_budget):
    """Stable length ordering without dropping or duplicating sampled examples."""
    if token_budget <= 0:
        raise ValueError("Microbatch token budget must be positive")
    batches, current = [], []
    for index in sorted(indices, key=lengths.__getitem__):
        length = lengths[index]
        if length > token_budget:
            raise ValueError("A sampled example exceeds the microbatch token budget")
        if current and length * (len(current) + 1) > token_budget:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def selection_score(metrics, selection):
    if selection == "macro-f1":
        return metrics["macro_f1"]
    if selection in {
        "length-macro-f1",
        "source-macro-f1",
        "source-present-macro-f1",
    }:
        field = "length_bucket" if selection == "length-macro-f1" else "source"
        metric = (
            "macro_f1_present_labels"
            if selection == "source-present-macro-f1"
            else "macro_f1"
        )
        groups = list(metrics["breakdowns"][field].values())
        if not groups:
            raise ValueError("Grouped selection requires measured groups")
        return sum(group[metric] for group in groups) / len(groups)
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


def source_sampling_weights(rows, specification):
    """Resolve explicit probabilities in the same order as source pools."""
    if specification is None:
        return None
    weights = json.loads(specification)
    sources = list(dict.fromkeys(row.get("source") for row in rows))
    if (
        not sources
        or not all(sources)
        or not isinstance(weights, dict)
        or set(weights) != set(sources)
    ):
        raise ValueError("Source weights must name every eligible source exactly")
    values = [weights[source] for source in sources]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError("Source weights must be finite positive numbers")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("Total source weight must be finite")
    return [value / total for value in values]


def sample_source_balanced(pools, rng, balance_labels, source_weights=None):
    source = (
        rng.choice(pools)
        if source_weights is None
        else rng.choices(pools, weights=source_weights, k=1)[0]
    )
    return sample_length_balanced(source, rng, balance_labels)


def sampling_exposure(rows, drawn):
    """Report sampling counts separately from class/length selection metrics."""
    sources = defaultdict(
        lambda: {"rows": 0, "draws": 0, "unique_rows": 0, "label_draws": Counter()}
    )
    for index, row in enumerate(rows):
        item = sources[row.get("source", "unspecified")]
        count = drawn[index]
        item["rows"] += 1
        item["draws"] += count
        item["unique_rows"] += int(count > 0)
        item["label_draws"][row["label"]] += count
    return dict(sources)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--fresh-head", action="store_true")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--trainable-last-layers", type=int)
    parser.add_argument("--retention-targets", type=Path)
    parser.add_argument("--retention-weight", type=float)
    parser.add_argument("--retention-temperature", type=float)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--evaluation-dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--selection",
        choices=[
            "macro-f1",
            "length-macro-f1",
            "source-macro-f1",
            "source-present-macro-f1",
            BINARY_FP_SELECTION,
        ],
        default="macro-f1",
    )
    parser.add_argument("--positive-label")
    parser.add_argument("--selection-false-positive-budget", type=float)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--length-balanced-sampling", action="store_true")
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument(
        "--source-weights", help="JSON object of explicit source weights"
    )
    parser.add_argument("--microbatch-token-budget", type=int)
    parser.add_argument("--training-order", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="Skip development data and selection; save only the final checkpoint",
    )
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.train_only:
        if args.dev:
            parser.error("--train-only must not receive development data")
        if (
            args.selection != "macro-f1"
            or args.positive_label is not None
            or args.selection_false_positive_budget is not None
        ):
            parser.error("--train-only does not select a development operating point")
    elif not args.dev:
        parser.error("--dev is required unless --train-only is set")
    validate_method(args.method, args.adapter, args.fresh_head)
    if args.trainable_last_layers is not None and args.method != "full":
        raise ValueError("A trainable encoder tail requires full checkpoint training")
    if args.retention_targets is None:
        if args.retention_weight is not None or args.retention_temperature is not None:
            raise ValueError("Retention settings require explicit TRAIN targets")
    else:
        if args.method != "full" or args.fresh_head:
            raise ValueError("Retention requires the unchanged complete initial head")
        args.retention_weight = (
            0.5 if args.retention_weight is None else args.retention_weight
        )
        args.retention_temperature = (
            2.0 if args.retention_temperature is None else args.retention_temperature
        )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (args.retention_weight, args.retention_temperature)
        ):
            raise ValueError(
                "Retention weight and temperature must be finite and positive"
            )
    if args.training_order and (
        args.balanced_sampling
        or args.length_balanced_sampling
        or args.source_balanced_sampling
        or args.source_weights is not None
    ):
        raise ValueError("An explicit training order cannot also configure sampling")
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
        or (
            args.microbatch_token_budget is not None
            and args.microbatch_token_budget <= 0
        )
    ):
        raise ValueError("Training budgets and learning rate must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a training run")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    initial_artifacts = initial_artifact_receipt(args.base, args.adapter)
    model, tokenizer, label_to_id, id_to_label = load_trainable_model(
        args.base, args.contract, args.adapter, args.method, args.fresh_head
    )
    if model.config.problem_type != "single_label_classification":
        raise ValueError("This training loop requires single-label targets")
    if args.retention_targets is not None:
        verify_full_checkpoint(model, args.base)
    trainable_scope = (
        configure_trainable_tail(model, args.trainable_last_layers)
        if args.trainable_last_layers is not None
        else None
    )
    frozen_parameters = (
        frozen_parameter_receipt(model) if trainable_scope is not None else None
    )
    if not args.train_only:
        validate_selection_options(
            args.selection,
            label_to_id,
            args.positive_label,
            args.selection_false_positive_budget,
        )
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Training budget exceeds checkpoint position capacity")
    rows, dev = (
        read_records(args.train, label_to_id),
        [] if args.train_only else read_records(args.dev, label_to_id),
    )
    assert_disjoint(rows, dev)
    if args.selection == BINARY_FP_SELECTION:
        binary_selection_support(dev, label_to_id, args.positive_label)
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
    retention = None
    if args.retention_targets is not None:
        if any(
            not isinstance(row.get("retention_replay", False), bool) for row, _ in train
        ):
            raise ValueError("retention_replay must be a boolean TRAIN annotation")
        retention = RetentionCache.load(
            args.retention_targets,
            initialization=initial_artifacts,
            contract_sha256=hashlib.sha256(
                Path(args.contract).read_bytes()
            ).hexdigest(),
            tokenizer_files=tokenizer_file_receipt(args.base),
            train_rows=[row for row, _ in train],
            tokenized_rows={row["id"]: encoded for row, encoded in train},
            label_to_id=label_to_id,
            replay_ids=[
                row["id"] for row, _ in train if row.get("retention_replay", False)
            ],
        )
    train_lengths = [len(encoded["input_ids"]) for _, encoded in train]
    if (
        args.microbatch_token_budget is not None
        and max(train_lengths) > args.microbatch_token_budget
    ):
        raise ValueError(
            "Microbatch token budget cannot fit the longest eligible example"
        )
    training_order = (
        load_training_order(
            args.training_order,
            [row for row, _ in train],
            args.train,
            steps=args.steps,
            global_batch=args.batch_size * args.accumulate,
        )
        if args.training_order
        else None
    )
    pool_values = list(pools.values())
    length_pools = length_sampling_pools([row for row, _encoded in train])
    balance_sources = args.source_balanced_sampling or args.source_weights is not None
    source_weights = source_sampling_weights(
        [row for row, _encoded in train], args.source_weights
    )
    source_pools = (
        source_sampling_pools(
            [row for row, _encoded in train], args.length_balanced_sampling
        )
        if balance_sources
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
    groups = optimizer_groups(model, args.learning_rate, args.head_learning_rate)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "method": args.method,
        "fresh_head": args.fresh_head,
        "trainable_scope": trainable_scope,
        "frozen_parameters": frozen_parameters,
        "retention": (
            {
                "targets": retention.receipt,
                "weight": args.retention_weight,
                "temperature": args.retention_temperature,
                "normalization": "all examples in the optimizer step",
            }
            if retention is not None
            else None
        ),
        "initial_artifacts": initial_artifacts,
        "initial_adapter_sha256": initial_artifacts.get("adapter_model.safetensors"),
        "initial_adapter_config_sha256": initial_artifacts.get("adapter_config.json"),
        "contract_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        "classifier_pooling": model.config.classifier_pooling,
        "task_head": task_head_scope(model),
        "problem_type": model.config.problem_type,
        "train_files": file_receipts(args.train),
        "dev_files": file_receipts(args.dev or []),
        "train_only": args.train_only,
        "checkpoint_policy": (
            "final-only" if args.train_only else "development-and-final"
        ),
        "label2id": label_to_id,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_label_counts": label_counts([row for row, _ in train]),
        "rejected": rejected,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accumulate": args.accumulate,
        "examples_per_optimizer_step": args.batch_size * args.accumulate,
        "microbatch_token_budget": args.microbatch_token_budget,
        "dropout_trajectory_equivalence": args.microbatch_token_budget is None,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "optimizer_groups": [
            {
                "name": group["name"],
                "learning_rate": group["initial_lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in groups
        ],
        "balanced_sampling": args.balanced_sampling,
        "training_order": training_order.receipt if training_order else None,
        "length_balanced_sampling": args.length_balanced_sampling,
        "source_balanced_sampling": balance_sources,
        "source_sampling_probabilities": (
            dict(
                zip(
                    dict.fromkeys(row["source"] for row, _ in train),
                    source_weights,
                    strict=True,
                )
            )
            if source_weights is not None
            else None
        ),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "selection": None if args.train_only else args.selection,
        "evaluation_dtype": None if args.train_only else args.evaluation_dtype,
        "test_used": False,
        "precision": "FP32 parameters and explicit FP32 mean CE / BF16 autocast",
        "attention": "sdpa",
        "checkpointing": "non-reentrant",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }
    if args.selection == BINARY_FP_SELECTION:
        metadata["binary_selection"] = {
            "positive_label": args.positive_label,
            "false_positive_budget": args.selection_false_positive_budget,
            "support": binary_selection_support(dev, label_to_id, args.positive_label),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).with_name("selection.py").read_bytes()
            ).hexdigest(),
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
    best_key = None
    drawn = Counter()

    def evaluate_checkpoint(step):
        nonlocal best, best_key
        metrics, predictions = evaluate_records(
            model,
            tokenizer,
            dev,
            label_to_id,
            id_to_label,
            args.max_length,
            dtype=args.evaluation_dtype,
        )
        operating_point = None
        if args.selection == BINARY_FP_SELECTION:
            operating_point = fit_binary_operating_point(
                dev,
                predictions,
                label_to_id,
                args.positive_label,
                args.selection_false_positive_budget,
            )
            metrics["binary_fp_budget"] = operating_point
            key = binary_checkpoint_key(operating_point)
            score = operating_point["recall"] if key is not None else -1.0
            (args.output / f"dev-predictions-step-{step}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
        else:
            score = selection_score(metrics, args.selection)
            key = (score,)
        (args.output / f"sampling-step-{step}.json").write_text(
            json.dumps(sampling_exposure([row for row, _ in train], drawn), indent=2)
            + "\n"
        )
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
        if key is not None and (best_key is None or key > best_key):
            best = score
            best_key = key
            if frozen_parameters is not None:
                assert_frozen_parameters_preserved(frozen_parameters, model)
            save_training_checkpoint(
                model,
                tokenizer,
                args.output
                / ("best-adapter" if args.method == "lora" else "best-model"),
                args.method,
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
                        **(
                            {"operating_point": operating_point}
                            if operating_point is not None
                            else {}
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )

    def draw_microbatch_indices():
        return [
            (
                sample_source_balanced(
                    source_pools, rng, args.balanced_sampling, source_weights
                )
                if balance_sources
                else (
                    sample_length_balanced(length_pools, rng, args.balanced_sampling)
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

    if not args.probe_only and not args.train_only:
        evaluate_checkpoint(0)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start, step_loss, lengths = time.perf_counter(), 0.0, []
        step_ce_loss, step_retention_loss, retention_examples = 0.0, 0.0, 0
        if training_order:
            planned = training_order.indices_for_step(step)
            microbatches = (
                token_budget_microbatches(
                    planned, train_lengths, args.microbatch_token_budget
                )
                if args.microbatch_token_budget is not None
                else [
                    planned[start : start + args.batch_size]
                    for start in range(0, len(planned), args.batch_size)
                ]
            )
        elif args.microbatch_token_budget is None:
            microbatches = (draw_microbatch_indices() for _ in range(args.accumulate))
        else:
            sampled = [
                index
                for _ in range(args.accumulate)
                for index in draw_microbatch_indices()
            ]
            microbatches = token_budget_microbatches(
                sampled, train_lengths, args.microbatch_token_budget
            )
        microbatch_count, max_padded_tokens = 0, 0
        for indices in microbatches:
            microbatch_count += 1
            max_padded_tokens = max(
                max_padded_tokens, max(train_lengths[i] for i in indices) * len(indices)
            )
            drawn.update(indices)
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
                loss = (
                    microbatch_loss(logits, gold, args.accumulate)
                    if args.microbatch_token_budget is None
                    else global_batch_loss(
                        logits, gold, args.batch_size * args.accumulate
                    )
                )
            step_ce_loss += float(loss.detach())
            if retention is not None:
                teacher_logits, eligible = retention.targets(
                    [row for row, _ in selected], device=logits.device
                )
                preservation = args.retention_weight * masked_teacher_kl(
                    logits,
                    teacher_logits,
                    eligible,
                    examples_per_step=args.batch_size * args.accumulate,
                    temperature=args.retention_temperature,
                )
                loss = loss + preservation
                step_retention_loss += float(preservation.detach())
                retention_examples += int(eligible.sum())
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
            group["lr"] = group["initial_lr"] * rate
        optimizer.step()
        if training_order:
            trace = training_order.record_step(step, microbatches)
            with (args.output / "actual-training-order.jsonl").open("a") as stream:
                stream.write(json.dumps(trace) + "\n")
        torch.cuda.synchronize()
        log = {
            "step": step,
            "loss": step_loss,
            "cross_entropy_loss": step_ce_loss,
            "retention_loss": step_retention_loss,
            "retention_examples": retention_examples,
            "grad_norm": float(norm),
            "finite": True,
            "gradient_tensors": len(gradients),
            "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "microbatches": microbatch_count,
            "max_padded_tokens": max_padded_tokens,
            "step_seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        with (args.output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(log) + "\n")
        if step == 1 or step % 10 == 0 or args.probe_only:
            print(json.dumps(log), flush=True)
        if (
            not args.probe_only
            and not args.train_only
            and (step % args.eval_every == 0 or step == args.steps)
        ):
            evaluate_checkpoint(step)
    if training_order:
        order_receipt = training_order.finish()
        order_receipt["actual_trace_sha256"] = hashlib.sha256(
            (args.output / "actual-training-order.jsonl").read_bytes()
        ).hexdigest()
        (args.output / "training-order-completed.json").write_text(
            json.dumps(order_receipt, indent=2) + "\n"
        )
    (args.output / "sampling-final.json").write_text(
        json.dumps(sampling_exposure([row for row, _ in train], drawn), indent=2) + "\n"
    )
    if frozen_parameters is not None:
        assert_frozen_parameters_preserved(frozen_parameters, model)
    if not args.probe_only:
        save_training_checkpoint(
            model,
            tokenizer,
            args.output / ("last-adapter" if args.method == "lora" else "last-model"),
            args.method,
            args.base_id,
            args.base_revision,
        )
        if best_key is None and not args.train_only:
            raise ValueError(
                "No checkpoint met the development FP budget; last checkpoint and "
                "infeasibility receipts were retained without selecting a best checkpoint"
            )
    print(
        json.dumps(
            {
                "event": "complete",
                "best_dev_score": (
                    best if not (args.probe_only or args.train_only) else None
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
