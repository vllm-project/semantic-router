"""Single-GPU Vela Hazard training with masked BCE and development-only selection."""

# Torch is optional for dependency-light data checks.
# ruff: noqa: PLC0415

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, file_receipts
from ..sequence_repair.optimization import (
    initial_artifact_receipt,
    load_trainable_model,
    optimizer_groups,
    save_training_checkpoint,
    validate_method,
)
from ..sequence_repair.train import token_budget_microbatches
from ..sequence_repair.training_order import load_training_order
from .vela_hazard import evaluate, masked_loss, read_rows
from .vela_hazard_diagnostics import SupervisionDiagnostics
from .vela_hazard_joint import safe_group_indices, select_joint_operating_point
from .vela_hazard_operating import select_operating_point

SAFE_SAMPLE_FRACTION = 0.3


def grouped_pools(rows, label_count, balance_sources, balance_lengths):
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        source = row.get("source") if balance_sources else "all"
        if not source:
            raise ValueError("Source-balanced sampling requires explicit source")
        length = (
            str(row.get("length_bucket", "unspecified")) if balance_lengths else "all"
        )
        groups[source][length].append(index)
    return [
        [
            (
                [index for index in indices if not any(rows[index]["targets"])],
                [
                    pool
                    for label in range(label_count)
                    if (
                        pool := [
                            index for index in indices if rows[index]["targets"][label]
                        ]
                    )
                ],
            )
            for indices in lengths.values()
        ]
        for lengths in groups.values()
    ]


def source_sampling_weights(rows, specification):
    """Validate explicit source probabilities in the same order as grouped pools."""
    if specification is None:
        return None
    weights = json.loads(specification)
    sources = list(dict.fromkeys(row.get("source") for row in rows))
    if (
        not all(sources)
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


def sample_grouped(pools, rng, source_weights=None):
    source = (
        rng.choice(pools)
        if source_weights is None
        else rng.choices(pools, weights=source_weights, k=1)[0]
    )
    negative, positive = rng.choice(source)
    if negative and (not positive or rng.random() < SAFE_SAMPLE_FRACTION):
        return rng.choice(negative)
    return rng.choice(rng.choice(positive))


def development_selection_score(metrics, selection, minimum_support=1):
    """Exclude insufficiently supported label APs from checkpoint selection only."""
    if minimum_support < 1:
        raise ValueError("Selection support must be positive")
    groups = (
        list(metrics["breakdowns"]["source"].values())
        if selection == "source-macro-ap"
        else [metrics]
    )
    scores = []
    for group in groups:
        if minimum_support == 1:
            score = group["macro_ap"]
        else:
            values = [
                item["ap"]
                for item in group["per_label"].values()
                if item["ap"] is not None
                and min(item["positive_support"], item["negative_support"])
                >= minimum_support
            ]
            score = sum(values) / len(values) if values else None
        if score is not None:
            scores.append(score)
    if not scores:
        raise ValueError("No development label has sufficient selection support")
    return sum(scores) / len(scores)


def main():
    import torch

    parser = argparse.ArgumentParser()
    for key in [
        "base",
        "base-revision",
        "contract",
        "output",
    ]:
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--fresh-head", action="store_true")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument(
        "--training-order",
        help="Complete content-bound TRAIN ID order; mutually exclusive with sampling flags",
    )
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument(
        "--source-weights",
        help="JSON object of positive source weights; overrides equal-source sampling",
    )
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--length-balanced-sampling", action="store_true")
    parser.add_argument(
        "--selection",
        choices=[
            "macro-ap",
            "source-macro-ap",
            "fp-budget-macro-f1",
            "joint-fp-budget-macro-f1",
        ],
        default="macro-ap",
    )
    parser.add_argument("--selection-false-positive-budget", type=float, default=0.05)
    parser.add_argument(
        "--selection-safe-groups",
        help="Joint selector only: JSON mapping group names to fully-safe DEV IDs; all-safe is always constrained",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--microbatch-token-budget", type=int)
    parser.add_argument("--supervision-diagnostics", action="store_true")
    parser.add_argument(
        "--loss-normalization", choices=["observed", "taxonomy"], default="observed"
    )
    parser.add_argument("--selection-minimum-support", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument(
        "--evaluation-dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    validate_method(args.method, args.adapter, args.fresh_head)
    if args.training_order and (
        args.source_balanced_sampling
        or args.length_balanced_sampling
        or args.source_weights is not None
    ):
        raise ValueError("A fixed training order cannot also configure sampling")
    if args.selection_safe_groups and args.selection != "joint-fp-budget-macro-f1":
        raise ValueError("Explicit safe groups require joint-fp-budget-macro-f1")
    if (
        args.selection == "joint-fp-budget-macro-f1"
        and args.selection_minimum_support != 1
    ):
        raise ValueError(
            "Joint selection preserves all-taxonomy F1 with minimum support 1"
        )
    if (
        min(
            args.steps,
            args.batch_size,
            args.accumulate,
            args.max_length,
            args.eval_every,
            args.selection_minimum_support,
        )
        <= 0
        or args.learning_rate <= 0
        or not math.isfinite(args.selection_false_positive_budget)
        or not 0 <= args.selection_false_positive_budget < 1
        or (
            args.microbatch_token_budget is not None
            and args.microbatch_token_budget <= 0
        )
    ):
        raise ValueError("Training budgets must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing overwrite")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    initial_artifacts = initial_artifact_receipt(args.base, args.adapter)
    model, tokenizer, _label_to_id, id_to_label = load_trainable_model(
        args.base, args.contract, args.adapter, args.method, args.fresh_head
    )
    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Hazard must have a multi-label config contract")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Budget exceeds backbone capacity")
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    rows, dev = read_rows(args.train, len(labels)), read_rows(args.dev, len(labels))
    assert_disjoint(rows, dev)
    selection_safe_groups = (
        json.loads(Path(args.selection_safe_groups).read_text())
        if args.selection_safe_groups
        else None
    )
    if args.selection == "joint-fp-budget-macro-f1":
        safe_group_indices(dev, selection_safe_groups)
    train, rejected, pools = [], [], [[] for _ in labels]
    safe = []
    for row in rows:
        encoded = tokenizer(row["text"], truncation=False, padding=False)
        if len(encoded["input_ids"]) > args.max_length:
            rejected.append({"id": row["id"], "tokens": len(encoded["input_ids"])})
            continue
        index = len(train)
        train.append((row, encoded))
        for i, positive in enumerate(row["targets"]):
            if positive:
                pools[i].append(index)
        if not any(row["targets"]) and all(row["label_mask"]):
            safe.append(index)
    if not safe or any(not pool for pool in pools):
        raise ValueError("Every class needs positive supervision and safe negatives")
    train_lengths = [len(encoded["input_ids"]) for _, encoded in train]
    if (
        args.microbatch_token_budget is not None
        and max(train_lengths) > args.microbatch_token_budget
    ):
        raise ValueError("An eligible input exceeds the microbatch token budget")
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
    balance_sources = args.source_balanced_sampling or args.source_weights is not None
    source_weights = source_sampling_weights(
        [row for row, _ in train], args.source_weights
    )
    balanced_pools = (
        grouped_pools(
            [row for row, _ in train],
            len(labels),
            balance_sources,
            args.length_balanced_sampling,
        )
        if balance_sources or args.length_balanced_sampling
        else None
    )
    output.mkdir(parents=True)
    metadata = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "method": args.method,
        "fresh_head": args.fresh_head,
        "initial_artifacts": initial_artifacts,
        "initial_adapter_sha256": initial_artifacts.get("adapter_model.safetensors"),
        "train": file_receipts(args.train),
        "dev": file_receipts(args.dev),
        "selection_safe_groups": file_receipts(
            [args.selection_safe_groups] if args.selection_safe_groups else []
        ),
        "rejected": rejected,
        "labels": labels,
        "objective": (
            f"mean per-example masked BCE normalized by {args.loss_normalization}; "
            "no loss for unknown labels"
        ),
        "examples_per_optimizer_step": args.batch_size * args.accumulate,
        "microbatch_token_budget": args.microbatch_token_budget,
        "dropout_trajectory_equivalence": args.microbatch_token_budget is None,
        "sampling": (
            "Explicit complete training order"
            if training_order
            else "30% explicit safe; 70% uniform positive category then row"
        ),
        "training_order": training_order.receipt if training_order else None,
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
        "scores": "unconditional independent sigmoid scores; not calibrated posteriors",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "precision": "FP32 parameters/BCE, BF16 autocast",
        "evaluation_dtype": args.evaluation_dtype,
        "attention": "sdpa",
        "test_used": False,
        "arguments": vars(args),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("vela_hazard.py"),
                Path(__file__).with_name("vela_hazard_diagnostics.py"),
                Path(__file__).with_name("vela_hazard_operating.py"),
                Path(__file__).with_name("vela_hazard_joint.py"),
                Path(__file__).parent.parent / "sequence_repair/model.py",
                Path(__file__).parent.parent / "sequence_repair/optimization.py",
                Path(__file__).parent.parent / "sequence_repair/data.py",
                Path(__file__).parent.parent / "sequence_repair/train.py",
                Path(__file__).parent.parent / "sequence_repair/training_order.py",
            ]
        },
        "requirements": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "scikit-learn",
                "numpy",
                "tokenizers",
            ]
            + (["peft"] if args.method == "lora" else [])
        },
        "torch_hip": torch.version.hip,
        "config_sha256": hashlib.sha256(
            json.dumps(model.config.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to("cuda")
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.learning_rate, args.head_learning_rate),
        weight_decay=0.01,
    )
    best = -math.inf
    sampled_unique, sampled_sources, sampled_languages = set(), Counter(), Counter()
    positive_exposures = [0] * len(labels)
    diagnostics = (
        SupervisionDiagnostics(labels, args.loss_normalization)
        if args.supervision_diagnostics
        else None
    )

    def draw_microbatch_indices():
        return [
            (
                sample_grouped(balanced_pools, rng, source_weights)
                if balanced_pools
                else (
                    rng.choice(safe)
                    if rng.random() < SAFE_SAMPLE_FRACTION
                    else rng.choice(rng.choice(pools))
                )
            )
            for _ in range(args.batch_size)
        ]

    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            total = 0.0
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
                microbatches = (
                    draw_microbatch_indices() for _ in range(args.accumulate)
                )
            else:
                indices = [
                    index
                    for _ in range(args.accumulate)
                    for index in draw_microbatch_indices()
                ]
                microbatches = token_budget_microbatches(
                    indices, train_lengths, args.microbatch_token_budget
                )
            microbatch_count, max_padded_tokens = 0, 0
            for indices in microbatches:
                microbatch_count += 1
                max_padded_tokens = max(
                    max_padded_tokens,
                    max(train_lengths[index] for index in indices) * len(indices),
                )
                selected = [train[index] for index in indices]
                sampled_unique.update(indices)
                for row, _ in selected:
                    sampled_sources[row.get("source", "unspecified")] += 1
                    sampled_languages[row.get("language", "unspecified")] += 1
                    for index, target_value in enumerate(row["targets"]):
                        positive_exposures[index] += target_value
                batch = tokenizer.pad(
                    [encoded for _, encoded in selected],
                    padding=True,
                    return_tensors="pt",
                )
                batch = {key: value.to("cuda") for key, value in batch.items()}
                target = torch.tensor(
                    [row["targets"] for row, _ in selected],
                    dtype=torch.float32,
                    device="cuda",
                )
                mask = torch.tensor(
                    [row["label_mask"] for row, _ in selected],
                    dtype=torch.float32,
                    device="cuda",
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**batch).logits
                    if args.microbatch_token_budget is None:
                        loss = masked_loss(
                            logits,
                            target,
                            mask,
                            args.accumulate,
                            normalization=args.loss_normalization,
                        )
                    else:
                        loss = masked_loss(
                            logits, target, mask, normalization=args.loss_normalization
                        ) * (len(selected) / (args.batch_size * args.accumulate))
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite hazard loss")
                loss.backward()
                if diagnostics is not None:
                    diagnostics.observe(
                        [row for row, _ in selected],
                        logits,
                        target,
                        mask,
                        args.batch_size * args.accumulate,
                    )
                total += float(loss.detach())
            if diagnostics is not None:
                diagnostics.observe_head_gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            warmup = max(1, round(args.steps * 0.06))
            factor = (
                step / warmup
                if step <= warmup
                else 0.5
                * (
                    1
                    + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))
                )
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.step()
            if training_order:
                trace = training_order.record_step(step, microbatches)
                with (output / "actual-training-order.jsonl").open("a") as stream:
                    stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
            log = {
                "step": step,
                "loss": total,
                "grad_norm": float(norm),
                "seconds": time.perf_counter() - started,
                "microbatches": microbatch_count,
                "max_padded_tokens": max_padded_tokens,
            }
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(log) + "\n")
            if step % 10 == 0:
                print(json.dumps(log), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if diagnostics is not None:
                (output / f"supervision-step-{step}.json").write_text(
                    json.dumps(diagnostics.snapshot(), indent=2) + "\n"
                )
            coverage = {
                "step": step,
                "draws": sum(sampled_sources.values()),
                "eligible_rows": len(train),
                "unique_rows": len(sampled_unique),
                "source_draws": dict(sampled_sources),
                "source_unique_rows": dict(
                    Counter(
                        train[index][0].get("source", "unspecified")
                        for index in sampled_unique
                    )
                ),
                "language_draws": dict(sampled_languages),
                "positive_exposures": dict(
                    zip(labels, positive_exposures, strict=True)
                ),
            }
            (output / f"coverage-step-{step}.json").write_text(
                json.dumps(coverage, indent=2) + "\n"
            )
            metrics, probabilities = evaluate(
                model,
                tokenizer,
                dev,
                labels,
                args.max_length,
                dtype=args.evaluation_dtype,
            )
            (output / f"dev-step-{step}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            # Keep operating-point evidence even when the AP selector rejects
            # this checkpoint. These are development predictions, never test.
            (output / f"dev-probabilities-step-{step}.json").write_text(
                json.dumps(
                    [
                        {"id": row["id"], "probabilities": scores}
                        for row, scores in zip(dev, probabilities, strict=True)
                    ]
                )
                + "\n"
            )
            operating_point = None
            if args.selection in {"fp-budget-macro-f1", "joint-fp-budget-macro-f1"}:
                options = {
                    "false_positive_budget": args.selection_false_positive_budget,
                    "minimum_support": args.selection_minimum_support,
                }
                selector = select_operating_point
                if args.selection == "joint-fp-budget-macro-f1":
                    selector = select_joint_operating_point
                    options["safe_groups"] = selection_safe_groups
                operating_point = selector(dev, probabilities, labels, **options)
                score = operating_point["selection_score"]
                (output / f"dev-operating-point-step-{step}.json").write_text(
                    json.dumps(operating_point, indent=2) + "\n"
                )
            else:
                score = development_selection_score(
                    metrics, args.selection, args.selection_minimum_support
                )
            print(
                json.dumps(
                    {
                        "event": "dev",
                        "step": step,
                        "selection_score": score,
                        "macro_ap": metrics["macro_ap"],
                        "macro_f1": metrics["macro_f1"],
                        "safe_any_hazard_rate": metrics["safe_any_hazard_rate"],
                    }
                ),
                flush=True,
            )
            if score > best:
                best = score
                save_training_checkpoint(
                    model,
                    tokenizer,
                    output
                    / ("best-adapter" if args.method == "lora" else "best-model"),
                    args.method,
                    args.base_id,
                    args.base_revision,
                )
                (output / "selection.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "score": score,
                            "selection": args.selection,
                            "selection_minimum_support": args.selection_minimum_support,
                            "operating_point": operating_point,
                            "test_used": False,
                        }
                    )
                    + "\n"
                )
                (output / "best-dev-probabilities.json").write_text(
                    json.dumps(
                        [
                            {"id": row["id"], "probabilities": scores}
                            for row, scores in zip(dev, probabilities, strict=True)
                        ]
                    )
                    + "\n"
                )
    if training_order:
        order_receipt = training_order.finish()
        order_receipt["actual_trace_sha256"] = hashlib.sha256(
            (output / "actual-training-order.jsonl").read_bytes()
        ).hexdigest()
        (output / "training-order-completed.json").write_text(
            json.dumps(order_receipt, indent=2) + "\n"
        )
    save_training_checkpoint(
        model,
        tokenizer,
        output / ("last-adapter" if args.method == "lora" else "last-model"),
        args.method,
        args.base_id,
        args.base_revision,
    )


if __name__ == "__main__":
    main()
