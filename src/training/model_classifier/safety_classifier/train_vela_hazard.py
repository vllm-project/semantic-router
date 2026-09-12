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
from ..sequence_repair.model import load_model, save_adapter
from ..sequence_repair.train import token_budget_microbatches
from .vela_hazard import evaluate, masked_loss, read_rows

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


def main():
    import torch

    parser = argparse.ArgumentParser()
    for key in [
        "base",
        "base-revision",
        "adapter",
        "contract",
        "output",
    ]:
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument(
        "--source-weights",
        help="JSON object of positive source weights; overrides equal-source sampling",
    )
    parser.add_argument("--base-id", default="llm-semantic-router/mmbert-32k-yarn")
    parser.add_argument("--length-balanced-sampling", action="store_true")
    parser.add_argument(
        "--selection", choices=["macro-ap", "source-macro-ap"], default="macro-ap"
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--microbatch-token-budget", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--eval-every", type=int, default=150)
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
    model, tokenizer, _label_to_id, id_to_label = load_model(
        args.base, args.contract, args.adapter, trainable=True
    )
    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Hazard must have a multi-label config contract")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Budget exceeds backbone capacity")
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    rows, dev = read_rows(args.train, len(labels)), read_rows(args.dev, len(labels))
    assert_disjoint(rows, dev)
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
        "initial_adapter_sha256": hashlib.sha256(
            (Path(args.adapter) / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "train": file_receipts(args.train),
        "dev": file_receipts(args.dev),
        "rejected": rejected,
        "labels": labels,
        "objective": "mean per-example masked BCE; no loss for unknown labels",
        "examples_per_optimizer_step": args.batch_size * args.accumulate,
        "microbatch_token_budget": args.microbatch_token_budget,
        "dropout_trajectory_equivalence": args.microbatch_token_budget is None,
        "sampling": "30% explicit safe; 70% uniform positive category then row",
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
        "precision": "FP32 parameters/BCE, BF16 autocast",
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
                Path(__file__).parent.parent / "sequence_repair/model.py",
                Path(__file__).parent.parent / "sequence_repair/data.py",
                Path(__file__).parent.parent / "sequence_repair/train.py",
            ]
        },
        "requirements": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "peft",
                "scikit-learn",
                "numpy",
                "tokenizers",
            ]
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
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    best = -1.0
    sampled_unique, sampled_sources, sampled_languages = set(), Counter(), Counter()
    positive_exposures = [0] * len(labels)

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
            if args.microbatch_token_budget is None:
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
                        loss = masked_loss(logits, target, mask, args.accumulate)
                    else:
                        loss = masked_loss(logits, target, mask) * (
                            len(selected) / (args.batch_size * args.accumulate)
                        )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite hazard loss")
                loss.backward()
                total += float(loss.detach())
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
                group["lr"] = args.learning_rate * factor
            optimizer.step()
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
                model, tokenizer, dev, labels, args.max_length
            )
            (output / f"dev-step-{step}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            score = metrics["macro_ap"]
            if args.selection == "source-macro-ap":
                source_scores = [
                    item["macro_ap"]
                    for item in metrics["breakdowns"]["source"].values()
                    if item["macro_ap"] is not None
                ]
                if not source_scores:
                    raise ValueError("No development sources support AP selection")
                score = sum(source_scores) / len(source_scores)
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
                save_adapter(
                    model,
                    tokenizer,
                    output / "best-adapter",
                    args.base_id,
                    args.base_revision,
                )
                (output / "selection.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "score": score,
                            "selection": args.selection,
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
    save_adapter(
        model,
        tokenizer,
        output / "last-adapter",
        args.base_id,
        args.base_revision,
    )


if __name__ == "__main__":
    main()
