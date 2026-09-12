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
from collections import defaultdict
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, file_receipts
from ..sequence_repair.model import load_model, save_adapter
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


def sample_grouped(pools, rng):
    negative, positive = rng.choice(rng.choice(pools))
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
    parser.add_argument("--length-balanced-sampling", action="store_true")
    parser.add_argument(
        "--selection", choices=["macro-ap", "source-macro-ap"], default="macro-ap"
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
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
    balanced_pools = (
        grouped_pools(
            [row for row, _ in train],
            len(labels),
            args.source_balanced_sampling,
            args.length_balanced_sampling,
        )
        if args.source_balanced_sampling or args.length_balanced_sampling
        else None
    )
    output.mkdir(parents=True)
    metadata = {
        "base_model": "llm-semantic-router/mmbert-32k-yarn",
        "base_revision": args.base_revision,
        "initial_adapter_sha256": hashlib.sha256(
            (Path(args.adapter) / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "train": file_receipts(args.train),
        "dev": file_receipts(args.dev),
        "rejected": rejected,
        "labels": labels,
        "objective": "mean per-example masked BCE; no loss for unknown labels",
        "sampling": "30% explicit safe; 70% uniform positive category then row",
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
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            total = 0.0
            for _ in range(args.accumulate):
                indices = [
                    (
                        sample_grouped(balanced_pools, rng)
                        if balanced_pools
                        else (
                            rng.choice(safe)
                            if rng.random() < SAFE_SAMPLE_FRACTION
                            else rng.choice(rng.choice(pools))
                        )
                    )
                    for _ in range(args.batch_size)
                ]
                selected = [train[index] for index in indices]
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
                    loss = masked_loss(
                        model(**batch).logits, target, mask, args.accumulate
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
            }
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(log) + "\n")
            if step % 10 == 0:
                print(json.dumps(log), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
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
                    "llm-semantic-router/mmbert-32k-yarn",
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
        "llm-semantic-router/mmbert-32k-yarn",
        args.base_revision,
    )


if __name__ == "__main__":
    main()
