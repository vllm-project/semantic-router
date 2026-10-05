#!/usr/bin/env python3
"""Self-distill a fitted mapper (stage 2) and write it as the next bundle.

Reads a stage-1 artifact, streams chat conversations from a pinned dataset,
splits each at an assistant turn into (prefix, continuation), trains the maps
with `distill.train`, and writes the same artifact layout with
`bundle_version` bumped and the recipe recorded under `calibration.stage2`.
Needs torch, transformers and datasets; it is not a contract test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Direct execution resolves repository imports after adding the repository root.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import read_artifact, write_artifact
from src.training.kv_mapper.distill import LinearMapper, mean_kl, train
from src.training.kv_mapper.mapper_id import make_mapper_id


def _load(name: str, revision: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    return model.to(device).eval()


def split_conversation(tokenizer, messages, rng, max_prefix, max_continuation):
    """Cut before one assistant reply: the prefix ends with the generation prompt."""
    turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not turns:
        return None
    turn = rng.choice(turns)
    prefix_text = tokenizer.apply_chat_template(
        messages[:turn], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages[: turn + 1], tokenize=False)
    if not full_text.startswith(prefix_text):
        return None
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    continuation = tokenizer.encode(
        full_text[len(prefix_text) :], add_special_tokens=False
    )[:max_continuation]
    if len(prefix) < 2 or len(prefix) > max_prefix or not continuation:  # noqa: PLR2004
        return None
    return prefix, continuation


def conversations(args, epoch: int):
    from datasets import load_dataset  # noqa: PLC0415

    data = load_dataset(
        args.dataset, split=args.split, revision=args.dataset_revision, streaming=True
    )
    return data.shuffle(seed=args.seed + epoch, buffer_size=args.shuffle_buffer)


def conversation_key(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def validation_and_training(tokenizer, args):
    """Hold out the first `val_count` usable conversations, then cycle the rest."""
    rng = random.Random(args.seed)
    validation, held_out = [], set()
    for row in conversations(args, 0):
        sample = split_conversation(
            tokenizer,
            row[args.messages_field],
            rng,
            args.max_prefix,
            args.max_continuation,
        )
        if sample is not None:
            validation.append(sample)
            held_out.add(conversation_key(row[args.messages_field]))
            if len(validation) == args.val_count:
                break

    def training():
        for epoch in itertools.count():
            for row in conversations(args, epoch):
                messages = row[args.messages_field]
                if conversation_key(messages) in held_out:
                    continue
                sample = split_conversation(
                    tokenizer, messages, rng, args.max_prefix, args.max_continuation
                )
                if sample is not None:
                    yield sample

    return validation, training()


def run(args) -> dict:
    manifest, tensors = read_artifact(args.artifact)
    compat = manifest.compatibility
    bundle = args.bundle_version or int(manifest.mapper_id.rsplit("-b", 1)[1]) + 1
    mapper_id = make_mapper_id(
        pair_slug=args.pair_slug,
        variant=compat.variant,
        precision=compat.precision,
        source_revision=compat.source_revision,
        target_revision=compat.target_revision,
        source_tp=compat.source_tp,
        target_tp=compat.target_tp,
        n_kv_heads=compat.num_kv_heads,
        bundle_version=bundle,
    )
    out = args.output_dir / mapper_id
    if out.exists():
        raise FileExistsError(
            f"artifact already exists: {out}; choose another --bundle-version"
        )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[compat.precision]
    source = _load(
        compat.source_model, compat.source_revision, args.source_device, dtype
    )
    target = _load(
        compat.target_model, compat.target_revision, args.target_device, dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        compat.target_model, revision=compat.target_revision
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        compat.source_model, revision=compat.source_revision
    )
    if tokenizer.get_vocab() != source_tokenizer.get_vocab():
        raise ValueError("source and target tokenizers differ")
    mapper = LinearMapper(manifest, tensors, rank=args.rank).to(target.device)
    validation, stream = validation_and_training(tokenizer, args)
    kl_before = mean_kl(source, target, mapper, validation)
    print(f"validation kl before {kl_before:.5f}", flush=True)
    history = train(
        source,
        target,
        mapper,
        stream,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        clip=args.clip,
        relative_lr=args.relative_lr,
        log=lambda line: print(line, flush=True),
    )
    kl_after = mean_kl(source, target, mapper, validation)
    print(f"validation kl after {kl_after:.5f}", flush=True)

    stage2 = {
        "method": "self_distillation",
        "reference": "arXiv:2609.32610",
        "parent_mapper_id": manifest.mapper_id,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "warmup_fraction": args.warmup_fraction,
        "clip": args.clip,
        "relative_lr": args.relative_lr,
        "parameterization": f"nora_rank_{args.rank}" if args.rank else "full",
        "max_prefix": args.max_prefix,
        "max_continuation": args.max_continuation,
        "seed": args.seed,
        "val_count": args.val_count,
        "val_kl_before": kl_before,
        "val_kl_after": kl_after,
    }
    calibration = {**manifest.calibration, "stage2": stage2}
    write_artifact(
        out,
        replace(manifest, mapper_id=mapper_id, calibration=calibration),
        mapper.export(),
    )
    (args.output_dir / f"{mapper_id}.history.json").write_text(json.dumps(history))
    print(f"wrote {out}", flush=True)
    return stage2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, required=True, help="stage-1 mapper dir"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-slug", required=True, help="e.g. qwen3-14b-32b")
    parser.add_argument("--dataset", required=True, help="chat dataset with messages")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--relative-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale each tensor's rate by the RMS of its map",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="train a NoRA low-rank correction; 0 trains W",
    )
    parser.add_argument("--max-prefix", type=int, default=2048)
    parser.add_argument("--max-continuation", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bundle-version",
        type=int,
        default=0,
        help="bundle of the output; 0 takes the next one after the input",
    )
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
