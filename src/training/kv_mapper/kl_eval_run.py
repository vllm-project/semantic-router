#!/usr/bin/env python3
"""Teacher-forced KL of the target on mapped caches, for several artifacts on the same samples.

For each sample the target's own next-token distribution over the continuation is
computed once, the source prefills the prefix once, and every artifact's maps
run on that source cache. The report gives, per sample set, the mean KL from the
target's own distribution and the gold-token NLL for every artifact, with paired
bootstrap intervals against the first artifact. Chat samples are cut before an
assistant reply like the stage-2 training data, and plain-text samples are the
first tokens of documents from a second corpus. HellaSwag cannot separate
mappers that already match the cold target on short contexts, and this does.
Needs torch, transformers and datasets.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Direct execution resolves repository imports after adding the repository root.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import read_artifact
from src.training.kv_mapper.distill import LinearMapper, mapped_cache, source_pairs
from src.training.kv_mapper.distill_run import _load, split_conversation
from src.training.kv_mapper.eval import build_report, write_report


def chat_samples(args, tokenizer, count, seed):
    from datasets import load_dataset  # noqa: PLC0415

    data = load_dataset(
        args.chat_dataset,
        split=args.chat_split,
        revision=args.chat_revision,
        streaming=True,
    )
    rng, out = random.Random(seed), []
    for row in data.shuffle(seed=seed, buffer_size=10_000):
        sample = split_conversation(
            tokenizer, row["messages"], rng, args.max_prefix, args.max_continuation
        )
        if sample is not None:
            out.append(sample)
        if len(out) == count:
            return out
    raise ValueError("not enough chat samples")


def text_window(ids, prefix: int, continuation: int):
    """The first prefix + continuation tokens of a document, or None if shorter."""
    if len(ids) < prefix + continuation:
        return None
    return ids[:prefix], ids[prefix : prefix + continuation]


def text_samples(args, tokenizer, count, seed):
    from datasets import load_dataset  # noqa: PLC0415

    data = load_dataset(
        args.text_dataset,
        args.text_config,
        split="train",
        revision=args.text_revision,
        streaming=True,
    )
    out = []
    for row in data.shuffle(seed=seed, buffer_size=10_000):
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        sample = text_window(ids, args.text_prefix, args.text_continuation)
        if sample is not None:
            out.append(sample)
        if len(out) == count:
            return out
    raise ValueError("not enough text samples")


@torch.no_grad()
def score(source, target, mappers, samples, num_kv_heads, head_dim):
    """Per sample: the target's own gold NLL, and per mapper the mean KL and NLL."""
    device = next(target.parameters()).device
    rows = []
    for prefix, continuation in samples:
        ids = torch.tensor([list(prefix) + list(continuation)], device=device)
        n, m = len(prefix), len(continuation)
        teacher = target(input_ids=ids[:, :-1], use_cache=False).logits[0, n - 1 :]
        teacher = teacher.float().log_softmax(-1)
        gold = ids[0, n:, None]
        pairs = source_pairs(source, ids[:, : n - 1], num_kv_heads, head_dim)
        row = {"cold_nll": float(-teacher.gather(1, gold).mean())}
        for name, mapper in mappers.items():
            student = target(
                input_ids=ids[:, n - 1 : n + m - 1],
                past_key_values=mapped_cache(target, mapper(pairs)),
                use_cache=True,
            ).logits[0]
            student = student.float().log_softmax(-1)
            row[name] = {
                "kl": float((teacher.exp() * (teacher - student)).sum(-1).mean()),
                "nll": float(-student.gather(1, gold).mean()),
            }
        rows.append(row)
    return rows


def report(set_name: str, rows: list[dict], names: list[str]) -> dict:
    ids = [f"{set_name}:{i}" for i in range(len(rows))]
    out = {
        "n": len(rows),
        "mean_cold_nll": sum(r["cold_nll"] for r in rows) / len(rows),
        "rows": rows,
    }
    for metric in ("kl", "nll"):
        arms = {
            name: [
                {"id": i, "score": r[name][metric]}
                for i, r in zip(ids, rows, strict=True)
            ]
            for name in names
        }
        out[metric] = build_report(metric, arms, reference=names[0])
    return out


def run(args) -> dict:
    named = [spec.split("=", 1) for spec in args.artifact]
    loaded = {name: read_artifact(Path(path)) for name, path in named}
    compat = loaded[named[0][0]][0].compatibility
    for name, (manifest, _) in loaded.items():
        if manifest.compatibility != compat:
            raise ValueError(f"{name} is fitted for another model pair or layout")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[compat.precision]
    source = _load(compat.source_model, compat.source_revision, args.device, dtype)
    target = _load(compat.target_model, compat.target_revision, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        compat.target_model, revision=compat.target_revision
    )
    mappers = {
        name: LinearMapper(manifest, tensors).to(args.device).eval()
        for name, (manifest, tensors) in loaded.items()
    }
    samples = {}
    if args.chat_count:
        samples["chat"] = chat_samples(args, tokenizer, args.chat_count, args.seed)
    if args.text_count:
        samples["text"] = text_samples(args, tokenizer, args.text_count, args.seed)
    result = {
        "artifacts": {
            name: manifest.mapper_id for name, (manifest, _) in loaded.items()
        },
        "reference": named[0][0],
        "sets": {},
    }
    names = [name for name, _ in named]
    for set_name, items in samples.items():
        rows = score(
            source, target, mappers, items, compat.num_kv_heads, compat.head_dim
        )
        result["sets"][set_name] = report(set_name, rows, names)
        means = result["sets"][set_name]["kl"]["arm_means"]
        print(set_name, {k: round(v, 4) for k, v in means.items()}, flush=True)
    write_report(args.output, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="name=path, repeated; the first is the reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--chat-revision")
    parser.add_argument("--chat-split", default="test_sft")
    parser.add_argument("--chat-count", type=int, default=128)
    parser.add_argument("--text-dataset", default="wikimedia/wikipedia")
    parser.add_argument("--text-config", default="20231101.en")
    parser.add_argument("--text-revision")
    parser.add_argument("--text-count", type=int, default=128)
    parser.add_argument("--max-prefix", type=int, default=2048)
    parser.add_argument("--max-continuation", type=int, default=256)
    parser.add_argument("--text-prefix", type=int, default=1024)
    parser.add_argument("--text-continuation", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.chat_count and not args.chat_revision:
        parser.error("--chat-revision pins the chat dataset")
    if args.text_count and not args.text_revision:
        parser.error("--text-revision pins the text dataset")
    return args


if __name__ == "__main__":
    run(parse_args())
