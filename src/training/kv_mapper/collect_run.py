#!/usr/bin/env python3
"""Collect pre-RoPE K/V from source and target models. GPU; not run in CI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.collect import (
    ActivationRunMeta,
    calibration_windows,
    parse_layer_subset,
    write_run_metadata,
)
from src.training.kv_mapper.hooks import capture_kv


def _load_lm(model_id: str, revision: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok


def _corpus_windows(args: argparse.Namespace, tokenizer):
    from datasets import load_dataset

    corpus = load_dataset(
        args.corpus, args.dataset_config, split="train", streaming=True
    ).shuffle(seed=args.seed, buffer_size=10_000)
    documents = (
        tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        for row in corpus
        if row.get("text")
    )
    return list(calibration_windows(
        documents, seq_len=args.seq_len, stride=args.stride,
        num_sequences=args.num_sequences,
    ))


def _capture_windows(model, windows, device, n_kv_heads, head_dim, layers):
    keys = [[] for _ in layers]
    values = [[] for _ in layers]
    for window in windows:
        ids = torch.tensor([window], dtype=torch.long, device=device)
        captured = capture_kv(model, ids, n_kv_heads, head_dim, layers)
        for index, (key, value) in enumerate(zip(captured.keys, captured.values)):
            keys[index].append(key)
            values[index].append(value)
    return [torch.stack(items) for items in keys], [torch.stack(items) for items in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--source-layer-subset", default="all")
    parser.add_argument("--target-layer-subset", default="all")
    parser.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(
        args.dtype, torch.float32
    )
    src_model, tok = _load_lm(
        args.source_model, args.source_revision, args.source_device, dtype
    )
    n_src = len(src_model.model.layers)
    src_layers = parse_layer_subset(args.source_layer_subset, n_src)
    windows = _corpus_windows(args, tok)
    src_keys, src_values = _capture_windows(
        src_model, windows, args.source_device, args.num_kv_heads,
        args.head_dim, src_layers,
    )
    del src_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_model, tgt_tok = _load_lm(
        args.target_model, args.target_revision, args.target_device, dtype
    )
    if tok.get_vocab() != tgt_tok.get_vocab():
        raise ValueError("source and target tokenizers differ; paired token windows require the same vocabulary")
    n_tgt = len(tgt_model.model.layers)
    tgt_layers = parse_layer_subset(args.target_layer_subset, n_tgt)
    tgt_keys, tgt_values = _capture_windows(
        tgt_model, windows, args.target_device, args.num_kv_heads,
        args.head_dim, tgt_layers,
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = ActivationRunMeta(
        corpus=args.corpus,
        dataset_config=args.dataset_config,
        source_model=args.source_model,
        source_revision=args.source_revision,
        target_model=args.target_model,
        target_revision=args.target_revision,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        num_sequences=len(windows),
        source_layers=src_layers,
        target_layers=tgt_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        precision=args.dtype,
    )
    write_run_metadata(out, meta)
    torch.save(
        {
            "source_keys": src_keys,
            "source_values": src_values,
            "target_keys": tgt_keys,
            "target_values": tgt_values,
        },
        out / "activations.pt",
    )
    (out / "shapes.json").write_text(
        json.dumps(
            {
                "source_keys": [list(t.shape) for t in src_keys],
                "target_keys": [list(t.shape) for t in tgt_keys],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
