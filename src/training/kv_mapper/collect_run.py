#!/usr/bin/env python3
"""Collect pre-RoPE K/V from source and target models. GPU; not run in CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Direct execution resolves repository imports after adding the repository root.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.collect import (
    ActivationRunMeta,
    calibration_windows,
    parse_layer_subset,
    token_fingerprint,
    validate_activation_chunk,
    write_activation_chunk,
    write_run_metadata,
)
from src.training.kv_mapper.hooks import capture_kv


def _load_lm(model_id: str, revision: str, device: str, dtype: torch.dtype):
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
    corpus = load_dataset(
        args.corpus,
        args.dataset_config,
        split="train",
        streaming=True,
        revision=args.dataset_revision,
    ).shuffle(seed=args.seed, buffer_size=10_000)
    documents = (
        tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        for row in corpus
        if row.get("text")
    )
    return list(
        calibration_windows(
            documents,
            seq_len=args.seq_len,
            window_stride=args.window_stride,
            num_sequences=args.num_sequences,
        )
    )


def _capture_windows(
    model, windows, device, n_kv_heads, head_dim, layers, token_step, out_dir
):
    for index, window in enumerate(windows):
        path = out_dir / f"{index:06d}.npz"
        n_rows = len(range(0, len(window), token_step))
        try:
            if validate_activation_chunk(
                path, len(layers), n_rows, n_kv_heads, head_dim
            ):
                continue
        except ValueError:
            # A Spot interruption may leave the chunk without its checksum.
            path.unlink(missing_ok=True)
            path.with_suffix(".sha256").unlink(missing_ok=True)
        ids = torch.tensor([window], dtype=torch.long, device=device)
        captured = capture_kv(model, ids, n_kv_heads, head_dim, layers)
        keys = [tensor[::token_step].numpy() for tensor in captured.keys]
        values = [tensor[::token_step].numpy() for tensor in captured.values]
        write_activation_chunk(path, keys, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument(
        "--dtype", choices=("float16", "fp16", "bfloat16", "bf16"), default="bf16"
    )
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--source-layer-subset", default="all")
    parser.add_argument("--target-layer-subset", default="all")
    parser.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=0)
    parser.add_argument("--fitting-token-step", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.fitting_token_step <= 0:
        parser.error("--fitting-token-step must be positive")
    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    src_model, tok = _load_lm(
        args.source_model, args.source_revision, args.source_device, dtype
    )
    n_src = len(src_model.model.layers)
    src_layers = parse_layer_subset(args.source_layer_subset, n_src)
    out = args.output_dir
    token_path = out / "tokens.npy"
    if token_path.exists() and not (out / "run.json").exists():
        raise ValueError(
            "saved token windows have no run metadata; choose another output directory"
        )
    if token_path.exists():
        windows = np.load(token_path, allow_pickle=False)
    else:
        windows = np.asarray(_corpus_windows(args, tok), dtype=np.int64)
    if windows.shape != (args.num_sequences, args.seq_len):
        raise ValueError(f"token windows have unexpected shape {windows.shape}")
    token_sha = token_fingerprint(windows)
    tgt_tok = AutoTokenizer.from_pretrained(
        args.target_model, revision=args.target_revision, trust_remote_code=True
    )
    if tok.get_vocab() != tgt_tok.get_vocab():
        raise ValueError(
            "source and target tokenizers differ; paired token windows require the same vocabulary"
        )
    n_tgt = AutoConfig.from_pretrained(
        args.target_model, revision=args.target_revision, trust_remote_code=True
    ).num_hidden_layers
    tgt_layers = parse_layer_subset(args.target_layer_subset, n_tgt)
    meta = ActivationRunMeta(
        corpus=args.corpus,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        source_model=args.source_model,
        source_revision=args.source_revision,
        target_model=args.target_model,
        target_revision=args.target_revision,
        seed=args.seed,
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        fitting_token_step=args.fitting_token_step,
        token_sha256=token_sha,
        num_sequences=len(windows),
        source_layers=src_layers,
        target_layers=tgt_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        precision=args.dtype,
    )
    write_run_metadata(out, meta)
    if not token_path.exists():
        with token_path.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, windows)
        token_path.with_suffix(".tmp").replace(token_path)
    _capture_windows(
        src_model,
        windows,
        args.source_device,
        args.num_kv_heads,
        args.head_dim,
        src_layers,
        args.fitting_token_step,
        out / "source",
    )
    del src_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_model, _ = _load_lm(
        args.target_model, args.target_revision, args.target_device, dtype
    )
    _capture_windows(
        tgt_model,
        windows,
        args.target_device,
        args.num_kv_heads,
        args.head_dim,
        tgt_layers,
        args.fitting_token_step,
        out / "target",
    )


if __name__ == "__main__":
    main()
