"""Argument parser for the historical mmBERT BGE-style trainer."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 41e60e17ca960718dd1b71a23a86992128b9ed61.

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible command-line parser."""
    parser = argparse.ArgumentParser(description="BGE-Style Training for mmBERT-32K")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Immutable Hugging Face revision for the base model",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to BGE-format training data directory",
    )
    parser.add_argument("--use_nli", action="store_true", help="Also use AllNLI data")
    parser.add_argument(
        "--nli_revision",
        type=str,
        default=None,
        help="Immutable revision for sentence-transformers/all-nli",
    )
    parser.add_argument(
        "--stsb_revision",
        type=str,
        default=None,
        help="Immutable revision for sentence-transformers/stsb",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_file", type=int, default=50000)
    parser.add_argument("--max_nli_samples", type=int, default=500000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_adaptive_layer", action="store_true")
    parser.add_argument("--use_matryoshka", action="store_true")
    parser.add_argument("--use_2d_matryoshka", action="store_true")
    parser.add_argument("--matryoshka_dims", type=str, default="768,512,256,128,64")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse embedder trainer arguments."""
    return build_parser().parse_args(argv)
