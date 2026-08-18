"""Argument parser for the historical 2D Matryoshka reranker trainer."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_2d_matryoshka", action="store_true")
    parser.add_argument("--layer_indices", type=str, default=None)
    parser.add_argument("--dim_indices", type=str, default=None)
    parser.add_argument(
        "--pooling_strategy",
        type=str,
        default="cls",
        choices=["cls", "mean", "last"],
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--negatives_per_query", type=int, default=7)
    parser.add_argument("--use_quora", action="store_true")
    parser.add_argument("--use_fever", action="store_true")
    parser.add_argument("--max_quora_samples", type=int, default=100000)
    parser.add_argument("--max_fever_samples", type=int, default=100000)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the source-compatible reranker CLI parser."""
    parser = argparse.ArgumentParser(description="Train 2D Matryoshka Reranker")
    _add_model_arguments(parser)
    _add_data_arguments(parser)
    _add_training_arguments(parser)
    return parser


def _log_configuration(args) -> None:
    logger.info("%s", "=" * 70)
    logger.info("2D Matryoshka Reranker Training")
    logger.info("(BGE-reranker-v2-m3 style training)")
    logger.info("%s", "=" * 70)
    logger.info("Model: %s", args.model_name)
    logger.info("Train data: %s", args.train_data or "None")
    logger.info("  + Quora: %s (max %s)", args.use_quora, args.max_quora_samples)
    logger.info("  + FEVER: %s (max %s)", args.use_fever, args.max_fever_samples)
    logger.info("2D Matryoshka: %s", args.use_2d_matryoshka)
    logger.info("Flash Attention: %s", args.use_flash_attn)
    logger.info("BF16: %s", args.bf16)
    logger.info("%s", "=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the historical reranker training flow."""
    from .reranker_training import train  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    _log_configuration(args)
    train(args)
