"""Command-line contract for mmBERT 32K foundation training."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: a8ef9416fb4ce6e6374e5d92f5fefb4dd27221e0.

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument(
        "--rope_scaling_type", type=str, default="yarn", choices=["yarn", "none"]
    )
    parser.add_argument(
        "--rope_original_max_position_embeddings", type=int, default=8192
    )
    parser.add_argument("--yarn_beta_fast", type=float, default=32.0)
    parser.add_argument("--yarn_beta_slow", type=float, default=1.0)
    parser.add_argument(
        "--yarn_extrapolation_factor",
        type=float,
        default=1.0,
        help="Legacy compatibility flag; official YaRN requires the default 1.0",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="Flash Attention 2 bypasses config-driven ModernBERT YaRN",
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--expected_train_samples", type=int, required=True)
    parser.add_argument("--packing_source_repo_id", type=str, required=True)
    parser.add_argument("--packing_languages", type=str, required=True)
    parser.add_argument("--packing_source_etags", type=str, required=True)
    parser.add_argument("--packing_source_content_lengths", type=str, required=True)
    parser.add_argument("--packing_max_document_bytes", type=int, required=True)
    parser.add_argument("--packing_max_document_tokens", type=int, required=True)
    parser.add_argument(
        "--acknowledge_cc100_license_unknown",
        action="store_true",
        help="Acknowledge the release-blocking CC-100 data-governance review",
    )
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--mlm_probability", type=float, default=0.30)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant_with_warmup",
        choices=["constant_with_warmup", "cosine", "linear"],
    )


def _add_regularization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use_retrieval_masking", action="store_true")
    parser.add_argument("--retrieval_probability", type=float, default=0.10)
    parser.add_argument("--min_distance_for_retrieval", type=int, default=512)
    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--ewc_samples", type=int, default=200)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the historical foundation-training parser."""
    parser = argparse.ArgumentParser(description="mmBERT 32K Training")
    _add_model_arguments(parser)
    _add_data_arguments(parser)
    _add_training_arguments(parser)
    _add_regularization_arguments(parser)
    return parser


def _log_configuration(args) -> None:
    logger.info("%s", "=" * 70)
    logger.info("mmBERT 32K Training (Modern Multilingual BERT - 1800+ Languages)")
    logger.info("%s", "=" * 70)
    logger.info("Model: %s", args.model_name_or_path)
    logger.info("Dataset: %s", args.dataset_path)
    logger.info("Max length: %s", args.model_max_length)
    logger.info("RoPE scaling: %s", args.rope_scaling_type)
    logger.info("%s", "-" * 70)
    logger.info("Long-Range Preservation:")
    logger.info("  Learning rate: %s", args.learning_rate)
    logger.info("  LR scheduler: %s", args.lr_scheduler_type)
    logger.info("  Retrieval masking: %s", args.use_retrieval_masking)
    logger.info("  EWC regularization: %s", args.use_ewc)
    logger.info("%s", "=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and delegate to the foundation trainer."""
    from .foundation_training import train  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    _log_configuration(args)
    train(args)
