"""CLI parser and dispatch for mmBERT foundation-data preparation."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 7feef3045b9a733d14228436b6c78993eb402a3e.

from __future__ import annotations

import argparse
import logging

from .foundation_data_local import (
    concatenate_to_long_context,
    tokenize_local_files,
    verify_dataset,
)
from .foundation_data_remote import download_from_huggingface

logger = logging.getLogger(__name__)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "local", "verify"],
        default="huggingface",
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--languages", type=str, default=None)
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--detect_language", action="store_true")


def _add_transform_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concatenate_to_length", action="store_true")
    parser.add_argument("--force_retokenize", action="store_true")
    parser.add_argument("--source_tokenizer", type=str, default=None)
    parser.add_argument("--source_tokenizer_revision", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target_sequence_count", type=int, default=None)
    parser.add_argument("--source_etags", type=str, default=None)
    parser.add_argument("--source_content_lengths", type=str, default=None)
    parser.add_argument("--source_prefix_contract_dir", type=str, default=None)
    parser.add_argument("--max_document_bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max_document_tokens", type=int, default=1024 * 1024)
    parser.add_argument("--acknowledge_cc100_license_unknown", action="store_true")
    parser.add_argument("--num_proc", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible preparation parser."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for mmBERT 32K training (multilingual)"
    )
    _add_source_arguments(parser)
    _add_transform_arguments(parser)
    return parser


def _split_optional(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",")] if value else None


def _split_optional_ints(value: str | None) -> list[int] | None:
    return [int(item) for item in value.split(",")] if value else None


def _prepare_huggingface(args, parser) -> None:
    if not args.dataset_name:
        parser.error("--dataset_name is required when --source=huggingface")
    if args.target_sequence_count is not None and args.concatenate_to_length:
        parser.error(
            "--target_sequence_count already performs real token packing; "
            "do not combine it with --concatenate_to_length"
        )
    download_from_huggingface(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_proc=args.num_proc,
        languages=_split_optional(args.languages),
        max_length=args.max_length,
        model_name=args.model_name,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        force_retokenize=args.force_retokenize,
        source_tokenizer_name=args.source_tokenizer,
        source_tokenizer_revision=args.source_tokenizer_revision,
        target_sequence_count=args.target_sequence_count,
        source_etags=_split_optional(args.source_etags),
        source_content_lengths=_split_optional_ints(args.source_content_lengths),
        source_prefix_contract_dir=args.source_prefix_contract_dir,
        max_document_bytes=args.max_document_bytes,
        max_document_tokens=args.max_document_tokens,
        acknowledge_cc100_license_unknown=args.acknowledge_cc100_license_unknown,
        pad_to_max_length=not args.concatenate_to_length,
    )


def _prepare_local(args, parser) -> None:
    if not args.input_files:
        parser.error("--input_files is required when --source=local")
    tokenize_local_files(
        input_files=args.input_files,
        output_dir=args.output_dir,
        max_length=args.max_length,
        model_name=args.model_name,
        model_revision=args.model_revision,
        max_samples=args.max_samples,
        num_proc=args.num_proc,
        detect_language=args.detect_language,
    )


def _concatenate_saved_dataset(args) -> None:
    from datasets import load_from_disk  # noqa: PLC0415

    logger.info("\nConcatenating sequences to %s tokens...", args.max_length)
    dataset = load_from_disk(args.output_dir)
    dataset = concatenate_to_long_context(dataset, args.max_length, args.num_proc)
    dataset.save_to_disk(args.output_dir)
    logger.info("Saved concatenated dataset to: %s", args.output_dir)


def main(argv: list[str] | None = None) -> None:
    """Prepare or verify a dataset according to the imported CLI contract."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "verify":
        raise SystemExit(0 if verify_dataset(args.output_dir) else 1)
    if args.source == "huggingface":
        _prepare_huggingface(args, parser)
    elif args.source == "local":
        _prepare_local(args, parser)
    if args.concatenate_to_length:
        _concatenate_saved_dataset(args)
    verify_dataset(args.output_dir)
