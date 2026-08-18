#!/usr/bin/env python3
# Canonicalized from semantic-router/Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539
# Source: scripts/train_bge_style.py (blob 41e60e17ca960718dd1b71a23a86992128b9ed61)
"""Backward-compatible entrypoint for mmBERT BGE-style embedding training.

The imported algorithm is split into narrow data, evaluation, training, and CLI
modules.  This module deliberately retains the historical public names used by
launch scripts and downstream notebooks.
"""

from __future__ import annotations

import logging

from .embedder_cli import parse_args
from .embedder_data import (
    convert_bge_to_triplets,
    get_batch_size_for_length,
    load_bge_data_directory,
    load_bge_jsonl_file,
)
from .embedder_evaluation import (
    create_evaluator,
    get_model_info,
    load_evaluation_data,
    test_layer_reduction,
)
from .embedder_training import SelfDistillationLoss, train

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

__all__ = [
    "SelfDistillationLoss",
    "convert_bge_to_triplets",
    "create_evaluator",
    "get_batch_size_for_length",
    "get_model_info",
    "load_bge_data_directory",
    "load_bge_jsonl_file",
    "load_evaluation_data",
    "parse_args",
    "test_layer_reduction",
    "train",
]


def main() -> None:
    """Run the historical trainer CLI."""
    train(parse_args())


if __name__ == "__main__":
    main()
