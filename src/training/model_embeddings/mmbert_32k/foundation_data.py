#!/usr/bin/env python3
# Canonicalized from semantic-router/Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539
# Source: scripts/prepare_dataset.py (blob 7feef3045b9a733d14228436b6c78993eb402a3e)
"""Backward-compatible entrypoint for mmBERT 32K dataset preparation.

Exact streaming materialization, legacy Hugging Face transforms, local-corpus
handling, and CLI dispatch are separated while retaining the historical public
function names.
"""

from __future__ import annotations

import logging

from .foundation_data_cli import main
from .foundation_data_local import (
    concatenate_to_long_context,
    tokenize_local_files,
    verify_dataset,
)
from .foundation_data_remote import download_from_huggingface
from .foundation_data_transforms import retokenize_dataset, tokenize_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

__all__ = [
    "concatenate_to_long_context",
    "download_from_huggingface",
    "main",
    "retokenize_dataset",
    "tokenize_dataset",
    "tokenize_local_files",
    "verify_dataset",
]


if __name__ == "__main__":
    main()
