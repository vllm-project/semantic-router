#!/usr/bin/env python3
"""Backward-compatible entrypoint for mmBERT 32K foundation training.

The source-faithful masking, EWC, config-first model loading, training loop, and
CLI now live in narrow modules.  These re-exports preserve the historical
launcher and import surface.
"""

from __future__ import annotations

import logging

from .foundation_cli import main
from .foundation_collators import (
    ANCHOR_MASK_PROBABILITY,
    RetrievalMaskingCollator,
    StandardMLMCollator,
)
from .foundation_ewc import EWCRegularizer
from .foundation_training import load_dataset_from_path, train

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

__all__ = [
    "ANCHOR_MASK_PROBABILITY",
    "EWCRegularizer",
    "RetrievalMaskingCollator",
    "StandardMLMCollator",
    "load_dataset_from_path",
    "main",
    "train",
]


if __name__ == "__main__":
    main()
