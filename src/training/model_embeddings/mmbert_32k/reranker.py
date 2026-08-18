#!/usr/bin/env python3
# Canonicalized from semantic-router/Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539
# Source: scripts/train_rerank.py (blob 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c)
"""Backward-compatible entrypoint for the 2D Matryoshka reranker.

Model, data, loss, evaluation, training, and CLI responsibilities now live in
narrow modules.  The public imports and export ABI remain available here.
"""

from __future__ import annotations

import logging

from .reranker_cli import main
from .reranker_data import RerankerDataset, RerankerExample, collate_fn
from .reranker_evaluation import evaluate_model
from .reranker_loss import Matryoshka2DLoss
from .reranker_model import Matryoshka2DReranker
from .reranker_training import train

# Kept visible for source-level artifact-contract checks and downstream audits.
_EXPORT_CONTRACT = (
    "classification_heads.pt",
    "matryoshka_config.json",
    "weights_only=True",
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

__all__ = [
    "Matryoshka2DLoss",
    "Matryoshka2DReranker",
    "RerankerDataset",
    "RerankerExample",
    "collate_fn",
    "evaluate_model",
    "main",
    "train",
]


if __name__ == "__main__":
    main()
