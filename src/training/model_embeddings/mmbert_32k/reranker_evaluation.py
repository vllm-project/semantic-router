"""Qualitative evaluation retained from the imported reranker trainer."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TEST_PAIRS = [
    (
        "What is machine learning?",
        "Machine learning is a subset of AI that enables systems to learn from data.",
    ),
    ("What is machine learning?", "The weather is sunny today."),
    (
        "How to cook pasta?",
        "Boil water, add pasta, cook for 10 minutes, drain and serve.",
    ),
    ("How to cook pasta?", "Python is a programming language."),
]


def _log_pair_scores(pairs, scores, *, detailed: bool) -> None:
    for (query, passage), score in zip(pairs, scores, strict=True):
        if detailed:
            logger.info("  Q: %s...", query[:50])
            logger.info("  P: %s...", passage[:50])
            logger.info("  Score: %.4f", score)
            logger.info("")
        else:
            logger.info(
                "  Score: %.4f - %s... / %s...", score, query[:30], passage[:30]
            )


def evaluate_model(model, tokenizer, device) -> None:
    """Run the historical full and reduced-exit qualitative samples."""
    del device  # Retained in the public API used by historical callers.
    model.eval()
    logger.info("\n%s", "=" * 60)
    logger.info("Model Evaluation")
    logger.info("%s", "=" * 60)
    scores = model.compute_score(_TEST_PAIRS, tokenizer, normalize=True)
    logger.info("\nFull model (all layers, full dimension):")
    _log_pair_scores(_TEST_PAIRS, scores, detailed=True)
    if len(model.layer_indices) > 1:
        middle_layer = model.layer_indices[len(model.layer_indices) // 2]
        scores = model.compute_score(
            _TEST_PAIRS, tokenizer, layer_idx=middle_layer, normalize=True
        )
        logger.info("\nReduced model (layer %s):", middle_layer)
        _log_pair_scores(_TEST_PAIRS, scores, detailed=False)
    if len(model.dim_indices) > 1:
        middle_dimension = model.dim_indices[len(model.dim_indices) // 2]
        scores = model.compute_score(
            _TEST_PAIRS, tokenizer, dim_idx=middle_dimension, normalize=True
        )
        logger.info("\nReduced model (dim %s):", middle_dimension)
        _log_pair_scores(_TEST_PAIRS, scores, detailed=False)
