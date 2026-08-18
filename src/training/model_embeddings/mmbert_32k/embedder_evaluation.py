"""Evaluation and layer-reduction helpers for mmBERT embedding training."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 41e60e17ca960718dd1b71a23a86992128b9ed61.

from __future__ import annotations

import logging

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)

logger = logging.getLogger(__name__)


def load_evaluation_data(revision: str | None = None):
    """Load the historical STS validation and test splits."""
    logger.info("Loading STS benchmark for evaluation...")
    validation = load_dataset(
        "sentence-transformers/stsb", split="validation", revision=revision
    )
    test = load_dataset("sentence-transformers/stsb", split="test", revision=revision)
    return validation, test


def create_evaluator(dataset, name: str):
    """Create the cosine STS evaluator used by the upstream trainer."""
    return EmbeddingSimilarityEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        scores=dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name=name,
    )


def get_model_info(model: SentenceTransformer) -> dict:
    """Return the architecture details printed by the historical trainer."""
    auto_model = model[0].auto_model
    if hasattr(auto_model, "layers"):
        num_layers = len(auto_model.layers)
        layer_attr = "layers"
    elif hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
        num_layers = len(auto_model.encoder.layer)
        layer_attr = "encoder.layer"
    else:
        num_layers = -1
        layer_attr = "unknown"
    return {
        "num_layers": num_layers,
        "layer_attr": layer_attr,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "max_seq_length": model.max_seq_length,
    }


def _replace_layers(auto_model, layer_attr: str, layers: list) -> None:
    layer_list = torch.nn.ModuleList(layers)
    if layer_attr == "layers":
        auto_model.layers = layer_list
    else:
        auto_model.encoder.layer = layer_list


def test_layer_reduction(model: SentenceTransformer):
    """Run the upstream qualitative layer-reduction smoke and restore the model."""
    sentences = [
        "The weather is beautiful today.",
        "It's a lovely sunny day outside.",
        "I love programming in Python.",
        "Machine learning is fascinating.",
    ]
    info = get_model_info(model)
    if info["layer_attr"] == "unknown":
        return
    auto_model = model[0].auto_model
    if info["layer_attr"] == "layers":
        original_layers = list(auto_model.layers)
    else:
        original_layers = list(auto_model.encoder.layer)
    num_layers = len(original_layers)
    logger.info("\n%s", "=" * 60)
    logger.info("Testing Adaptive Layer Performance")
    logger.info("%s", "=" * 60)
    for layer_count in [num_layers, num_layers // 2, 6, 3]:
        if layer_count > num_layers or layer_count < 1:
            continue
        _replace_layers(auto_model, info["layer_attr"], original_layers[:layer_count])
        embeddings = model.encode(sentences, normalize_embeddings=True)
        logger.info("\nLayers: %s/%s", layer_count, num_layers)
        logger.info(
            "  'weather' vs 'sunny': %.4f", float(embeddings[0] @ embeddings[1])
        )
        logger.info(
            "  'weather' vs 'Python': %.4f", float(embeddings[0] @ embeddings[2])
        )
    _replace_layers(auto_model, info["layer_attr"], original_layers)
