"""Apply explicit pooling metadata when using the SentenceTransformer reader.

Metadata is intentionally not guessed for legacy checkpoints. Call this after
loading a SentenceTransformer; a saved Pooling module alone does not encode its
accumulation precision.
"""

from __future__ import annotations

import torch
from sentence_transformers.models import Pooling

from .representation_contract import read_representation_contract


def _fp32_pooling_inputs(_module, inputs):
    features = inputs[0]
    mask = features["attention_mask"]
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("Cannot pool an all-padding input")
    if "token_weights_sum" in features:
        raise ValueError("Explicit attention-mask mean cannot use token weights")
    features = dict(features)
    features["token_embeddings"] = features["token_embeddings"].float()
    return (features, *inputs[1:])


def configure_sentence_transformer_representation(model):
    """Validate and install FP32 mean pooling for an explicitly declared model."""
    contract = read_representation_contract(model[0].auto_model.config, "embedding")
    if contract is None:
        return None
    if model[0].auto_model.config.model_type != "modernbert":
        raise ValueError("Explicit representation requires native ModernBERT")
    pools = [module for module in model if isinstance(module, Pooling)]
    if len(pools) != 1:
        raise ValueError("Explicit representation requires exactly one mean pool")
    pooling = pools[0]
    modes = pooling.get_config_dict()
    if not modes.get("pooling_mode_mean_tokens") or any(
        value
        for key, value in modes.items()
        if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens"
    ):
        raise ValueError("Explicit representation supports attention-mask mean only")
    if not modes.get("include_prompt", True):
        raise ValueError(
            "Explicit attention-mask mean must include attended prompt tokens"
        )
    if not getattr(pooling, "_vela_fp32_pooling_installed", False):
        pooling.register_forward_pre_hook(_fp32_pooling_inputs)
        pooling._vela_fp32_pooling_installed = True
    return contract
