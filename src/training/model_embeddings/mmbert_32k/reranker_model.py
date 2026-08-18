"""2D Matryoshka cross-encoder model and export contract."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import importlib.util
import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


def _attention_implementation(use_flash_attention: bool) -> str:
    if not use_flash_attention:
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is not None:
        logger.info("Using Flash Attention 2")
        return "flash_attention_2"
    logger.warning("flash-attn not installed, falling back to SDPA")
    return "sdpa"


def _layer_attribute(encoder) -> str | None:
    if hasattr(encoder, "layers"):
        return "layers"
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return "encoder.layer"
    logger.warning("Could not detect layer structure")
    return None


def _default_layer_indices(layer_count: int) -> list[int]:
    return [
        max(1, layer_count // 4),
        layer_count // 2,
        3 * layer_count // 4,
        layer_count,
    ]


def _default_dimension_indices(hidden_size: int) -> list[int]:
    return [hidden_size, 3 * hidden_size // 4, hidden_size // 2, hidden_size // 4]


def _final_normalization(encoder):
    if hasattr(encoder, "final_norm"):
        logger.info("Found final_norm layer (will apply to intermediate outputs)")
        return encoder.final_norm
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "final_layernorm"):
        logger.info("Found final_layernorm layer")
        return encoder.encoder.final_layernorm
    return None


def _classification_head(dimension: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dimension, dimension // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(dimension // 2, 1),
    )


def _build_classification_heads(
    layer_indices: list[int], dimension_indices: list[int]
) -> nn.ModuleDict:
    heads = nn.ModuleDict()
    for layer_index in layer_indices:
        heads[str(layer_index)] = nn.ModuleDict(
            {
                str(dimension): _classification_head(dimension)
                for dimension in dimension_indices
            }
        )
    return heads


def _initialize_classification_head(head: nn.Sequential) -> None:
    for module in head:
        if not isinstance(module, nn.Linear):
            continue
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Matryoshka2DReranker(nn.Module):
    """Cross-encoder with independent heads across layer and dimension exits."""

    def __init__(
        self,
        model_name_or_path: str,
        model_revision: str | None = None,
        layer_indices: list[int] | None = None,
        dim_indices: list[int] | None = None,
        use_flash_attn: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        pooling_strategy: str = "cls",
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name_or_path, revision=model_revision, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=_attention_implementation(use_flash_attn),
        )
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.pooling_strategy = pooling_strategy
        self.torch_dtype = torch_dtype
        self.layer_attr = _layer_attribute(self.encoder)
        selected_layers = (
            _default_layer_indices(self.num_layers)
            if layer_indices is None
            else layer_indices
        )
        self.layer_indices = sorted(selected_layers)
        dimensions = (
            _default_dimension_indices(self.hidden_size)
            if dim_indices is None
            else dim_indices
        )
        self.dim_indices = sorted(
            (dimension for dimension in dimensions if dimension <= self.hidden_size),
            reverse=True,
        )
        logger.info("Layer indices: %s", self.layer_indices)
        logger.info("Dimension indices: %s", self.dim_indices)
        self.final_norm = _final_normalization(self.encoder)
        self.layer_heads = _build_classification_heads(
            self.layer_indices, self.dim_indices
        )
        self._init_heads()
        head_count = len(self.layer_indices) * len(self.dim_indices)
        logger.info("Created %s classification heads", head_count)

    def _init_heads(self) -> None:
        """Initialize every classification head with the imported scheme."""
        for layer_heads in self.layer_heads.values():
            for head in layer_heads.values():
                _initialize_classification_head(head)

    def _get_layers(self):
        """Return the detected encoder layer container."""
        if self.layer_attr == "layers":
            return self.encoder.layers
        if self.layer_attr == "encoder.layer":
            return self.encoder.encoder.layer
        return None

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0]
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pooling_strategy == "last":
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths
            ]
        return hidden_states[:, 0]

    def _score_dimensions(self, pooled, layer_index, dimension_indices):
        scores = {}
        for dimension in dimension_indices:
            truncated = pooled[:, :dimension]
            head = self.layer_heads[str(layer_index)][str(dimension)]
            head_dtype = next(head.parameters()).dtype
            if truncated.dtype != head_dtype:
                truncated = truncated.to(head_dtype)
            key = f"layer_{layer_index}_dim_{dimension}"
            scores[key] = head(truncated).squeeze(-1)
        return scores

    @staticmethod
    def _average_loss(all_scores, labels):
        total_loss = 0.0
        loss_count = 0
        for score in all_scores.values():
            total_loss += functional.binary_cross_entropy_with_logits(
                score, labels.float()
            )
            loss_count += 1
        return total_loss / loss_count if loss_count > 0 else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        layer_idx: int | None = None,
        dim_idx: int | None = None,
        return_all_scores: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Score selected exits, optionally averaging pointwise losses."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        layers = [layer_idx] if layer_idx is not None else self.layer_indices
        dimensions = [dim_idx] if dim_idx is not None else self.dim_indices
        all_scores = {}
        for layer_index in layers:
            if layer_index > len(hidden_states) - 1:
                continue
            hidden = hidden_states[layer_index]
            if self.final_norm is not None and layer_index < self.num_layers:
                hidden = self.final_norm(hidden)
            pooled = self._pool(hidden, attention_mask)
            all_scores.update(self._score_dimensions(pooled, layer_index, dimensions))
        result = {}
        if labels is not None:
            average_loss = self._average_loss(all_scores, labels)
            if average_loss is not None:
                result["loss"] = average_loss
        primary_key = f"layer_{self.layer_indices[-1]}_dim_{self.dim_indices[0]}"
        result["logits"] = all_scores.get(primary_key, list(all_scores.values())[-1])
        if return_all_scores:
            result["all_scores"] = all_scores
        return result

    def compute_score(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        layer_idx: int | None = None,
        dim_idx: int | None = None,
        max_length: int = 512,
        normalize: bool = False,
    ) -> list[float]:
        """Compute relevance scores for query/passage pairs."""
        self.eval()
        layer_idx = self.layer_indices[-1] if layer_idx is None else layer_idx
        dim_idx = self.dim_indices[0] if dim_idx is None else dim_idx
        encoded = tokenizer(
            [[query, passage] for query, passage in pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                layer_idx=layer_idx,
                dim_idx=dim_idx,
            )
        scores = outputs["logits"].cpu().numpy().tolist()
        if normalize:
            scores = [1 / (1 + np.exp(-score)) for score in scores]
        return scores

    def save_pretrained(self, save_path: str) -> None:
        """Save encoder weights, custom heads, and their ABI configuration."""
        os.makedirs(save_path, exist_ok=True)
        self.encoder.save_pretrained(save_path)
        torch.save(
            self.layer_heads.state_dict(),
            os.path.join(save_path, "classification_heads.pt"),
        )
        config = {
            "layer_indices": self.layer_indices,
            "dim_indices": self.dim_indices,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "pooling_strategy": self.pooling_strategy,
            "has_final_norm": self.final_norm is not None,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
        }
        with open(os.path.join(save_path, "matryoshka_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)
        logger.info("Model saved to %s", save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Restore an exported encoder and its custom classification heads."""
        config_path = os.path.join(model_path, "matryoshka_config.json")
        if os.path.exists(config_path):
            with open(config_path) as handle:
                config = json.load(handle)
            kwargs.update(
                {
                    "layer_indices": config["layer_indices"],
                    "dim_indices": config["dim_indices"],
                    "pooling_strategy": config.get("pooling_strategy", "cls"),
                }
            )
        model = cls(model_path, **kwargs)
        heads_path = os.path.join(model_path, "classification_heads.pt")
        if os.path.exists(heads_path):
            state = torch.load(heads_path, map_location="cpu", weights_only=True)
            model.layer_heads.load_state_dict(state)
            logger.info("Loaded classification heads")
        return model
