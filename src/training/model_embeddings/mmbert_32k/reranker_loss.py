"""Weighted two-dimensional Matryoshka loss."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import torch
from torch.nn import functional


def _normalized_inverse_weights(size: int, *, reverse: bool) -> list[float]:
    if reverse:
        weights = [1.0 / (size - index) for index in range(size)]
    else:
        weights = [1.0 / (index + 1) for index in range(size)]
    total = sum(weights)
    return [weight / total for weight in weights]


class Matryoshka2DLoss:
    """Weight BCE losses across every configured layer/dimension exit."""

    def __init__(
        self,
        layer_indices: list[int],
        dim_indices: list[int],
        layer_weights: list[float] | None = None,
        dim_weights: list[float] | None = None,
    ):
        self.layer_indices = layer_indices
        self.dim_indices = dim_indices
        self.layer_weights = (
            _normalized_inverse_weights(len(layer_indices), reverse=True)
            if layer_weights is None
            else layer_weights
        )
        self.dim_weights = (
            _normalized_inverse_weights(len(dim_indices), reverse=False)
            if dim_weights is None
            else dim_weights
        )

    def __call__(
        self, all_scores: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted average of all available exit losses."""
        total_loss = 0.0
        total_weight = 0.0
        for layer_position, layer_index in enumerate(self.layer_indices):
            for dimension_position, dimension in enumerate(self.dim_indices):
                key = f"layer_{layer_index}_dim_{dimension}"
                if key not in all_scores:
                    continue
                loss = functional.binary_cross_entropy_with_logits(
                    all_scores[key], labels.float()
                )
                weight = (
                    self.layer_weights[layer_position]
                    * self.dim_weights[dimension_position]
                )
                total_loss += weight * loss
                total_weight += weight
        return total_loss / max(total_weight, 1e-9)
