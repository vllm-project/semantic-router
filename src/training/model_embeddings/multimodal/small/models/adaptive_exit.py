"""Confidence-based adaptive layer exit for 2DMSE."""

import logging
from typing import Any

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence of embeddings at each layer.
    Used for adaptive early exit in 2DMSE.

    The estimator learns to predict when early layer embeddings
    are sufficient for the task, avoiding unnecessary computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        threshold: float = 0.8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.threshold = threshold

        # Per-layer confidence heads (small MLPs)
        self.confidence_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, embedding: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Estimate confidence for embedding at given layer.

        Args:
            embedding: [batch_size, hidden_dim] embedding
            layer_idx: Which layer this embedding came from

        Returns:
            confidence: [batch_size, 1] confidence scores in [0, 1]
        """
        return self.confidence_heads[layer_idx](embedding)

    def should_exit(
        self, embedding: torch.Tensor, layer_idx: int
    ) -> tuple[bool, float]:
        """
        Decide whether to exit at this layer based on confidence.

        Returns:
            (should_exit, confidence_score)
        """
        with torch.no_grad():
            confidence = self.forward(embedding, layer_idx)
            mean_conf = confidence.mean().item()
            return mean_conf >= self.threshold, mean_conf


class AdaptiveLayerExitEncoder(nn.Module):
    """
    Wrapper that adds adaptive layer exit capability to any encoder.

    Automatically selects the optimal layer based on:
    1. Confidence estimation
    2. Minimum/maximum layer constraints
    3. Quality vs latency tradeoff
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_layers: int,
        exit_layers: list[int] | None = None,
        confidence_threshold: float = 0.8,
        min_layer: int = 0,
        max_layer: int | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.confidence_threshold = confidence_threshold
        self.min_layer = min_layer
        self.max_layer = max_layer or num_layers

        # Default exit layers (evenly spaced)
        if exit_layers is None:
            # For 22 layers: [5, 10, 15, 22]
            # For 27 layers: [6, 13, 20, 27]
            step = max(1, num_layers // 4)
            exit_layers = list(range(step, num_layers, step))
            if num_layers not in exit_layers:
                exit_layers.append(num_layers)

        self.exit_layers = sorted(exit_layers)

        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator(
            hidden_dim=hidden_dim,
            num_layers=len(exit_layers),
            threshold=confidence_threshold,
        )

        # Track exit statistics
        self.register_buffer("exit_counts", torch.zeros(len(exit_layers)))
        self.register_buffer("total_samples", torch.tensor(0))

    def forward(
        self,
        *args,
        adaptive: bool = True,
        target_layer: int | None = None,
        return_confidence: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward with adaptive layer exit.

        Args:
            adaptive: If True, use confidence-based exit. If False, use target_layer.
            target_layer: Fixed layer to exit at (overrides adaptive)
            return_confidence: If True, return (embedding, confidence, exit_layer)
        """
        # If target_layer specified, use fixed exit
        if target_layer is not None:
            embedding = self.encoder(*args, target_layer=target_layer, **kwargs)
            if return_confidence:
                return embedding, 1.0, target_layer
            return embedding

        # If not adaptive, use full model
        if not adaptive:
            embedding = self.encoder(*args, **kwargs)
            if return_confidence:
                return embedding, 1.0, self.num_layers
            return embedding

        # Adaptive exit: try each exit layer until confident
        for i, layer in enumerate(self.exit_layers):
            if layer < self.min_layer:
                continue
            if layer > self.max_layer:
                break

            # Get embedding at this layer
            embedding = self.encoder(*args, target_layer=layer, **kwargs)

            # Check confidence
            should_exit, confidence = self.confidence_estimator.should_exit(
                embedding, i
            )

            if should_exit or layer == self.exit_layers[-1]:
                # Update statistics (only during training)
                if self.training:
                    self.exit_counts[i] += embedding.shape[0]
                    self.total_samples += embedding.shape[0]

                if return_confidence:
                    return embedding, confidence, layer
                return embedding

        # Fallback to full model
        embedding = self.encoder(*args, **kwargs)
        if return_confidence:
            return embedding, 1.0, self.num_layers
        return embedding

    def get_exit_statistics(self) -> dict[str, float]:
        """Get statistics about which layers are being used."""
        if self.total_samples == 0:
            return {}

        stats = {}
        for i, layer in enumerate(self.exit_layers):
            pct = (self.exit_counts[i] / self.total_samples * 100).item()
            stats[f"layer_{layer}_pct"] = pct

        # Average layer
        weighted_sum = sum(
            layer * self.exit_counts[i].item()
            for i, layer in enumerate(self.exit_layers)
        )
        stats["avg_exit_layer"] = weighted_sum / max(1, self.total_samples.item())
        stats["speedup_estimate"] = self.num_layers / max(1, stats["avg_exit_layer"])

        return stats

    def reset_statistics(self):
        """Reset exit statistics."""
        self.exit_counts.zero_()
        self.total_samples.zero_()


class AdaptiveMultimodalExit(nn.Module):
    """
    Manages adaptive exit across multiple modalities.

    Each modality can exit independently based on its own confidence,
    allowing heterogeneous compute allocation.
    """

    def __init__(
        self,
        text_encoder: AdaptiveLayerExitEncoder | None = None,
        image_encoder: AdaptiveLayerExitEncoder | None = None,
        audio_encoder: AdaptiveLayerExitEncoder | None = None,
        fusion_module: nn.Module | None = None,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.fusion_module = fusion_module

    @staticmethod
    def _encode_modality(
        encoder: AdaptiveLayerExitEncoder,
        input_name: str,
        value: Any,
        *,
        adaptive: bool,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float | int] | None]:
        """Encode one modality and retain adaptive-exit metadata when requested."""
        encoder_kwargs = {input_name: value, **kwargs}
        if not adaptive:
            return encoder(adaptive=False, **encoder_kwargs), None

        embedding, confidence, layer = encoder(
            adaptive=True,
            return_confidence=True,
            **encoder_kwargs,
        )
        return embedding, {"confidence": confidence, "layer": layer}

    def forward(
        self,
        texts: list[str] | None = None,
        images: list | None = None,
        audio: list | None = None,
        adaptive: bool = True,
        return_exit_info: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode with adaptive exit per modality.

        Returns:
            embedding or (embedding, exit_info) if return_exit_info=True
        """
        embeddings: dict[str, torch.Tensor] = {}
        exit_info: dict[str, dict[str, float | int]] = {}
        modality_inputs = (
            ("text", "texts", texts, self.text_encoder),
            ("image", "images", images, self.image_encoder),
            ("audio", "audio", audio, self.audio_encoder),
        )
        for modality, input_name, value, encoder in modality_inputs:
            if value is None or encoder is None:
                continue
            embedding, info = self._encode_modality(
                encoder,
                input_name,
                value,
                adaptive=adaptive,
                kwargs=kwargs,
            )
            embeddings[modality] = embedding
            if info is not None:
                exit_info[modality] = info

        # Fuse if multiple modalities
        if len(embeddings) > 1 and self.fusion_module is not None:
            fused = self.fusion_module(embeddings)
        elif len(embeddings) == 1:
            fused = next(iter(embeddings.values()))
        else:
            raise ValueError("No embeddings produced")

        if return_exit_info:
            return fused, exit_info
        return fused

    def get_all_statistics(self) -> dict[str, dict[str, float]]:
        """Get exit statistics for all modalities."""
        stats = {}
        if self.text_encoder is not None:
            stats["text"] = self.text_encoder.get_exit_statistics()
        if self.image_encoder is not None:
            stats["image"] = self.image_encoder.get_exit_statistics()
        if self.audio_encoder is not None:
            stats["audio"] = self.audio_encoder.get_exit_statistics()
        return stats


class ConfidenceTrainingLoss(nn.Module):
    """
    Loss function for training confidence estimators.

    The confidence estimator should predict high confidence when:
    1. Early layer embedding is similar to full model embedding
    2. The task can be solved with early exit
    """

    def __init__(self, similarity_threshold: float = 0.95):
        super().__init__()
        self.similarity_threshold = similarity_threshold

    def forward(
        self,
        early_embedding: torch.Tensor,
        full_embedding: torch.Tensor,
        predicted_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence training loss.

        Target: confidence should be high when early ≈ full embedding.
        """
        # Compute similarity between early and full embeddings
        similarity = functional.cosine_similarity(
            early_embedding,
            full_embedding,
            dim=-1,
        )

        # Target confidence: 1 if similar enough, 0 otherwise
        target_confidence = (similarity >= self.similarity_threshold).float()

        # BCE loss
        loss = functional.binary_cross_entropy(
            predicted_confidence.squeeze(-1),
            target_confidence,
        )

        return loss
