"""Matryoshka Representation Learning (MRL) and 2DMSE loss functions."""

import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning (MRL) loss.

    Trains embeddings to be useful at multiple dimension truncations.
    """

    def __init__(
        self,
        dim_schedule: list[int],
        base_loss_fn: Callable | None = None,
        dim_weights: list[float] | None = None,
        temperature: float = 0.05,
    ):
        """
        Args:
            dim_schedule: List of dimensions to train [64, 128, 256, 512, 1024]
            base_loss_fn: Base contrastive loss function
            dim_weights: Weights for each dimension (default: equal)
            temperature: Temperature for default InfoNCE loss
        """
        super().__init__()

        self.dim_schedule = sorted(dim_schedule)
        self.temperature = temperature

        if base_loss_fn is None:
            self.base_loss_fn = self._infonce_loss
        else:
            self.base_loss_fn = base_loss_fn

        if dim_weights is None:
            self.dim_weights = [1.0] * len(dim_schedule)
        else:
            assert len(dim_weights) == len(dim_schedule)
            self.dim_weights = dim_weights

    def _infonce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """Default InfoNCE loss."""
        batch_size = anchor.shape[0]
        device = anchor.device

        # Normalize
        anchor = functional.normalize(anchor, dim=-1)
        positive = functional.normalize(positive, dim=-1)

        # Similarity matrix
        sim = torch.matmul(anchor, positive.T) / self.temperature

        # Labels (diagonal is positive)
        labels = torch.arange(batch_size, device=device)

        return functional.cross_entropy(sim, labels)

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MRL loss across all dimension levels.

        Args:
            anchor_embeddings: [batch_size, full_dim]
            positive_embeddings: [batch_size, full_dim]
        """
        total_loss = 0.0
        total_weight = 0.0

        for dim, weight in zip(self.dim_schedule, self.dim_weights, strict=False):
            # Truncate to dimension
            anchor_trunc = anchor_embeddings[:, :dim]
            positive_trunc = positive_embeddings[:, :dim]

            # Compute loss at this dimension
            loss = self.base_loss_fn(anchor_trunc, positive_trunc)

            total_loss += weight * loss
            total_weight += weight

        return total_loss / total_weight


class TwoDMSELoss(nn.Module):
    """
    2D Matryoshka Sentence Embedding (2DMSE) loss.

    Extends MRL to also train across transformer layers,
    enabling both dimension and layer flexibility.
    """

    def __init__(
        self,
        layer_schedule: list[int],
        dim_schedule: list[int],
        temperature: float = 0.05,
        kl_weight: float = 0.1,
        contrastive_weight: float = 1.0,
        layer_weights: list[float] | None = None,
        dim_weights: list[float] | None = None,
    ):
        """
        Args:
            layer_schedule: Which layers to train [4, 8, 12, 16, 24]
            dim_schedule: Which dimensions to train [64, 128, 256, 512, 1024]
            temperature: Contrastive loss temperature
            kl_weight: Weight for KL divergence alignment
            contrastive_weight: Weight for contrastive loss
            layer_weights: Per-layer weights (deeper layers weighted more)
            dim_weights: Per-dimension weights
        """
        super().__init__()

        self.layer_schedule = sorted(layer_schedule)
        self.dim_schedule = sorted(dim_schedule)
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight

        # Default weights: deeper layers and larger dims weighted more
        if layer_weights is None:
            self.layer_weights = [
                (i + 1) / len(layer_schedule) for i in range(len(layer_schedule))
            ]
        else:
            self.layer_weights = layer_weights

        if dim_weights is None:
            self.dim_weights = [
                (i + 1) / len(dim_schedule) for i in range(len(dim_schedule))
            ]
        else:
            self.dim_weights = dim_weights

    def forward(
        self,
        model: nn.Module,
        anchor_inputs: dict,
        positive_inputs: dict,
        teacher_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute 2DMSE loss across layer-dimension grid.

        Args:
            model: Embedding model with target_layer and target_dim support
            anchor_inputs: Dict of inputs for anchor samples
            positive_inputs: Dict of inputs for positive samples
            teacher_embeddings: Optional precomputed full model embeddings

        Returns:
            Dict with total loss and breakdown
        """
        # Get teacher embeddings (full model, full dimension)
        if teacher_embeddings is None:
            with torch.no_grad():
                teacher_anchor = model(
                    **anchor_inputs, target_layer=None, target_dim=None
                )
                teacher_positive = model(
                    **positive_inputs, target_layer=None, target_dim=None
                )
        else:
            teacher_anchor = teacher_embeddings["anchor"]
            teacher_positive = teacher_embeddings["positive"]

        total_loss = 0.0
        contrastive_losses = []
        kl_losses = []

        for layer_idx, layer_weight in zip(
            self.layer_schedule, self.layer_weights, strict=False
        ):
            for dim, dim_weight in zip(
                self.dim_schedule, self.dim_weights, strict=False
            ):
                # Get student embeddings at (layer, dim)
                student_anchor = model(
                    **anchor_inputs,
                    target_layer=layer_idx,
                    target_dim=dim,
                )
                student_positive = model(
                    **positive_inputs,
                    target_layer=layer_idx,
                    target_dim=dim,
                )

                # 1. Contrastive loss
                contrastive_loss = self._contrastive_loss(
                    student_anchor,
                    student_positive,
                )
                contrastive_losses.append(contrastive_loss.item())

                # 2. KL alignment with teacher
                teacher_anchor_trunc = functional.normalize(
                    teacher_anchor[:, :dim], dim=-1
                )
                teacher_positive_trunc = functional.normalize(
                    teacher_positive[:, :dim], dim=-1
                )

                kl_loss = self._kl_alignment_loss(
                    student_anchor,
                    teacher_anchor_trunc,
                ) + self._kl_alignment_loss(
                    student_positive,
                    teacher_positive_trunc,
                )
                kl_loss = kl_loss / 2
                kl_losses.append(kl_loss.item())

                # Combine
                pair_weight = layer_weight * dim_weight
                pair_loss = (
                    self.contrastive_weight * contrastive_loss
                    + self.kl_weight * kl_loss
                )
                total_loss += pair_weight * pair_loss

        # Normalize by total weight
        total_weight = sum(
            lw * dw for lw in self.layer_weights for dw in self.dim_weights
        )
        total_loss = total_loss / total_weight

        return {
            "loss": total_loss,
            "contrastive_loss": sum(contrastive_losses) / len(contrastive_losses),
            "kl_loss": sum(kl_losses) / len(kl_losses),
        }

    def _contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        batch_size = anchor.shape[0]
        device = anchor.device

        # Already normalized by model
        sim = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(batch_size, device=device)

        loss = functional.cross_entropy(sim, labels) + functional.cross_entropy(
            sim.T, labels
        )
        return loss / 2

    def _kl_alignment_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence for aligning student to teacher distributions."""
        # Softmax over embedding dimension
        student_dist = functional.log_softmax(student / self.temperature, dim=-1)
        teacher_dist = functional.softmax(teacher / self.temperature, dim=-1)

        return functional.kl_div(student_dist, teacher_dist, reduction="batchmean")


class LayerWiseDistillationLoss(nn.Module):
    """
    Layer-wise distillation loss for 2DMSE.

    Distills knowledge from deeper layers to shallower layers.
    """

    def __init__(
        self,
        layer_pairs: list[tuple],  # [(shallow, deep), ...]
        loss_type: str = "mse",  # mse, cosine, kl
        temperature: float = 1.0,
    ):
        super().__init__()

        self.layer_pairs = layer_pairs
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        layer_embeddings: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute distillation loss between layer pairs.

        Args:
            layer_embeddings: Dict mapping layer index to embeddings
        """
        total_loss = 0.0

        for shallow_layer, deep_layer in self.layer_pairs:
            if (
                shallow_layer not in layer_embeddings
                or deep_layer not in layer_embeddings
            ):
                continue

            student = layer_embeddings[shallow_layer]
            teacher = layer_embeddings[deep_layer].detach()

            if self.loss_type == "mse":
                loss = functional.mse_loss(student, teacher)
            elif self.loss_type == "cosine":
                loss = 1 - functional.cosine_similarity(student, teacher, dim=-1).mean()
            elif self.loss_type == "kl":
                student_dist = functional.log_softmax(
                    student / self.temperature, dim=-1
                )
                teacher_dist = functional.softmax(teacher / self.temperature, dim=-1)
                loss = functional.kl_div(
                    student_dist, teacher_dist, reduction="batchmean"
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss += loss

        return total_loss / len(self.layer_pairs) if self.layer_pairs else total_loss
