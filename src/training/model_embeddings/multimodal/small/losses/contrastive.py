"""Contrastive loss functions for embedding training."""

import logging

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss (NT-Xent).

    Computes contrastive loss between anchor and positive embeddings
    using in-batch negatives.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, dim] normalized anchor embeddings
            positive_embeddings: [batch_size, dim] normalized positive embeddings

        Returns:
            Contrastive loss where diagonal elements are positive pairs
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        # Normalize (in case not already normalized)
        anchor_embeddings = functional.normalize(anchor_embeddings, dim=-1)
        positive_embeddings = functional.normalize(positive_embeddings, dim=-1)

        # Compute similarity matrix: [batch, batch]
        # sim[i,j] = similarity between anchor_i and positive_j
        sim_matrix = (
            torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature
        )

        # Labels: diagonal elements are positive pairs (anchor_i matches positive_i)
        labels = torch.arange(batch_size, device=device)

        # Cross entropy loss (InfoNCE)
        # For each anchor, positive is at diagonal position
        loss_a2p = functional.cross_entropy(sim_matrix, labels, reduction="none")

        # Symmetric: for each positive, anchor is at diagonal position
        loss_p2a = functional.cross_entropy(sim_matrix.T, labels, reduction="none")

        loss = (loss_a2p + loss_p2a) / 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SigmoidContrastiveLoss(nn.Module):
    """
    Sigmoid contrastive loss (SigLIP style).

    Instead of softmax normalization, uses independent sigmoid
    for each pair, providing better geometric properties.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        bias: float = -10.0,  # Learnable bias init
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.tensor(bias))
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, dim]
            positive_embeddings: [batch_size, dim]
            negative_embeddings: Optional [batch_size, num_negatives, dim]
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        # Positive similarities
        pos_sim = functional.cosine_similarity(anchor_embeddings, positive_embeddings)
        pos_logits = pos_sim / self.temperature + self.bias

        # Positive loss: -log(sigmoid(pos_logits))
        pos_loss = functional.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction="none",
        )

        if negative_embeddings is not None:
            # Explicit negatives provided
            # negative_embeddings: [batch, num_neg, dim]
            # Compute negative similarities
            # anchor: [batch, 1, dim], neg: [batch, num_neg, dim]
            anchor_exp = anchor_embeddings.unsqueeze(1)
            neg_sim = functional.cosine_similarity(
                anchor_exp, negative_embeddings, dim=-1
            )
            neg_logits = neg_sim / self.temperature + self.bias

            # Negative loss: -log(sigmoid(-neg_logits)) = -log(1 - sigmoid(neg_logits))
            neg_loss = functional.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits),
                reduction="none",
            )
            neg_loss = neg_loss.mean(dim=1)  # Average over negatives
        else:
            # In-batch negatives
            # All other samples in batch are negatives
            sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)
            sim_matrix = sim_matrix / self.temperature + self.bias

            # Diagonal is positive, off-diagonal is negative
            labels = torch.eye(batch_size, device=device)
            neg_loss = functional.binary_cross_entropy_with_logits(
                sim_matrix,
                labels,
                reduction="none",
            )
            # Exclude diagonal (positive pairs)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_loss = neg_loss[mask].view(batch_size, -1).mean(dim=1)

        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss.

    Optimized for retrieval where all other samples in batch
    serve as negatives.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, dim]
            positive_embeddings: [batch_size, dim]
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        # Compute similarity matrix
        sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)
        sim_matrix = sim_matrix / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)

        # Cross entropy loss
        loss = self.cross_entropy(sim_matrix, labels)

        return loss


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive loss with hard negative mining.

    Selects hardest negatives within batch for more effective training.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        num_hard_negatives: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, dim]
            positive_embeddings: [batch_size, dim]
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        # Compute similarity matrix
        sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)

        # Mask diagonal (positive pairs)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        # For each anchor, find hardest negatives
        negative_sims = sim_matrix.masked_fill(~mask, float("-inf"))
        hard_neg_indices = negative_sims.topk(
            min(self.num_hard_negatives, batch_size - 1),
            dim=1,
        ).indices

        # Gather hard negative similarities
        hard_neg_sims = torch.gather(sim_matrix, 1, hard_neg_indices)

        # Positive similarities (diagonal)
        pos_sims = sim_matrix.diag().unsqueeze(1)

        # Concatenate: [pos, hard_negs]
        logits = torch.cat([pos_sims, hard_neg_sims], dim=1) / self.temperature

        # Labels: first column is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        loss = functional.cross_entropy(logits, labels, reduction=self.reduction)

        return loss
