"""
Contrastive Loss Functions for Cache Embedding Training
=======================================================

Implementation of various contrastive learning losses optimized for
semantic cache matching, following best practices from the research literature.

Supported losses:
- Triplet Loss: Classic triplet margin loss
- InfoNCE Loss: Contrastive loss with temperature scaling
- Multiple Negatives Ranking (MNR) Loss: Recommended by the paper
- Cosine Embedding Loss: Simple cosine-based loss
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    """
    Triplet Loss for learning embeddings.

    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)

    Where d() is typically Euclidean or cosine distance.
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "cosine",
        reduction: str = "mean"
    ):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: "cosine" or "euclidean"
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction

        logger.info(f"TripletLoss initialized: margin={margin}, metric={distance_metric}")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
            neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
        elif self.distance_metric == "euclidean":
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Triplet loss: max(pos_dist - neg_dist + margin, 0)
        losses = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Normalized Temperature-scaled Cross Entropy Loss).

    Used in SimCLR and other contrastive learning methods.
    Encourages embeddings of similar pairs to be close while pushing
    apart dissimilar pairs.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        """
        Args:
            temperature: Temperature parameter for scaling (lower = harder)
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

        logger.info(f"InfoNCELoss initialized: temperature={temperature}")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negatives: Negative embeddings [batch_size, num_negatives, embedding_dim]

        Returns:
            Loss value
        """
        batch_size = anchor.size(0)

        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch_size]

        # If negatives has shape [batch_size, num_neg, dim]
        if negatives.dim() == 3:
            neg_sim = torch.bmm(
                negatives,
                anchor.unsqueeze(-1)
            ).squeeze(-1) / self.temperature  # [batch_size, num_neg]

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_neg]
        else:
            # Single negative per sample: [batch_size, dim]
            neg_sim = torch.sum(anchor * negatives, dim=-1) / self.temperature
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch_size, 2]

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking (MNR) Loss.

    Recommended by the research paper for semantic caching.
    Uses in-batch negatives: all other positives in the batch serve as negatives.

    This is efficient and effective for learning semantic similarity.
    """

    def __init__(self, temperature: float = 0.05, reduction: str = "mean"):
        """
        Args:
            temperature: Temperature parameter for scaling
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

        logger.info(f"MultipleNegativesRankingLoss initialized: temperature={temperature}")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MNR loss using in-batch negatives.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature

        # Labels: diagonal elements are positives (i matches i)
        labels = torch.arange(anchor.size(0), device=anchor.device)

        # Cross-entropy loss: maximize similarity with correct positive
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)

        return loss


class CosineEmbeddingLoss(nn.Module):
    """
    Cosine Embedding Loss.

    Simple loss for learning embeddings based on cosine similarity.
    """

    def __init__(self, margin: float = 0.5, reduction: str = "mean"):
        """
        Args:
            margin: Margin for dissimilar pairs
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

        logger.info(f"CosineEmbeddingLoss initialized: margin={margin}")

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine embedding loss.

        Args:
            embedding1: First embeddings [batch_size, embedding_dim]
            embedding2: Second embeddings [batch_size, embedding_dim]
            label: Labels [batch_size] (1 for similar, -1 for dissimilar)

        Returns:
            Loss value
        """
        return self.loss_fn(embedding1, embedding2, label)


class HardTripletLoss(nn.Module):
    """
    Hard Triplet Mining Loss.

    Selects hardest positive and negative examples within batch
    for more efficient training.
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "cosine",
        hard_mining: str = "hardest"
    ):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: "cosine" or "euclidean"
            hard_mining: "hardest", "semi_hard", or "all"
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.hard_mining = hard_mining

        logger.info(
            f"HardTripletLoss initialized: margin={margin}, "
            f"metric={distance_metric}, mining={hard_mining}"
        )

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.distance_metric == "cosine":
            # Cosine distance
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            dot_product = torch.matmul(embeddings_norm, embeddings_norm.T)
            distances = 1.0 - dot_product
        else:
            # Euclidean distance
            dot_product = torch.matmul(embeddings, embeddings.T)
            squared_norm = torch.diag(dot_product)
            distances = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0).sqrt()

        return distances

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard triplet mining loss.

        Args:
            embeddings: All embeddings [batch_size, embedding_dim]
            labels: Labels for each embedding [batch_size]

        Returns:
            Loss value
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)

        # Create masks for positives and negatives
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal

        # Mask to exclude self-comparisons
        mask_self = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

        # Positive mask: same label, not self
        positive_mask = labels_equal & (~mask_self)

        # Negative mask: different label
        negative_mask = labels_not_equal

        # Mine hardest positives and negatives
        if self.hard_mining == "hardest":
            # Hardest positive: maximum distance among positives
            max_positive_dist = torch.max(
                pairwise_dist * positive_mask.float() - (~positive_mask).float() * 1e9,
                dim=1
            )[0]

            # Hardest negative: minimum distance among negatives
            min_negative_dist = torch.min(
                pairwise_dist + (~negative_mask).float() * 1e9,
                dim=1
            )[0]

            # Triplet loss
            losses = F.relu(max_positive_dist - min_negative_dist + self.margin)

        elif self.hard_mining == "semi_hard":
            # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
            # This is more complex and typically requires iterating
            # For simplicity, fall back to hardest
            losses = self.forward(embeddings, labels)  # Recursion with hardest

        else:  # "all"
            # Use all valid triplets
            # Expand distances for triplet computation
            anchor_positive = pairwise_dist.unsqueeze(2)
            anchor_negative = pairwise_dist.unsqueeze(1)

            # Triplet loss matrix
            triplet_loss = anchor_positive - anchor_negative + self.margin

            # Mask valid triplets
            valid_triplets = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
            triplet_loss = triplet_loss * valid_triplets.float()

            # Remove easy triplets (negative loss)
            triplet_loss = F.relu(triplet_loss)

            # Average over valid triplets
            num_valid = valid_triplets.sum().float()
            losses = triplet_loss.sum() / (num_valid + 1e-16)
            return losses

        return losses.mean()


def get_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name.

    Args:
        loss_type: Type of loss ("triplet", "infonce", "mnr", "cosine", "hard_triplet")
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance

    Example:
        >>> loss_fn = get_loss_function("mnr", temperature=0.05)
        >>> loss = loss_fn(anchor_emb, positive_emb)
    """
    loss_registry = {
        "triplet": TripletLoss,
        "infonce": InfoNCELoss,
        "mnr": MultipleNegativesRankingLoss,
        "cosine": CosineEmbeddingLoss,
        "hard_triplet": HardTripletLoss,
    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_registry.keys())}"
        )

    return loss_registry[loss_type](**kwargs)
