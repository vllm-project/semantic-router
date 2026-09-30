"""Cross-modal alignment loss functions."""

import logging

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class CrossModalAlignmentLoss(nn.Module):
    """
    Loss for aligning embeddings across different modalities.

    Ensures that semantically similar content from different modalities
    maps to nearby points in the embedding space.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        symmetric: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.reduction = reduction

    def forward(
        self,
        modality_a: torch.Tensor,
        modality_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute cross-modal alignment loss.

        Args:
            modality_a: Embeddings from modality A [batch, dim]
            modality_b: Embeddings from modality B [batch, dim]
            labels: Optional matching labels (default: diagonal is positive)
        """
        batch_size = modality_a.shape[0]
        device = modality_a.device

        # Normalize
        modality_a = functional.normalize(modality_a, dim=-1)
        modality_b = functional.normalize(modality_b, dim=-1)

        # Cross-modal similarity
        sim_a_to_b = torch.matmul(modality_a, modality_b.T) / self.temperature

        if labels is None:
            # Default: diagonal is positive (matched pairs)
            labels = torch.arange(batch_size, device=device)

        # A -> B loss
        loss_a_to_b = functional.cross_entropy(
            sim_a_to_b, labels, reduction=self.reduction
        )

        if self.symmetric:
            # B -> A loss
            sim_b_to_a = sim_a_to_b.T
            loss_b_to_a = functional.cross_entropy(
                sim_b_to_a, labels, reduction=self.reduction
            )
            return (loss_a_to_b + loss_b_to_a) / 2

        return loss_a_to_b


class MultiModalAlignmentLoss(nn.Module):
    """
    Alignment loss for multiple modalities.

    Aligns all modality pairs and optionally to a fused representation.
    """

    def __init__(
        self,
        modalities: list[str] | None = None,
        temperature: float = 0.05,
        align_to_fusion: bool = True,
    ):
        if modalities is None:
            modalities = ["text", "image", "audio"]
        super().__init__()
        self.modalities = modalities
        self.temperature = temperature
        self.align_to_fusion = align_to_fusion

        self.pairwise_loss = CrossModalAlignmentLoss(
            temperature=temperature,
            symmetric=True,
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        fused_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-modal alignment loss.

        Args:
            embeddings: Dict mapping modality name to embeddings
            fused_embedding: Optional fused representation

        Returns:
            Dict with total loss and per-pair losses
        """
        losses = {}
        total_loss = 0.0
        num_pairs = 0

        # Pairwise alignment
        modality_list = [m for m in self.modalities if m in embeddings]

        for i, mod_a in enumerate(modality_list):
            for mod_b in modality_list[i + 1 :]:
                pair_name = f"{mod_a}-{mod_b}"
                loss = self.pairwise_loss(
                    embeddings[mod_a],
                    embeddings[mod_b],
                )
                losses[pair_name] = loss
                total_loss += loss
                num_pairs += 1

        # Align to fusion
        if self.align_to_fusion and fused_embedding is not None:
            for modality in modality_list:
                fusion_name = f"{modality}-fusion"
                loss = self.pairwise_loss(
                    embeddings[modality],
                    fused_embedding,
                )
                losses[fusion_name] = loss
                total_loss += loss
                num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        losses["total"] = total_loss
        return losses


class IDAlignmentLoss(nn.Module):
    """
    ID-guided alignment loss (from DiffCL).

    Uses stable ID embeddings to guide cross-modal semantic alignment.
    ID embeddings act as anchors that don't change during training.
    """

    def __init__(
        self,
        id_dim: int = 256,
        hidden_dim: int = 1024,
        temperature: float = 0.1,
        margin: float = 0.2,
    ):
        super().__init__()
        self.id_dim = id_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.margin = margin

        # ID projection (frozen during training)
        self.id_projection = nn.Sequential(
            nn.Linear(hidden_dim, id_dim),
            nn.LayerNorm(id_dim),
        )

        # Initialize and freeze
        self._init_id_projection()

    def _init_id_projection(self):
        """Initialize ID projection with orthogonal weights."""
        for module in self.id_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Freeze
        for param in self.id_projection.parameters():
            param.requires_grad = False

    def get_id_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get stable ID embedding."""
        with torch.no_grad():
            id_emb = self.id_projection(embedding)
            return functional.normalize(id_emb, dim=-1)

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ID alignment loss.

        Ensures that paired samples have similar ID embeddings.

        Args:
            embedding_a: Embeddings from view A [batch, dim]
            embedding_b: Embeddings from view B [batch, dim]
        """
        # Get ID embeddings
        id_a = self.get_id_embedding(embedding_a)
        id_b = self.get_id_embedding(embedding_b)

        # Positive similarity (matched pairs)
        pos_sim = functional.cosine_similarity(id_a, id_b, dim=-1)

        # Negative similarity (unmatched pairs)
        batch_size = embedding_a.shape[0]
        device = embedding_a.device

        # Similarity matrix
        sim_matrix = torch.matmul(id_a, id_b.T)

        # Mask diagonal (positive pairs)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim = sim_matrix[mask].view(batch_size, -1)

        # Hard negative
        hard_neg_sim = neg_sim.max(dim=1)[0]

        # Triplet-style loss with margin
        loss = functional.relu(self.margin - pos_sim + hard_neg_sim)

        return loss.mean()


class SemanticConsistencyLoss(nn.Module):
    """
    Ensures semantic consistency between augmented views.

    Different augmentations of the same content should have similar embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        consistency_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.consistency_weight = consistency_weight

    def forward(
        self,
        original_embedding: torch.Tensor,
        augmented_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic consistency loss.

        Args:
            original_embedding: Embedding of original content
            augmented_embedding: Embedding of augmented content
        """
        # Normalize
        original = functional.normalize(original_embedding, dim=-1)
        augmented = functional.normalize(augmented_embedding, dim=-1)

        # Cosine similarity
        similarity = functional.cosine_similarity(original, augmented, dim=-1)

        # Loss: encourage high similarity
        loss = 1 - similarity.mean()

        return self.consistency_weight * loss
