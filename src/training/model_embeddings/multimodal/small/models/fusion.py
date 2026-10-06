"""Modality fusion module for combining embeddings from different modalities."""

import logging

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class ModalityProjector(nn.Module):
    """Projects modality-specific embeddings to a common dimension."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = output_dim

        # Final projection without activation
        layers.append(nn.Linear(current_dim, output_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different modality embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, dim]
            key: [batch, seq_k, dim]
            value: [batch, seq_v, dim]
        """
        # Cross attention
        attn_output, _ = self.attention(query, key, value)
        x = self.norm1(query + attn_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class FusionTransformerBlock(nn.Module):
    """Transformer block for multimodal fusion."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class ModalityFusion(nn.Module):
    """
    Fuses embeddings from multiple modalities into a unified representation.

    Supports:
    - Simple concatenation + projection
    - Attention-based fusion
    - Transformer-based fusion with early exit support
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        num_modalities: int = 3,  # text, image, audio
        fusion_type: str = "attention",  # simple, attention, transformer
        num_fusion_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        enable_layer_outputs: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        self.num_fusion_layers = num_fusion_layers
        self.enable_layer_outputs = enable_layer_outputs

        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(num_modalities, hidden_dim)

        # Input projections for each modality
        self.input_projections = nn.ModuleDict(
            {
                "text": ModalityProjector(input_dim, hidden_dim),
                "image": ModalityProjector(input_dim, hidden_dim),
                "audio": ModalityProjector(input_dim, hidden_dim),
            }
        )

        if fusion_type == "simple":
            # Simple concatenation + MLP
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        elif fusion_type == "attention":
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.output_projection = nn.Linear(hidden_dim, output_dim)

        elif fusion_type == "transformer":
            # Full transformer fusion
            self.fusion_layers = nn.ModuleList(
                [
                    FusionTransformerBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for _ in range(num_fusion_layers)
                ]
            )
            self.output_projection = nn.Linear(hidden_dim, output_dim)

            # Per-layer projections for 2DMSE
            if enable_layer_outputs:
                self.layer_projections = nn.ModuleList(
                    [
                        nn.Linear(hidden_dim, output_dim)
                        for _ in range(num_fusion_layers)
                    ]
                )
                self.layer_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(num_fusion_layers)]
                )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _project_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int, torch.device]:
        """Project available modalities and attach learned modality embeddings."""
        available = {
            name: value for name, value in embeddings.items() if value is not None
        }
        if not available:
            raise ValueError("At least one modality embedding must be provided")

        first_embedding = next(iter(available.values()))
        batch_size = first_embedding.shape[0]
        device = first_embedding.device
        modality_idx = {"text": 0, "image": 1, "audio": 2}
        projected = {}
        for modality, embedding in available.items():
            if embedding.shape[0] != batch_size:
                raise ValueError(
                    "All modality embeddings must have the same batch size"
                )
            projection = self.input_projections[modality](embedding)
            type_embedding = self.modality_embeddings(
                torch.tensor([modality_idx[modality]], device=device)
            ).expand(batch_size, -1)
            projected[modality] = projection + type_embedding
        return projected, batch_size, device

    def _simple_fusion(
        self,
        projected: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Concatenate projected modalities, padding missing inputs with zeros."""
        modality_list = [
            projected.get(
                modality,
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for modality in ("text", "image", "audio")
        ]
        return self.fusion(torch.cat(modality_list, dim=-1))

    def _attention_fusion(
        self,
        projected: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse available modalities with cross-modal attention."""
        available = list(projected.values())
        if len(available) == 1:
            return self.output_projection(available[0])
        stacked = torch.stack(available, dim=1)
        query = stacked.mean(dim=1, keepdim=True)
        fused = self.cross_attention(query, stacked, stacked)
        return self.output_projection(fused.squeeze(1))

    def _transformer_fusion(
        self,
        projected: dict[str, torch.Tensor],
        target_layer: int | None,
    ) -> torch.Tensor:
        """Fuse available modalities through the transformer stack."""
        available = list(projected.values())
        x = (
            available[0].unsqueeze(1)
            if len(available) == 1
            else torch.stack(available, dim=1)
        )
        for layer_idx, layer in enumerate(self.fusion_layers):
            x = layer(x)
            if target_layer is None or layer_idx != target_layer:
                continue
            pooled = x.mean(dim=1)
            if self.enable_layer_outputs:
                normalized = self.layer_norms[layer_idx](x)
                return self.layer_projections[layer_idx](normalized.mean(dim=1))
            return self.output_projection(pooled)
        return self.output_projection(x.mean(dim=1))

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        target_layer: int | None = None,  # For 2DMSE
        target_dim: int | None = None,  # For MRL
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Fuse modality embeddings.

        Args:
            embeddings: Dict with keys 'text', 'image', 'audio' and embedding tensors
            target_layer: Exit at this fusion layer (transformer mode only)
            target_dim: Truncate output to this dimension
            normalize: Whether to L2 normalize output

        Returns:
            fused_embedding: [batch_size, output_dim]
        """
        projected, batch_size, device = self._project_modalities(embeddings)
        if self.fusion_type == "simple":
            output = self._simple_fusion(projected, batch_size, device)
        elif self.fusion_type == "attention":
            output = self._attention_fusion(projected)
        else:
            output = self._transformer_fusion(projected, target_layer)

        # Dimension truncation (MRL)
        if target_dim is not None:
            output = output[:, :target_dim]

        # Normalize
        if normalize:
            output = functional.normalize(output, p=2, dim=-1)

        return output
