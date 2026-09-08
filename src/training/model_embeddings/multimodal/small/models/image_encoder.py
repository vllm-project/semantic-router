"""Image encoder based on SigLIP 2 / ViT architecture."""

import logging
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from transformers import AutoConfig, AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Image encoder using SigLIP 2 / ViT architecture.

    Default: SigLIP 2 SO400M (384px, 400M params)
    - SOTA vision encoder for multimodal learning
    - Multilingual support
    - Apache 2.0 license

    Supports adaptive layer exit (2DMSE) and dimension truncation (MRL).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-so400m-patch14-384",
        revision: str | None = None,
        output_dim: int = 768,
        pooling_mode: str = "mean",
        normalize: bool = True,
        enable_layer_outputs: bool = True,
    ):
        super().__init__()

        self.model_name = model_name_or_path
        self.revision = revision
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.normalize = normalize
        self.enable_layer_outputs = enable_layer_outputs

        # Check model type
        self.is_siglip2 = "siglip2" in model_name_or_path.lower()
        self.is_siglip = "siglip" in model_name_or_path.lower()

        # Load base model
        logger.info(f"Loading image encoder: {model_name_or_path}")

        try:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=True,
            )
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=True,
            )

            # Get vision encoder component
            if hasattr(self.encoder, "vision_model"):
                self.vision_encoder = self.encoder.vision_model
                vision_config = getattr(
                    self.encoder.config, "vision_config", self.config
                )
                self.hidden_size = vision_config.hidden_size
                self.num_layers = vision_config.num_hidden_layers
            else:
                self.vision_encoder = self.encoder
                self.hidden_size = self.config.hidden_size
                self.num_layers = self.config.num_hidden_layers

            logger.info(
                f"Image encoder: {self.num_layers} layers, {self.hidden_size}d hidden"
            )
            if self.is_siglip2:
                logger.info(
                    "Using SigLIP 2 with improved localization and dense features"
                )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load configured image encoder {model_name_or_path}; "
                "refusing to substitute a different architecture"
            ) from exc

        # Projection to output dimension
        if self.hidden_size != output_dim:
            self.projection = nn.Linear(self.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()

        # Per-layer projections for 2DMSE
        self.layer_projections: nn.ModuleList | None = None
        if enable_layer_outputs:
            self.layer_projections = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, output_dim)
                    for _ in range(self.num_layers)
                ]
            )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to get image embedding."""
        if self.pooling_mode == "mean":
            # Skip CLS token if present, average over patches
            if hidden_states.shape[1] > 1:
                return hidden_states[:, 1:].mean(dim=1)
            return hidden_states.mean(dim=1)
        elif self.pooling_mode == "cls":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def preprocess(
        self,
        images: Image.Image | list[Image.Image] | torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess images for the encoder."""
        if isinstance(images, torch.Tensor):
            return images

        if isinstance(images, Image.Image):
            images = [images]

        processed = self.processor(images=images, return_tensors="pt")
        return processed["pixel_values"]

    def _prepare_pixel_values(
        self,
        pixel_values: torch.Tensor | None,
        images: Image.Image | list[Image.Image] | None,
    ) -> torch.Tensor | None:
        """Preprocess raw images and move them to the encoder device."""
        if pixel_values is not None or images is None:
            return pixel_values
        prepared = self.preprocess(images)
        parameter = next(self.parameters(), None)
        device = (
            parameter.device
            if parameter is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return prepared.to(device)

    @staticmethod
    def _hidden_states(outputs: Any) -> tuple[torch.Tensor, ...] | None:
        """Return encoder-layer states without the input embedding state."""
        hidden_states = getattr(outputs, "hidden_states", None)
        return hidden_states[1:] if hidden_states is not None else None

    def _project_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Pool and project one intermediate vision layer."""
        pooled = self._pool(hidden_states)
        if self.layer_projections is not None:
            return self.layer_projections[layer_idx](pooled)
        return self.projection(pooled)

    def _select_embedding(
        self,
        outputs: Any,
        all_hidden_states: tuple[torch.Tensor, ...] | None,
        target_layer: int | None,
    ) -> torch.Tensor:
        """Select either an intermediate 2DMSE layer or the final output."""
        if target_layer is not None and all_hidden_states is not None:
            layer_idx = min(target_layer, len(all_hidden_states) - 1)
            return self._project_layer(all_hidden_states[layer_idx], layer_idx)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = self._pool(outputs.last_hidden_state)
        return self.projection(pooled)

    def _all_layer_embeddings(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Project every vision layer for confidence-based exit training."""
        embeddings = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            embedding = self._project_layer(hidden_states, layer_idx)
            if self.normalize:
                embedding = functional.normalize(embedding, p=2, dim=-1)
            embeddings.append(embedding)
        return embeddings

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        target_layer: int | None = None,
        target_dim: int | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for image encoding.

        Args:
            pixel_values: Preprocessed image tensors
            images: Raw PIL images
            target_layer: Exit at this layer (2DMSE)
            target_dim: Truncate to this dimension (MRL)
            return_all_layers: Return embeddings from all layers
        """
        pixel_values = self._prepare_pixel_values(pixel_values, images)

        # Forward through encoder
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=self.enable_layer_outputs or return_all_layers,
        )

        all_hidden_states = self._hidden_states(outputs)
        embedding = self._select_embedding(outputs, all_hidden_states, target_layer)

        # Dimension truncation (MRL)
        if target_dim is not None:
            embedding = embedding[:, :target_dim]

        # Normalize
        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        # Return all layer embeddings if requested
        if return_all_layers and all_hidden_states is not None:
            return embedding, self._all_layer_embeddings(all_hidden_states)

        return embedding

    def encode(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        target_layer: int | None = None,
        target_dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode images into embeddings."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.forward(
                    images=batch_images,
                    target_layer=target_layer,
                    target_dim=target_dim,
                    **kwargs,
                )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# Legacy alias for backward compatibility
DiffusionImageEncoder = ImageEncoder  # Deprecated
