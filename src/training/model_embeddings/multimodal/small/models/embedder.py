"""Main multimodal embedder combining all modality encoders."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional

from .audio_encoder import AudioEncoder
from .fusion import ModalityFusion
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

logger = logging.getLogger(__name__)


class MultimodalEmbedder(nn.Module):
    """
    Multimodal embedding model combining text, image, and audio encoders.

    Supports:
    - Single modality encoding
    - Multimodal fusion
    - Adaptive layer exit (2DMSE)
    - Dimension truncation (MRL)
    """

    def __init__(
        self,
        text_encoder_name: str = "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
        text_encoder_revision: str | None = None,
        image_encoder_name: str = "google/siglip2-so400m-patch14-384",
        image_encoder_revision: str | None = None,
        audio_encoder_name: str = "openai/whisper-base",
        audio_encoder_revision: str | None = None,
        output_dim: int = 768,
        fusion_type: str = "transformer",
        num_fusion_layers: int = 4,
        enable_layer_outputs: bool = True,
        pooling_mode: str = "mean",
        normalize: bool = True,
        max_text_length: int = 32768,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.normalize = normalize
        self.enable_layer_outputs = enable_layer_outputs
        self.fusion_type = fusion_type
        self.num_fusion_layers = num_fusion_layers
        self.max_text_length = max_text_length
        self.text_encoder_revision = text_encoder_revision
        self.image_encoder_revision = image_encoder_revision
        self.audio_encoder_revision = audio_encoder_revision

        # Initialize encoders
        logger.info("Initializing text encoder...")
        self.text_encoder = TextEncoder(
            model_name_or_path=text_encoder_name,
            revision=text_encoder_revision,
            output_dim=output_dim,
            pooling_mode=pooling_mode,
            normalize=False,
            enable_layer_outputs=enable_layer_outputs,
            max_length=max_text_length,
        )

        logger.info("Initializing image encoder...")
        self.image_encoder = ImageEncoder(
            model_name_or_path=image_encoder_name,
            revision=image_encoder_revision,
            output_dim=output_dim,
            pooling_mode=pooling_mode,
            normalize=False,
            enable_layer_outputs=enable_layer_outputs,
        )

        logger.info("Initializing audio encoder...")
        self.audio_encoder = AudioEncoder(
            model_name_or_path=audio_encoder_name,
            revision=audio_encoder_revision,
            output_dim=output_dim,
            pooling_mode=pooling_mode,
            normalize=False,
            enable_layer_outputs=enable_layer_outputs,
        )

        # Initialize fusion module
        logger.info(f"Initializing {fusion_type} fusion module...")
        self.fusion = ModalityFusion(
            input_dim=output_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            fusion_type=fusion_type,
            num_fusion_layers=num_fusion_layers,
            enable_layer_outputs=enable_layer_outputs,
        )

        # Store encoder info
        self.encoder_layers = {
            "text": self.text_encoder.num_layers,
            "image": self.image_encoder.num_layers,
            "audio": self.audio_encoder.num_layers,
            "fusion": num_fusion_layers,
        }

    def encode_text(
        self,
        texts: str | list[str],
        target_layer: int | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor:
        """Encode text into embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embedding = self.text_encoder(
            texts=texts,
            target_layer=target_layer,
            target_dim=target_dim,
        )

        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        return embedding

    def encode_image(
        self,
        images: Image.Image | list[Image.Image] | torch.Tensor,
        target_layer: int | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor:
        """Encode images into embeddings."""
        if isinstance(images, Image.Image):
            images = [images]

        embedding = self.image_encoder(
            images=images,
            target_layer=target_layer,
            target_dim=target_dim,
        )

        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        return embedding

    def encode_audio(
        self,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor,
        sampling_rate: int = 16000,
        target_layer: int | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor:
        """Encode audio into embeddings."""
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            audio = [audio]

        embedding = self.audio_encoder(
            audio=audio,
            sampling_rate=sampling_rate,
            target_layer=target_layer,
            target_dim=target_dim,
        )

        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        return embedding

    def encode_multimodal(
        self,
        texts: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | torch.Tensor | None = None,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
        sampling_rate: int = 16000,
        target_layer: int | None = None,
        target_dim: int | None = None,
        encoder_target_layers: dict[str, int] | None = None,
    ) -> torch.Tensor:
        """
        Encode multimodal inputs into a fused embedding.

        Args:
            texts: Text input(s)
            images: Image input(s)
            audio: Audio input(s)
            sampling_rate: Audio sampling rate
            target_layer: Fusion layer to exit at (2DMSE)
            target_dim: Output dimension to truncate to (MRL)
            encoder_target_layers: Dict of encoder-specific layer exits
        """
        embeddings = {}
        enc_layers = encoder_target_layers or {}

        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            text_layer = enc_layers.get("text", None)
            embeddings["text"] = self.text_encoder(
                texts=texts,
                target_layer=text_layer,
            )

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            image_layer = enc_layers.get("image", None)
            embeddings["image"] = self.image_encoder(
                images=images,
                target_layer=image_layer,
            )

        if audio is not None:
            if isinstance(audio, np.ndarray) and audio.ndim == 1:
                audio = [audio]
            audio_layer = enc_layers.get("audio", None)
            embeddings["audio"] = self.audio_encoder(
                audio=audio,
                sampling_rate=sampling_rate,
                target_layer=audio_layer,
            )

        if not embeddings:
            raise ValueError("At least one modality must be provided")

        fused = self.fusion(
            embeddings=embeddings,
            target_layer=target_layer,
            target_dim=target_dim,
            normalize=self.normalize,
        )
        return fused

    def forward(
        self,
        texts: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | torch.Tensor | None = None,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
        modality: str | None = None,
        target_layer: int | None = None,
        target_dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass - encode inputs into embeddings."""
        provided = sum([texts is not None, images is not None, audio is not None])

        if provided == 0:
            raise ValueError("At least one input must be provided")

        if provided == 1 or modality is not None:
            if texts is not None or modality == "text":
                return self.encode_text(texts, target_layer, target_dim)
            elif images is not None or modality == "image":
                return self.encode_image(images, target_layer, target_dim)
            elif audio is not None or modality == "audio":
                return self.encode_audio(
                    audio, target_layer=target_layer, target_dim=target_dim, **kwargs
                )

        return self.encode_multimodal(
            texts=texts,
            images=images,
            audio=audio,
            target_layer=target_layer,
            target_dim=target_dim,
            **kwargs,
        )

    def get_layer_info(self) -> dict[str, int]:
        """Get number of layers for each encoder."""
        return self.encoder_layers.copy()

    def _adaptive_embedding(
        self,
        encoder: nn.Module,
        encoder_kwargs: dict[str, Any],
        exit_layers: list[int],
        confidence_threshold: float,
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        """Select the first candidate layer that satisfies the confidence gate."""
        final_embedding, layer_embeddings = encoder(
            **encoder_kwargs,
            return_all_layers=True,
        )
        selected_layer = encoder.num_layers
        selected_embedding = final_embedding
        selected_confidence = 1.0
        for layer in exit_layers:
            if layer > len(layer_embeddings):
                continue
            candidate = layer_embeddings[layer - 1]
            confidence = self._estimate_confidence(candidate)
            if confidence >= confidence_threshold:
                selected_layer = layer
                selected_embedding = candidate
                selected_confidence = confidence
                break
        return selected_embedding, {
            "layer": selected_layer,
            "confidence": selected_confidence,
        }

    def encode_adaptive(
        self,
        texts: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | torch.Tensor | None = None,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
        confidence_threshold: float = 0.8,
        return_exit_info: bool = False,
        **kwargs,
    ):
        """
        Encode with confidence-based adaptive layer exit.

        Each modality independently decides when to exit based on confidence.
        Uses single forward pass with all layer outputs for efficiency.

        Args:
            texts: Text input(s)
            images: Image input(s)
            audio: Audio input(s)
            confidence_threshold: Minimum confidence to exit early (0-1)
            return_exit_info: If True, return (embedding, exit_info_dict)

        Returns:
            embedding or (embedding, exit_info) with layer/confidence per modality
        """
        embeddings = {}
        exit_info = {}

        text_exit_layers = [6, 11, 16, min(22, self.text_encoder.num_layers)]
        image_exit_layers = [6, 13, 20, min(27, self.image_encoder.num_layers)]

        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]

            embedding, info = self._adaptive_embedding(
                self.text_encoder,
                {"texts": texts},
                text_exit_layers,
                confidence_threshold,
            )
            embeddings["text"] = embedding
            exit_info["text"] = info

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            embedding, info = self._adaptive_embedding(
                self.image_encoder,
                {"images": images},
                image_exit_layers,
                confidence_threshold,
            )
            embeddings["image"] = embedding
            exit_info["image"] = info

        if audio is not None:
            emb = self.audio_encoder(audio=audio, **kwargs)
            embeddings["audio"] = emb
            exit_info["audio"] = {
                "layer": self.audio_encoder.num_layers,
                "confidence": 1.0,
            }

        # Combine embeddings
        if len(embeddings) == 0:
            raise ValueError("At least one modality must be provided")
        elif len(embeddings) == 1:
            result = next(iter(embeddings.values()))
        else:
            result = self.fusion(embeddings=embeddings, normalize=self.normalize)

        if self.normalize and len(embeddings) == 1:
            result = functional.normalize(result, p=2, dim=-1)

        if return_exit_info:
            return result, exit_info
        return result

    def encode_with_fixed_layers(
        self,
        texts: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | torch.Tensor | None = None,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
        text_layer: int | None = None,
        image_layer: int | None = None,
        audio_layer: int | None = None,
        target_dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode with explicit layer selection per modality.

        This is the most efficient method when you know which layers to use,
        as it only computes up to the specified layer (true early exit).

        Args:
            text_layer: Exit layer for text encoder (None = full)
            image_layer: Exit layer for image encoder (None = full)
            audio_layer: Exit layer for audio encoder (None = full)
            target_dim: Output dimension truncation (MRL)
        """
        return self.encode_multimodal(
            texts=texts,
            images=images,
            audio=audio,
            encoder_target_layers={
                "text": text_layer,
                "image": image_layer,
                "audio": audio_layer,
            },
            target_dim=target_dim,
            **kwargs,
        )

    def _estimate_confidence(self, embedding: torch.Tensor) -> float:
        """
        Estimate confidence based on embedding properties.

        For production, replace with trained ConfidenceEstimator
        that learns to predict when early layers are sufficient.
        """
        with torch.no_grad():
            # Variance-based: lower variance = more confident/stable
            variance = embedding.var(dim=-1).mean().item()
            conf_from_var = 1.0 / (1.0 + variance)

            # Norm-based: well-normalized = good embedding
            norm = embedding.norm(dim=-1).mean().item()
            conf_from_norm = min(1.0, norm)

            return (conf_from_var + conf_from_norm) / 2

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "MultimodalEmbedder":
        """Load a pretrained model."""
        model_path = Path(path)
        config_path = model_path / "config.json"
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)

        model = cls(**config, **kwargs)

        weights_paths = [
            model_path / "model.pt",
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin",
        ]

        for weights_path in weights_paths:
            if weights_path.exists():
                if weights_path.suffix == ".safetensors":
                    from safetensors.torch import load_file  # noqa: PLC0415

                    state_dict = load_file(str(weights_path))
                else:
                    state_dict = torch.load(
                        weights_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                model.load_state_dict(state_dict)
                break
        else:
            candidates = ", ".join(item.name for item in weights_paths)
            raise FileNotFoundError(
                f"No model weights found in {model_path}; expected one of: {candidates}"
            )

        return model

    def save_pretrained(self, path: str) -> None:
        """Save model to path."""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)

        config = {
            "output_dim": self.output_dim,
            "text_encoder_name": self.text_encoder.model_name,
            "text_encoder_revision": self.text_encoder_revision,
            "image_encoder_name": self.image_encoder.model_name,
            "image_encoder_revision": self.image_encoder_revision,
            "audio_encoder_name": self.audio_encoder.model_name,
            "audio_encoder_revision": self.audio_encoder_revision,
            "fusion_type": self.fusion_type,
            "num_fusion_layers": self.num_fusion_layers,
            "enable_layer_outputs": self.enable_layer_outputs,
        }

        with (model_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        torch.save(self.state_dict(), model_path / "model.pt")


# Legacy alias for backward compatibility
DiffusionMultimodalEmbedder = MultimodalEmbedder  # Deprecated
