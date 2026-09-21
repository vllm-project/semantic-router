"""Audio encoder based on Whisper architecture."""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """
    Audio encoder using Whisper encoder architecture.

    Whisper's encoder is a sequential transformer stack, making it
    compatible with adaptive layer exit (2DMSE).
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-base",
        revision: str | None = None,
        output_dim: int = 1024,
        pooling_mode: str = "mean",
        normalize: bool = True,
        enable_layer_outputs: bool = True,  # For 2DMSE
    ):
        super().__init__()

        self.model_name = model_name_or_path
        self.revision = revision
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.normalize = normalize
        self.enable_layer_outputs = enable_layer_outputs

        # Load Whisper model (encoder only)
        logger.info(f"Loading audio encoder: {model_name_or_path}")
        self.config = WhisperConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
        )
        whisper = WhisperModel.from_pretrained(
            model_name_or_path,
            revision=revision,
        )
        self.encoder = whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name_or_path,
            revision=revision,
        )

        # Get dimensions
        self.hidden_size = self.config.d_model
        self.num_layers = self.config.encoder_layers

        # Projection to output dimension
        if self.hidden_size != output_dim:
            self.projection = nn.Linear(self.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()

        # Per-layer projections for 2DMSE
        self.layer_projections: nn.ModuleList | None = None
        self.layer_norms: nn.ModuleList | None = None
        if enable_layer_outputs:
            self.layer_projections = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, output_dim)
                    for _ in range(self.num_layers)
                ]
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
            )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to get audio embedding."""
        if self.pooling_mode == "mean":
            return hidden_states.mean(dim=1)
        elif self.pooling_mode == "first":
            return hidden_states[:, 0]
        elif self.pooling_mode == "last":
            return hidden_states[:, -1]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def preprocess(
        self,
        audio: np.ndarray | list[np.ndarray] | torch.Tensor,
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Preprocess audio for the encoder.

        Args:
            audio: Audio waveform(s) as numpy arrays or tensors
            sampling_rate: Audio sampling rate (should be 16kHz for Whisper)

        Returns:
            input_features: Mel spectrogram features
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            audio = [audio]

        # Use feature extractor to get mel spectrograms
        features = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        return features["input_features"]

    def _prepare_input_features(
        self,
        input_features: torch.Tensor | None,
        audio: np.ndarray | list[np.ndarray] | None,
        sampling_rate: int,
    ) -> torch.Tensor | None:
        """Preprocess raw audio and move it to the encoder device."""
        if input_features is not None or audio is None:
            return input_features
        prepared = self.preprocess(audio, sampling_rate)
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
        """Pool and project one intermediate encoder layer."""
        if self.layer_norms is not None and self.layer_projections is not None:
            normalized = self.layer_norms[layer_idx](hidden_states)
            return self.layer_projections[layer_idx](self._pool(normalized))
        return self.projection(self._pool(hidden_states))

    def _select_embedding(
        self,
        outputs: Any,
        all_hidden_states: tuple[torch.Tensor, ...] | None,
        target_layer: int | None,
    ) -> torch.Tensor:
        """Select either an intermediate 2DMSE layer or the final layer."""
        if target_layer is None or all_hidden_states is None:
            return self.projection(self._pool(outputs.last_hidden_state))
        layer_idx = min(target_layer, len(all_hidden_states) - 1)
        return self._project_layer(all_hidden_states[layer_idx], layer_idx)

    def _all_layer_embeddings(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Project every encoder layer for confidence-based exit training."""
        embeddings = []
        for layer_idx, layer_hidden_states in enumerate(all_hidden_states):
            embedding = self._project_layer(layer_hidden_states, layer_idx)
            if self.normalize:
                embedding = functional.normalize(embedding, p=2, dim=-1)
            embeddings.append(embedding)
        return embeddings

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        audio: np.ndarray | list[np.ndarray] | None = None,
        sampling_rate: int = 16000,
        target_layer: int | None = None,  # For 2DMSE
        target_dim: int | None = None,  # For MRL
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for audio encoding.

        Args:
            input_features: Preprocessed mel spectrogram features
            audio: Raw audio waveforms (will be preprocessed if input_features not provided)
            sampling_rate: Audio sampling rate
            target_layer: Exit at this layer (None = full model)
            target_dim: Truncate to this dimension (None = full dim)
            return_all_layers: Return embeddings from all layers

        Returns:
            embeddings: [batch_size, output_dim] or truncated
            all_layer_embeddings: (optional) List of [batch_size, output_dim]
        """
        input_features = self._prepare_input_features(
            input_features,
            audio,
            sampling_rate,
        )

        # Forward through encoder
        outputs = self.encoder(
            input_features=input_features,
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
        audio_list: list[np.ndarray],
        batch_size: int = 16,
        sampling_rate: int = 16000,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Encode a list of audio waveforms into embeddings."""
        all_embeddings = []

        for i in range(0, len(audio_list), batch_size):
            batch_audio = audio_list[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.forward(
                    audio=batch_audio,
                    sampling_rate=sampling_rate,
                    **kwargs,
                )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
