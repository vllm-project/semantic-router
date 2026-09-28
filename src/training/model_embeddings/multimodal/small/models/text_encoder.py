"""Text encoder supporting 2D Matryoshka (layer exit + dimension truncation)."""

import logging
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    Text encoder with 2D Matryoshka support.

    Default model: mmbert-embed-32k-2d-matryoshka
    - 32K context length
    - 1800+ languages
    - Built-in layer exit (2DMSE)
    - Built-in dimension truncation (MRL)
    """

    def __init__(
        self,
        model_name_or_path: str = "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
        revision: str | None = None,
        output_dim: int = 768,
        pooling_mode: str = "mean",
        normalize: bool = True,
        max_length: int = 32768,
        enable_layer_outputs: bool = True,
    ):
        super().__init__()

        self.model_name = model_name_or_path
        self.revision = revision
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.normalize = normalize
        self.max_length = max_length
        self.enable_layer_outputs = enable_layer_outputs

        # Check if using mmbert (has built-in 2D matryoshka)
        self.is_mmbert = "mmbert" in model_name_or_path.lower()

        # Load base model
        logger.info(f"Loading text encoder: {model_name_or_path}")
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=True,
        )

        # For models requiring Flash Attention (like mmBERT), load directly to GPU
        device_map = "auto" if torch.cuda.is_available() and self.is_mmbert else None
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            revision=revision,
            trust_remote_code=True,
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=True,
        )

        # Get hidden size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        # Projection to output dimension (only if different)
        self.layer_projections: nn.ModuleList | None = None
        if self.hidden_size != output_dim:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            # Per-layer projections for 2DMSE when not using mmbert
            if enable_layer_outputs and not self.is_mmbert:
                self.layer_projections = nn.ModuleList(
                    [
                        nn.Linear(self.hidden_size, output_dim)
                        for _ in range(self.num_layers)
                    ]
                )
        else:
            self.projection = nn.Identity()

        logger.info(
            f"Text encoder: {self.num_layers} layers, {self.hidden_size}d -> {output_dim}d"
        )
        if self.is_mmbert:
            logger.info("Using mmBERT with built-in 2D Matryoshka support")

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool hidden states to get sentence embedding."""
        if self.pooling_mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask
                return hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return hidden_states.mean(dim=1)
        elif self.pooling_mode == "cls":
            return hidden_states[:, 0]
        elif self.pooling_mode == "last":
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                return hidden_states[
                    torch.arange(batch_size, device=hidden_states.device), seq_lens
                ]
            return hidden_states[:, -1]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def _apply_final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer norm if available (for mmBERT layer exit)."""
        if hasattr(self.encoder, "final_norm"):
            return self.encoder.final_norm(hidden_states)
        elif hasattr(self.encoder, "norm"):
            return self.encoder.norm(hidden_states)
        return hidden_states

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        texts: list[str] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Tokenize raw text and move tensors to the encoder device."""
        if input_ids is not None or texts is None:
            return input_ids, attention_mask
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        parameter = next(self.parameters(), None)
        device = (
            parameter.device
            if parameter is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @staticmethod
    def _hidden_states(outputs: Any) -> tuple[torch.Tensor, ...] | None:
        """Return encoder-layer states without the input embedding state."""
        hidden_states = getattr(outputs, "hidden_states", None)
        return hidden_states[1:] if hidden_states else None

    def _project_layer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Normalize, pool, and project one encoder layer."""
        normalized = self._apply_final_norm(hidden_states)
        pooled = self._pool(normalized, attention_mask)
        if layer_idx is not None and self.layer_projections is not None:
            return self.layer_projections[layer_idx](pooled)
        return self.projection(pooled)

    def _select_embedding(
        self,
        outputs: Any,
        all_hidden_states: tuple[torch.Tensor, ...] | None,
        attention_mask: torch.Tensor | None,
        target_layer: int | None,
    ) -> torch.Tensor:
        """Select either an intermediate 2DMSE layer or the final layer."""
        if target_layer is not None and all_hidden_states is not None:
            layer_idx = min(target_layer, len(all_hidden_states) - 1)
            return self._project_layer(
                all_hidden_states[layer_idx],
                attention_mask,
                layer_idx,
            )
        return self.projection(self._pool(outputs.last_hidden_state, attention_mask))

    def _all_layer_embeddings(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Project every encoder layer for confidence-based exit training."""
        embeddings = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            embedding = self._project_layer(
                hidden_states,
                attention_mask,
                layer_idx,
            )
            if self.normalize:
                embedding = functional.normalize(embedding, p=2, dim=-1)
            embeddings.append(embedding)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        texts: list[str] | None = None,
        target_layer: int | None = None,
        target_dim: int | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for text encoding.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            texts: Raw text strings (tokenized if input_ids not provided)
            target_layer: Exit at this layer (2DMSE)
            target_dim: Truncate to this dimension (MRL)
            return_all_layers: Return embeddings from all layers
        """
        input_ids, attention_mask = self._prepare_inputs(
            input_ids,
            attention_mask,
            texts,
        )

        # Forward through encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        all_hidden_states = self._hidden_states(outputs)
        embedding = self._select_embedding(
            outputs,
            all_hidden_states,
            attention_mask,
            target_layer,
        )

        # Dimension truncation (MRL)
        if target_dim is not None:
            embedding = embedding[:, :target_dim]

        # Normalize
        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        # Return all layer embeddings if requested
        if return_all_layers and all_hidden_states is not None:
            return embedding, self._all_layer_embeddings(
                all_hidden_states,
                attention_mask,
            )

        return embedding

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        target_layer: int | None = None,
        target_dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode texts into embeddings."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.forward(
                    texts=batch_texts,
                    target_layer=target_layer,
                    target_dim=target_dim,
                    **kwargs,
                )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# Legacy alias for backward compatibility
DiffusionTextEncoder = TextEncoder  # Deprecated
