"""Model construction, checkpoint loading, and staged unfreezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import MultimodalEmbedder

FROZEN_ALIGNMENT_STAGE = 1
PARTIAL_UNFREEZE_STAGE = 2
FULL_UNFREEZE_STAGE = 4
AUDIO_ONLY_STAGE = 5
AUDIO_TWO_TEXT_LAYERS_STAGE = 6
AUDIO_THREE_TEXT_LAYERS_STAGE = 7
AUDIO_STAGES = (
    AUDIO_ONLY_STAGE,
    AUDIO_TWO_TEXT_LAYERS_STAGE,
    AUDIO_THREE_TEXT_LAYERS_STAGE,
)


def create_model(model_config: dict[str, Any]) -> MultimodalEmbedder:
    """Build the compact model directly from the checked configuration."""
    return MultimodalEmbedder(
        text_encoder_name=model_config["text_encoder_name"],
        text_encoder_revision=model_config.get("text_encoder_revision"),
        image_encoder_name=model_config["image_encoder_name"],
        image_encoder_revision=model_config.get("image_encoder_revision"),
        audio_encoder_name=model_config["audio_encoder_name"],
        audio_encoder_revision=model_config.get("audio_encoder_revision"),
        output_dim=int(model_config["output_dim"]),
        fusion_type=model_config.get("fusion_type", "transformer"),
        num_fusion_layers=int(model_config.get("num_fusion_layers", 2)),
        enable_layer_outputs=bool(model_config.get("enable_layer_outputs", True)),
        max_text_length=int(model_config.get("max_text_length", 128)),
    )


def load_weights(model: torch.nn.Module, checkpoint: str | None) -> Path | None:
    """Load the first supported weight file from a checkpoint path."""
    if not checkpoint:
        return None
    root = Path(checkpoint).expanduser()
    candidates = (
        [root]
        if root.is_file()
        else [
            root / "model.safetensors",
            root / "model.pt",
            root / "pytorch_model.bin",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            from safetensors.torch import (  # noqa: PLC0415 - optional dependency
                load_file,
            )

            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return path
    raise FileNotFoundError(f"No supported model weights found under {root}")


def _set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _matches_layer(name: str, indices: tuple[int, ...]) -> bool:
    return any(
        marker in name
        for index in indices
        for marker in (f"layer.{index}.", f"layers.{index}.", f"blocks.{index}.")
    )


def _unfreeze_named_layers(
    module: torch.nn.Module,
    indices: tuple[int, ...],
    extra_markers: tuple[str, ...],
) -> None:
    for name, parameter in module.named_parameters():
        if _matches_layer(name, indices) or any(
            marker in name for marker in extra_markers
        ):
            parameter.requires_grad = True


def _unfreeze_last_text_layers(model: MultimodalEmbedder, count: int) -> None:
    encoder = getattr(getattr(model.text_encoder, "model", None), "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise RuntimeError("Cannot locate MiniLM encoder layers for staged unfreezing")
    for layer in layers[max(0, len(layers) - count) :]:
        _set_trainable(layer, True)


def apply_stage(model: MultimodalEmbedder, stage: int) -> str:
    """Apply the published image-text and audio-text unfreezing stages."""
    _set_trainable(model, True)
    if stage == FROZEN_ALIGNMENT_STAGE:
        for encoder in (model.text_encoder, model.image_encoder, model.audio_encoder):
            for name, parameter in encoder.named_parameters():
                if "projection" not in name and "proj" not in name:
                    parameter.requires_grad = False
        return "fusion and encoder projections"

    if stage == PARTIAL_UNFREEZE_STAGE:
        for encoder in (model.text_encoder, model.image_encoder, model.audio_encoder):
            _set_trainable(encoder, False)
        _unfreeze_named_layers(
            model.text_encoder, (3, 4, 5), ("projection", "proj", "pooler", "final")
        )
        _unfreeze_named_layers(
            model.image_encoder,
            (9, 10, 11),
            ("projection", "proj", "head", "post_layernorm", "final"),
        )
        _unfreeze_named_layers(model.audio_encoder, (2, 3), ("projection", "proj"))
        return "fusion, projections, and upper encoder layers"

    if stage == FULL_UNFREEZE_STAGE:
        return "all model parameters"

    if stage in AUDIO_STAGES:
        _set_trainable(model.text_encoder, False)
        _set_trainable(model.image_encoder, False)
        _set_trainable(model.audio_encoder, True)
        if stage in (AUDIO_TWO_TEXT_LAYERS_STAGE, AUDIO_THREE_TEXT_LAYERS_STAGE):
            text_layer_count = 2 if stage == AUDIO_TWO_TEXT_LAYERS_STAGE else 3
            _unfreeze_last_text_layers(model, text_layer_count)
        return (
            "audio encoder"
            if stage == AUDIO_ONLY_STAGE
            else f"audio plus stage-{stage} text layers"
        )

    raise ValueError(f"Unsupported training stage: {stage}")
