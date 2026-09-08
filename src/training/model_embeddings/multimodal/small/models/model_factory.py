"""Factory and parameter summaries for the compact multimodal model."""

from __future__ import annotations

from typing import Any

import yaml
from torch import nn

from .embedder import MultimodalEmbedder


def create_mobile_model(
    output_dim: int = 384,
    freeze_encoders: bool = True,
    **kwargs: Any,
) -> MultimodalEmbedder:
    """Create the MiniLM/SigLIP/Whisper-tiny compact model."""
    model = MultimodalEmbedder(
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        image_encoder_name="google/siglip-base-patch16-512",
        audio_encoder_name="openai/whisper-tiny",
        output_dim=output_dim,
        fusion_type="transformer",
        num_fusion_layers=2,
        enable_layer_outputs=True,
        **kwargs,
    )
    if freeze_encoders:
        for encoder in (model.text_encoder, model.image_encoder, model.audio_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False
    return model


def create_model_from_config(path: str) -> tuple[MultimodalEmbedder, dict[str, Any]]:
    """Create the model from the `model` section of a YAML configuration."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = config.get("model", {})
    model = MultimodalEmbedder(
        text_encoder_name=model_config["text_encoder_name"],
        text_encoder_revision=model_config.get("text_encoder_revision"),
        image_encoder_name=model_config["image_encoder_name"],
        image_encoder_revision=model_config.get("image_encoder_revision"),
        audio_encoder_name=model_config["audio_encoder_name"],
        audio_encoder_revision=model_config.get("audio_encoder_revision"),
        output_dim=int(model_config.get("output_dim", 384)),
        max_text_length=int(model_config.get("max_text_length", 128)),
        fusion_type=model_config.get("fusion_type", "transformer"),
        num_fusion_layers=int(model_config.get("num_fusion_layers", 2)),
        enable_layer_outputs=bool(model_config.get("enable_layer_outputs", True)),
    )
    return model, config


def get_model_info(model: nn.Module) -> dict[str, float | int]:
    """Return total/trainable parameter counts and estimated FP32 size."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    size_mb = total * 4 / (1024 * 1024)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "size_mb": size_mb,
        "size_gb": size_mb / 1024,
    }


def print_model_summary(model: nn.Module, name: str = "Model") -> None:
    info = get_model_info(model)
    print(f"{name}: {info['total_params']:,} parameters")
    print(f"Trainable: {info['trainable_params']:,}")
    print(f"Estimated FP32 size: {info['size_mb']:.1f} MiB")


MODEL_PRESETS = {
    "small": {
        "factory": create_mobile_model,
        "description": "Compact 384-dimensional text/image/audio embedder",
    }
}


def list_presets() -> None:
    """Print the model presets owned by this package."""
    for name, info in MODEL_PRESETS.items():
        print(f"{name}: {info['description']}")
