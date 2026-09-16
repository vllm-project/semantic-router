"""Thin modality-pair wrappers used by the contrastive trainer."""

from __future__ import annotations

from typing import Any

from torch import nn


class ContrastiveTrainingWrapper(nn.Module):
    """Expose image and text encoders through a DataParallel-safe forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.image_processor = model.image_encoder.processor
        self.tokenizer = model.text_encoder.tokenizer
        self.max_length = model.text_encoder.max_length

    def forward(self, pixel_values: Any, input_ids: Any, attention_mask: Any) -> Any:
        image_embedding = self.model.image_encoder(pixel_values=pixel_values)
        text_embedding = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return image_embedding, text_embedding

    def preprocess(
        self,
        images: list[Any],
        captions: list[str],
        device: Any,
    ) -> tuple[Any, Any, Any]:
        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt",
        )[
            "pixel_values"
        ].to(device, non_blocking=True)
        encoded = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            pixel_values,
            encoded["input_ids"].to(device, non_blocking=True),
            encoded["attention_mask"].to(device, non_blocking=True),
        )


class AudioContrastiveWrapper(nn.Module):
    """Expose audio and text encoders through a DataParallel-safe forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = model.text_encoder.tokenizer
        self.max_length = model.text_encoder.max_length

    def forward(self, input_features: Any, input_ids: Any, attention_mask: Any) -> Any:
        audio_embedding = self.model.audio_encoder(input_features=input_features)
        text_embedding = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return audio_embedding, text_embedding
