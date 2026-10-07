"""Native backbone architectures, keyed by the Transformers ``model_type`` of their config."""

from __future__ import annotations

from typing import Any

from torch import nn

from .bert import BertBackbone
from .clap import ClapAudioBackbone
from .modernbert import ModernBertBackbone
from .qwen3 import Qwen3Backbone
from .qwen3_5 import Qwen3_5Backbone
from .siglip import SiglipVisionBackbone
from .whisper import WhisperEncoderBackbone

ARCHITECTURES: dict[str, type[nn.Module]] = {
    BertBackbone.model_type: BertBackbone,
    ClapAudioBackbone.model_type: ClapAudioBackbone,
    ModernBertBackbone.model_type: ModernBertBackbone,
    Qwen3Backbone.model_type: Qwen3Backbone,
    Qwen3_5Backbone.model_type: Qwen3_5Backbone,
    SiglipVisionBackbone.model_type: SiglipVisionBackbone,
    WhisperEncoderBackbone.model_type: WhisperEncoderBackbone,
}


def build(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type not in ARCHITECTURES:
        raise ValueError(f"the native engine has no {model_type!r} backbone")
    return ARCHITECTURES[model_type](config)
