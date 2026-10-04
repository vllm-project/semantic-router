"""Native backbone architectures, keyed by the Transformers ``model_type`` of their config."""

from __future__ import annotations

from typing import Any

from torch import nn

from .qwen3 import Qwen3Backbone
from .qwen3_5 import Qwen3_5Backbone

ARCHITECTURES: dict[str, type[nn.Module]] = {
    Qwen3Backbone.model_type: Qwen3Backbone,
    Qwen3_5Backbone.model_type: Qwen3_5Backbone,
}


def build(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type not in ARCHITECTURES:
        raise ValueError(f"the native engine has no {model_type!r} backbone")
    return ARCHITECTURES[model_type](config)
