"""Model implementations for multimodal embeddings."""

from .adaptive_exit import (
    AdaptiveLayerExitEncoder,
    AdaptiveMultimodalExit,
    ConfidenceEstimator,
    ConfidenceTrainingLoss,
)
from .audio_encoder import AudioEncoder
from .embedder import MultimodalEmbedder
from .fusion import ModalityFusion
from .image_encoder import ImageEncoder
from .model_factory import (
    MODEL_PRESETS,
    create_mobile_model,
    create_model_from_config,
    get_model_info,
    list_presets,
    print_model_summary,
)
from .text_encoder import TextEncoder

__all__ = [
    "MODEL_PRESETS",
    "AdaptiveLayerExitEncoder",
    "AdaptiveMultimodalExit",
    "AudioEncoder",
    "ConfidenceEstimator",
    "ConfidenceTrainingLoss",
    "ImageEncoder",
    "ModalityFusion",
    "MultimodalEmbedder",
    "TextEncoder",
    # Factory functions
    "create_mobile_model",
    "create_model_from_config",
    "get_model_info",
    "list_presets",
    "print_model_summary",
]
