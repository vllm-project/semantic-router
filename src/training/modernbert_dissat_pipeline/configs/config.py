"""
Configuration for Feedback Detector Training Pipeline.

Compatible with: https://huggingface.co/llm-semantic-router/feedback-detector
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import os


# Labels matching feedback-detector
LABEL2ID: Dict[str, int] = {
    "SAT": 0,
    "NEED_CLARIFICATION": 1,
    "WRONG_ANSWER": 2,
    "WANT_DIFFERENT": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


@dataclass
class DataConfig:
    """Data processing configuration."""

    # HuggingFace dataset
    dataset_id: str = "llm-semantic-router/feedback-detector-dataset"

    # Local paths (fallback)
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Processing settings
    max_examples: Optional[int] = None  # None = use all
    train_split_ratio: float = 0.9
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""

    # Base model - mmBERT (multilingual ModernBERT)
    model_name: str = "jhu-clsp/mmBERT-base"
    # Alternative: "answerdotai/ModernBERT-base" for English-only

    # Sequence length (mmBERT supports up to 8192)
    max_seq_length: int = 512

    # Output
    output_dir: str = "models/mmbert_feedback_detector"

    # Labels (4-class)
    num_labels: int = 4
    label2id: Dict[str, int] = field(default_factory=lambda: LABEL2ID.copy())
    id2label: Dict[int, str] = field(default_factory=lambda: ID2LABEL.copy())


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Core hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5

    # Warmup and decay
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Class balancing
    use_class_weights: bool = True

    # Evaluation
    eval_steps: int = 500
    early_stopping_patience: int = 3

    # Reproducibility
    seed: int = 42


@dataclass
class LoRAConfig:
    """LoRA configuration."""

    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list = field(
        default_factory=lambda: [
            "attn.Wqkv",
            "attn.Wo",
            "mlp.Wi",
            "mlp.Wo",  # mmBERT/ModernBERT
        ]
    )
    merge_after_training: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self):
        """Create directories if needed."""
        os.makedirs(self.data.raw_data_dir, exist_ok=True)
        os.makedirs(self.data.processed_data_dir, exist_ok=True)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration (full fine-tuning)."""
    return PipelineConfig()


def get_lora_config() -> PipelineConfig:
    """Get configuration for LoRA training."""
    config = PipelineConfig()
    config.lora.enabled = True
    config.model.output_dir = "models/mmbert_feedback_detector_lora"
    return config
