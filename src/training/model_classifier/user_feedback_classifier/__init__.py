"""
ModernBERT Dissatisfaction Classifier Pipeline.

A complete fine-tuning pipeline for training a binary classifier
to detect user dissatisfaction from follow-up or re-send prompts.

Usage:
    from modernbert_dissat_pipeline import DissatisfactionClassifier, classify_dissatisfaction

    # Load classifier
    classifier = DissatisfactionClassifier("modernbert_dissat_resend_classifier")

    # Classify
    result = classifier.classify(
        original_query="What is X?",
        system_answer="X is...",
        user_followup="Can you explain more simply?"
    )

    print(result.label)  # "DISSAT" or "SAT"
    print(result.confidence)  # 0.0 to 1.0
"""

__version__ = "1.0.0"

from .inference import (
    DissatisfactionClassifier,
    ClassificationResult,
    classify_dissatisfaction,
    classify_dissatisfaction_detailed,
)

from .configs.config import (
    PipelineConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
    get_large_model_config,
)

__all__ = [
    # Inference
    "DissatisfactionClassifier",
    "ClassificationResult",
    "classify_dissatisfaction",
    "classify_dissatisfaction_detailed",
    # Config
    "PipelineConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "get_default_config",
    "get_large_model_config",
]
