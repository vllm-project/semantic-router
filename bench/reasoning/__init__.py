"""
vLLM Semantic Router Benchmark Suite

A comprehensive benchmark suite for evaluating vLLM semantic router performance
against direct vLLM across multiple reasoning datasets.

Supported Datasets:
- MMLU-Pro: Academic knowledge across 57 subjects
- ARC: AI2 Reasoning Challenge for scientific reasoning
- GPQA: Graduate-level Google-proof Q&A
- TruthfulQA: Truthful response evaluation
- CommonsenseQA: Commonsense reasoning evaluation
- HellaSwag: Commonsense natural language inference

Key Features:
- Dataset-agnostic architecture with factory pattern
- Router vs direct vLLM comparison
- Multiple evaluation modes (NR, XC, NR_REASONING)
- Reasoning mode evaluation (Issue #42) - standard vs reasoning comparison
- Comprehensive plotting and analysis tools
- Research-ready CSV output
- Configurable token limits per dataset
"""

__version__ = "1.1.0"
__author__ = "vLLM Semantic Router Team"

from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import DatasetInfo, DatasetInterface, PromptFormatter, Question

# Reasoning mode evaluation (Issue #42)
from .reasoning_mode_eval import (
    ReasoningModeMetrics,
    ReasoningModeComparison,
)

# Make key classes available at package level
__all__ = [
    "DatasetInterface",
    "Question",
    "DatasetInfo",
    "PromptFormatter",
    "DatasetFactory",
    "list_available_datasets",
    # Reasoning mode evaluation (Issue #42)
    "ReasoningModeMetrics",
    "ReasoningModeComparison",
    "__version__",
]
