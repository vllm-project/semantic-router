"""
Data Generation Package for Tool Verifier Training.

This package provides a modular, reusable architecture for generating
training data for jailbreak/prompt injection detection.

Structure:
- config: Centralized configuration (GeneratorConfig, LoaderConfig, Config)
- entities: Realistic entity pools (names, emails, paths, etc.)
- base: Base classes and data structures (SentinelSample, ToolCallSample)
- datasets: PyTorch Dataset classes for training
- generators/: Attack pattern generators (each with custom augmentation)
- loaders/: HuggingFace dataset loaders
- orchestrator: Combines generators and calls their augmentation methods

Each generator has customizable hooks:
- generate(): Core sample generation
- augment(): Text augmentation (synonyms, style, etc.)
- paraphrase(): LLM-based paraphrasing
- generate_adversarial(): Create hard negative examples

Usage:
    from data_generation import DataOrchestrator, Config

    # Basic usage
    config = Config(output_dir="data/generated")
    orchestrator = DataOrchestrator(config)
    train, dev = orchestrator.generate_stage2_data()

    # For training
    from data_generation import (
        SentinelDataset, load_sentinel_samples, load_sentinel_label_config,
        ToolCallVerifierDataset, load_tool_call_samples, load_verifier_label_config,
    )
"""

from .config import Config, GeneratorConfig, LoaderConfig
from .entities import Entities, get_entities
from .base import BaseGenerator, ToolCallSample, SentinelSample
from .orchestrator import DataOrchestrator
from .datasets import (
    # PyTorch Datasets
    SentinelDataset,
    ToolCallVerifierDataset,
    IntentAlignmentDataset,
    IntentAlignmentSample,
    # Label configs
    load_sentinel_label_config,
    load_verifier_label_config,
    SENTINEL_LABEL2ID,
    SENTINEL_ID2LABEL,
    VERIFIER_LABEL2ID,
    VERIFIER_ID2LABEL,
    VERIFIER_SEVERITY,
    # I/O utilities
    load_sentinel_samples,
    save_sentinel_samples,
    load_tool_call_samples,
    save_tool_call_samples,
    load_intent_samples,
    save_intent_samples,
)

__all__ = [
    # Config
    "Config",
    "GeneratorConfig",
    "LoaderConfig",
    # Entities
    "Entities",
    "get_entities",
    # Data classes
    "BaseGenerator",
    "ToolCallSample",
    "SentinelSample",
    "IntentAlignmentSample",
    # Orchestrator
    "DataOrchestrator",
    # PyTorch Datasets
    "SentinelDataset",
    "ToolCallVerifierDataset",
    "IntentAlignmentDataset",
    # Label configs
    "load_sentinel_label_config",
    "load_verifier_label_config",
    "SENTINEL_LABEL2ID",
    "SENTINEL_ID2LABEL",
    "VERIFIER_LABEL2ID",
    "VERIFIER_ID2LABEL",
    "VERIFIER_SEVERITY",
    # I/O utilities
    "load_sentinel_samples",
    "save_sentinel_samples",
    "load_tool_call_samples",
    "save_tool_call_samples",
    "load_intent_samples",
    "save_intent_samples",
]
