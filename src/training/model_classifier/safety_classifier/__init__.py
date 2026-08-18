"""Reproducible mmBERT safety-classifier reconstruction utilities."""

from importlib import import_module
from typing import Any

from .taxonomy import (
    LEVEL1_LABEL_TO_ID,
    LEVEL1_LABELS,
    LEVEL2_LABEL_TO_ID,
    LEVEL2_LABELS,
    TAXONOMY_VERSION,
    UnknownCategoryError,
    map_aegis_categories,
    map_synth_category,
)

_DATA_EXPORTS = frozenset(
    {
        "DEFAULT_LEVEL1_PER_LABEL",
        "DEFAULT_LEVEL2_PER_LABEL",
        "DEFAULT_SEED",
        "DataPreparationError",
        "DatasetBuild",
        "InsufficientClassSamplesError",
        "PreparedSample",
        "SchemaError",
        "build_level1_dataset",
        "build_level2_dataset",
        "materialize_dataset",
        "normalize_prompt",
        "prepare_materialized_data",
        "prompt_fingerprint",
    }
)

__all__ = [
    "DEFAULT_LEVEL1_PER_LABEL",
    "DEFAULT_LEVEL2_PER_LABEL",
    "DEFAULT_SEED",
    "LEVEL1_LABELS",
    "LEVEL1_LABEL_TO_ID",
    "LEVEL2_LABELS",
    "LEVEL2_LABEL_TO_ID",
    "TAXONOMY_VERSION",
    "DataPreparationError",
    "DatasetBuild",
    "InsufficientClassSamplesError",
    "PreparedSample",
    "SchemaError",
    "UnknownCategoryError",
    "build_level1_dataset",
    "build_level2_dataset",
    "map_aegis_categories",
    "map_synth_category",
    "materialize_dataset",
    "normalize_prompt",
    "prepare_materialized_data",
    "prompt_fingerprint",
]


def __getattr__(name: str) -> Any:
    """Load data helpers lazily so ``python -m ...data`` stays warning-free."""
    if name in _DATA_EXPORTS:
        return getattr(import_module(f"{__name__}.data"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
