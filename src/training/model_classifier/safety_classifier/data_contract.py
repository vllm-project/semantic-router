"""Stable constants and errors for safety-classifier data preparation."""

from __future__ import annotations

SPLITS = ("train", "validation", "test")
SPLIT_PRECEDENCE = ("test", "validation", "train")
SPLIT_RANK = {split: rank for rank, split in enumerate(SPLIT_PRECEDENCE)}
NORMALIZATION_VERSION = "nfkc-whitespace-casefold-v1"
DEFAULT_SEED = 42
DEFAULT_LEVEL1_PER_LABEL = 10_000
DEFAULT_LEVEL2_PER_LABEL = 2_000
SHA256_HEX_LENGTH = 64

AEGIS_SCHEMA = frozenset(
    {
        "id",
        "reconstruction_id_if_redacted",
        "prompt",
        "response",
        "prompt_label",
        "response_label",
        "violated_categories",
        "prompt_label_source",
        "response_label_source",
    }
)
SYNTH_SCHEMA = frozenset({"text", "category", "label"})

AUDIT_KEYS = (
    "aegis_rows_seen",
    "synthetic_rows_seen",
    "schema_valid_rows",
    "empty_prompt_rows",
    "redacted_prompt_rows",
    "level2_safe_rows_skipped",
    "level2_no_mapped_target_rows",
    "level2_multilabel_rows",
    "level2_multitarget_rows",
    "candidate_rows",
    "unique_fingerprint_groups",
    "label_conflict_groups",
    "label_conflict_rows",
    "lower_precedence_rows_dropped",
    "same_split_duplicate_rows_dropped",
    "deduplicated_rows",
    "unique_rows_after_dedup",
    "train_rows_available",
    "train_rows_selected",
    "train_rows_downsampled",
    "train_rows_oversampled",
    "emitted_rows",
)


class DataPreparationError(ValueError):
    """Base error for deterministic data preparation failures."""


class SchemaError(DataPreparationError):
    """Raised when an input row does not match its pinned source schema."""


class InsufficientClassSamplesError(DataPreparationError):
    """Raised when a class cannot satisfy its sampling contract."""


class DataContractError(DataPreparationError):
    """Raised when the reconstruction contract disagrees with the builder."""
