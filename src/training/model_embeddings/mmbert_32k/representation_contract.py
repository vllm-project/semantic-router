"""Versioned artifact metadata for mmBERT task representations.

Absent metadata is legacy. Callers must preserve their existing legacy
semantics instead of inferring a new normalization policy from a model name.
"""

from __future__ import annotations

from typing import Any

CONTRACT_FIELD = "representation_contract"
NORMALIZATION_FINAL = "final_norm"
NORMALIZATION_NONE = "none"


def vela_representation_contract(task: str) -> dict[str, Any]:
    """Return a fresh, complete Vela v1 task contract."""
    common = {
        "version": 1,
        "intermediate_normalization": NORMALIZATION_FINAL,
        "final_normalization": NORMALIZATION_FINAL,
    }
    if task == "embedding":
        return {
            **common,
            "intermediate_normalization": NORMALIZATION_NONE,
            "pooling": "attention_mask_mean",
            "pooling_accumulation_dtype": "float32",
            "truncate_before_l2_normalize": True,
        }
    if task == "reranker":
        return {**common, "pooling": "cls", "head_dtype": "float32"}
    raise ValueError(f"Unsupported representation task: {task}")


def read_representation_contract(config: Any, task: str) -> dict[str, Any] | None:
    """Read and strictly validate metadata; None explicitly means legacy."""
    expected = vela_representation_contract(task)
    if isinstance(config, dict):
        if CONTRACT_FIELD not in config:
            return None
        value = config[CONTRACT_FIELD]
    else:
        if not hasattr(config, CONTRACT_FIELD):
            return None
        value = getattr(config, CONTRACT_FIELD)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Incomplete or unknown representation_contract fields")
    for key, wanted in expected.items():
        if task == "embedding" and key == "intermediate_normalization":
            if value[key] not in (NORMALIZATION_NONE, NORMALIZATION_FINAL):
                raise ValueError("Unsupported intermediate_normalization")
            continue
        if type(value[key]) is not type(wanted) or value[key] != wanted:
            raise ValueError(
                f"Unsupported representation_contract {key}: {value[key]!r}"
            )
    return dict(value)


def set_vela_representation_contract(config: Any, task: str) -> dict[str, Any]:
    """Explicitly opt a newly trained artifact into the Vela contract."""
    value = vela_representation_contract(task)
    if isinstance(config, dict):
        config[CONTRACT_FIELD] = value
    else:
        setattr(config, CONTRACT_FIELD, value)
    return dict(value)
