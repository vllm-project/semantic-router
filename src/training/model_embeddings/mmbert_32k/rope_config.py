"""Fail-closed ModernBERT YaRN configuration helpers.

The pinned Transformers 4.57.6 implementation constructs rotary frequencies
from ``config.rope_scaling``.  ModernBERT's Flash Attention 2 path uses a
different unpadded rotary implementation, so a reproducible YaRN run must use
the SDPA or eager attention path and validate the instantiated modules.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_ATTENTION_IMPLEMENTATIONS = frozenset({"eager", "sdpa"})


def build_yarn_rope_scaling(
    *,
    original_max_position_embeddings: int,
    target_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
) -> dict[str, float | int | str]:
    """Build the official Transformers 4.57.6 YaRN configuration."""
    if original_max_position_embeddings <= 0:
        raise ValueError("original_max_position_embeddings must be positive")
    if target_max_position_embeddings <= original_max_position_embeddings:
        raise ValueError("YaRN target context must exceed the original context")
    if beta_slow <= 0 or beta_fast <= beta_slow:
        raise ValueError("YaRN requires beta_fast > beta_slow > 0")

    factor = target_max_position_embeddings / original_max_position_embeddings
    return {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": original_max_position_embeddings,
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
    }


def configure_modernbert_yarn(
    config: Any,
    *,
    original_max_position_embeddings: int,
    target_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    attention_implementation: str,
) -> Any:
    """Mutate a loaded ModernBERT config before model construction."""
    if getattr(config, "model_type", None) != "modernbert":
        raise TypeError(
            "the mmBERT-32K recipe requires a ModernBERT config, got "
            f"{getattr(config, 'model_type', None)!r}"
        )
    if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            "ModernBERT YaRN requires an attention implementation that consumes "
            f"config.rope_scaling; choose one of {sorted(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}"
        )
    observed_native_length = getattr(config, "max_position_embeddings", None)
    if observed_native_length != original_max_position_embeddings:
        raise ValueError(
            "base config context does not match the declared original context: "
            f"{observed_native_length!r} != {original_max_position_embeddings}"
        )

    config.max_position_embeddings = target_max_position_embeddings
    config.rope_scaling = build_yarn_rope_scaling(
        original_max_position_embeddings=original_max_position_embeddings,
        target_max_position_embeddings=target_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    return config


def assert_yarn_config(
    config: Any,
    *,
    original_max_position_embeddings: int,
    target_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
) -> None:
    """Require the complete YaRN state that must survive ``save_pretrained``."""
    expected = build_yarn_rope_scaling(
        original_max_position_embeddings=original_max_position_embeddings,
        target_max_position_embeddings=target_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if (
        getattr(config, "max_position_embeddings", None)
        != target_max_position_embeddings
    ):
        raise RuntimeError("model config did not retain the target context length")
    if getattr(config, "rope_scaling", None) != expected:
        raise RuntimeError(
            "model config did not retain the exact Transformers YaRN configuration"
        )


def verify_loaded_modernbert_yarn(
    model: Any,
    *,
    original_max_position_embeddings: int,
    target_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    attention_implementation: str,
) -> int:
    """Verify every instantiated ModernBERT attention uses config-driven YaRN."""
    if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise RuntimeError(
            "unsupported attention implementation reached model validation"
        )

    assert_yarn_config(
        model.config,
        original_max_position_embeddings=original_max_position_embeddings,
        target_max_position_embeddings=target_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    observed_implementation = getattr(model.config, "_attn_implementation", None)
    if observed_implementation != attention_implementation:
        raise RuntimeError(
            "loaded model changed attention implementation: "
            f"{observed_implementation!r} != {attention_implementation!r}"
        )

    failures: list[str] = []
    rotary_count = 0
    for name, module in model.named_modules():
        if not hasattr(module, "rotary_emb"):
            continue
        rotary_count += 1
        rotary = module.rotary_emb
        rotary_class = rotary.__class__.__name__
        if "Unpadded" in rotary_class or getattr(rotary, "rope_type", None) != "yarn":
            failures.append(
                f"{name}.rotary_emb={rotary_class} does not use config-driven YaRN"
            )
            continue
        rope_init_fn = getattr(rotary, "rope_init_fn", None)
        if (
            not hasattr(rotary, "inv_freq")
            or getattr(rope_init_fn, "__name__", None) != "_compute_yarn_parameters"
        ):
            failures.append(
                f"{name}.rotary_emb did not instantiate official YaRN frequencies"
            )
            continue
        rotary_config = getattr(rotary, "config", None)
        if rotary_config is None:
            failures.append(f"{name}.rotary_emb has no retained config")
            continue
        try:
            assert_yarn_config(
                rotary_config,
                original_max_position_embeddings=original_max_position_embeddings,
                target_max_position_embeddings=target_max_position_embeddings,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
        except RuntimeError as error:
            failures.append(f"{name}.rotary_emb: {error}")

    expected_count = getattr(model.config, "num_hidden_layers", None)
    if rotary_count == 0:
        failures.append("no ModernBERT attention rotary modules were instantiated")
    elif expected_count is not None and rotary_count != expected_count:
        failures.append(
            f"validated {rotary_count} rotary modules, expected {expected_count} layers"
        )
    if failures:
        raise RuntimeError("ModernBERT YaRN validation failed: " + "; ".join(failures))
    return rotary_count
