"""Fail-closed ModernBERT YaRN configuration helpers.

Transformers 5 constructs shared rotary frequencies from per-attention-type
``config.rope_parameters``. Historical configuration objects use
``rope_scaling``. A reproducible run validates the persisted settings and the
instantiated frequencies, using the SDPA or eager attention path.
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
    """Build the official Transformers YaRN configuration."""
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
            f"config-driven rotary parameters; choose one of {sorted(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}"
        )
    observed_native_length = getattr(config, "max_position_embeddings", None)
    if observed_native_length != original_max_position_embeddings:
        raise ValueError(
            "base config context does not match the declared original context: "
            f"{observed_native_length!r} != {original_max_position_embeddings}"
        )

    config.max_position_embeddings = target_max_position_embeddings
    scaling = build_yarn_rope_scaling(
        original_max_position_embeddings=original_max_position_embeddings,
        target_max_position_embeddings=target_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if isinstance(getattr(config, "rope_parameters", None), dict):
        config.rope_parameters = {
            layer_type: {**parameters, **scaling}
            for layer_type, parameters in config.rope_parameters.items()
        }
    else:
        config.rope_scaling = scaling
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
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict):
        layer_types = set(getattr(config, "layer_types", ()))
        if not layer_types or not layer_types <= parameters.keys():
            raise RuntimeError("model config omitted attention-type RoPE parameters")
        valid = all(
            all(
                parameters[layer_type].get(key) == value
                for key, value in expected.items()
            )
            for layer_type in layer_types
        )
    else:
        valid = getattr(config, "rope_scaling", None) == expected
    if not valid:
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
        if isinstance(getattr(rotary, "rope_type", None), dict):
            try:
                _verify_shared_rotary(rotary, model.config)
            except (RuntimeError, AssertionError, KeyError) as error:
                failures.append(f"{name}.rotary_emb: {error}")
            rotary_count += len(model.config.layer_types) - 1
            continue
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


def _verify_shared_rotary(rotary: Any, config: Any) -> None:
    """Validate the shared Transformers 5 rotary module for every layer type."""
    import torch  # noqa: PLC0415 - retain import-light config validation
    from transformers.modeling_rope_utils import (  # noqa: PLC0415
        ROPE_INIT_FUNCTIONS,
    )

    for layer_type in set(config.layer_types):
        if rotary.rope_type.get(layer_type) != "yarn":
            raise RuntimeError(f"{layer_type} does not use config-driven YaRN")
        if (
            rotary.config.rope_parameters[layer_type]
            != config.rope_parameters[layer_type]
        ):
            raise RuntimeError(f"{layer_type} rotary configuration changed")
        actual = getattr(rotary, f"{layer_type}_inv_freq", None)
        if actual is None:
            raise RuntimeError(f"{layer_type} has no instantiated YaRN frequencies")
        expected, scale = ROPE_INIT_FUNCTIONS["yarn"](
            config, device=actual.device, layer_type=layer_type
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
        if getattr(rotary, f"{layer_type}_attention_scaling", None) != scale:
            raise RuntimeError(f"{layer_type} YaRN attention scaling changed")
