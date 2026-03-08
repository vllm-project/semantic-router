"""Configuration merger for vLLM Semantic Router."""

import copy
from typing import Any

from cli.compat_blocks import get_typed_compat_blocks
from cli.models import UserConfig
from cli.router_translation import (
    extract_categories_from_decisions,
    translate_complexity_signals,
    translate_context_signals,
    translate_decisions,
    translate_embedding_signals,
    translate_fact_check_signals,
    translate_jailbreak_signals,
    translate_keyword_signals,
    translate_language_signals,
    translate_listeners,
    translate_pii_signals,
    translate_preference_signals,
    translate_providers_to_router_format,
    translate_user_feedback_signals,
)
from cli.user_config_top_level import LEGACY_RUNTIME_TOP_LEVEL_COMPATIBILITY_KEYS
from cli.utils import getLogger

log = getLogger(__name__)

_SIGNAL_TRANSLATIONS = (
    ("keywords", "keyword_rules", translate_keyword_signals, "keyword signals"),
    ("embeddings", "embedding_rules", translate_embedding_signals, "embedding signals"),
    (
        "fact_check",
        "fact_check_rules",
        translate_fact_check_signals,
        "fact check signals",
    ),
    (
        "user_feedbacks",
        "user_feedback_rules",
        translate_user_feedback_signals,
        "user feedback signals",
    ),
    (
        "preferences",
        "preference_rules",
        translate_preference_signals,
        "preference signals",
    ),
    ("language", "language_rules", translate_language_signals, "language signals"),
    ("context", "context_rules", translate_context_signals, "context signals"),
    (
        "complexity",
        "complexity_rules",
        translate_complexity_signals,
        "complexity signals",
    ),
    ("jailbreak", "jailbreak", translate_jailbreak_signals, "jailbreak signals"),
    ("pii", "pii", translate_pii_signals, "PII signals"),
)


def merge_configs(user_config: UserConfig, defaults: dict[str, Any]) -> dict[str, Any]:
    """
    Merge user configuration with embedded defaults.

    Args:
        user_config: Parsed user configuration
        defaults: Embedded default configuration

    Returns:
        dict: Merged router configuration
    """
    log.info("Merging user configuration with defaults...")

    merged = copy.deepcopy(defaults)
    _merge_listeners(merged, user_config)
    _merge_signals(merged, user_config)
    _merge_decisions(merged, user_config)
    _merge_providers(merged, user_config)
    _merge_optional_blocks(merged, user_config)
    _merge_typed_compat_blocks(merged, user_config)
    _merge_extra_fields(merged, getattr(user_config, "model_extra", None) or {})

    log.info("Configuration merged successfully")
    return merged


def _merge_listeners(merged: dict[str, Any], user_config: UserConfig) -> None:
    if not user_config.listeners:
        return

    merged["listeners"] = translate_listeners(user_config.listeners)
    log.info(f"  Added {len(user_config.listeners)} listeners")


def _merge_signals(merged: dict[str, Any], user_config: UserConfig) -> None:
    signals = user_config.signals
    if not signals:
        merged["categories"] = extract_categories_from_decisions(user_config.decisions)
        log.info(
            f"  Auto-generated {len(merged['categories'])} categories from decisions"
        )
        return

    for attr, target_key, translator, label in _SIGNAL_TRANSLATIONS:
        values = getattr(signals, attr)
        if not values:
            continue
        merged[target_key] = translator(values)
        log.info(f"  Added {len(values)} {label}")

    if signals.domains:
        merged["categories"] = [
            {
                "name": domain.name,
                "description": domain.description,
                "mmlu_categories": domain.mmlu_categories,
            }
            for domain in signals.domains
        ]
        log.info(f"  Added {len(signals.domains)} domains")
        return

    merged["categories"] = extract_categories_from_decisions(user_config.decisions)
    log.info(f"  Auto-generated {len(merged['categories'])} categories from decisions")


def _merge_decisions(merged: dict[str, Any], user_config: UserConfig) -> None:
    merged["decisions"] = translate_decisions(user_config.decisions)
    log.info(f"  Added {len(user_config.decisions)} decisions")


def _merge_providers(merged: dict[str, Any], user_config: UserConfig) -> None:
    provider_config = translate_providers_to_router_format(user_config.providers)
    if not provider_config.get("external_models"):
        provider_config.pop("external_models", None)

    merged.update(provider_config)
    log.info(f"  Added {len(user_config.providers.models)} models")
    log.info(f"  Added {len(provider_config['vllm_endpoints'])} endpoints")
    if provider_config.get("external_models"):
        log.info(f"  Added {len(provider_config['external_models'])} external_models")


def _merge_optional_blocks(merged: dict[str, Any], user_config: UserConfig) -> None:
    if user_config.memory:
        merged["memory"] = user_config.memory.model_dump(exclude_none=True)
        log.info(f"  Added memory configuration (enabled={user_config.memory.enabled})")

    if not user_config.embedding_models:
        return

    embedding_config = user_config.embedding_models.model_dump(exclude_none=True)
    default_embedding_config = merged.get("embedding_models", {})
    if isinstance(default_embedding_config, dict):
        merged["embedding_models"] = _merge_embedding_models(
            default_embedding_config, embedding_config
        )
    else:
        merged["embedding_models"] = embedding_config
    log.info("  Added embedding_models configuration")


def _merge_typed_compat_blocks(merged: dict[str, Any], user_config: UserConfig) -> None:
    compat_blocks = get_typed_compat_blocks(user_config)

    if compat_blocks.looper is not None:
        merged["looper"] = compat_blocks.looper.model_dump(exclude_none=True)
        log.info("  Added looper configuration")

    if compat_blocks.observability is not None:
        merged["observability"] = compat_blocks.observability.model_dump(
            exclude_none=True
        )
        log.info("  Added observability configuration")

    if compat_blocks.prompt_guard is not None:
        merged["prompt_guard"] = compat_blocks.prompt_guard.model_dump(
            exclude_none=True
        )
        log.info("  Added prompt_guard configuration")

    if compat_blocks.router_replay is not None:
        merged["router_replay"] = compat_blocks.router_replay.model_dump(
            exclude_none=True
        )
        log.info("  Added router_replay configuration")

    if compat_blocks.tools is not None:
        merged["tools"] = compat_blocks.tools.model_dump(exclude_none=True)
        log.info("  Added tools configuration")


def _merge_embedding_models(
    default_embedding_config: dict[str, Any],
    embedding_config: dict[str, Any],
) -> dict[str, Any]:
    merged_embedding = copy.deepcopy(default_embedding_config)
    for key, value in embedding_config.items():
        if (
            key == "hnsw_config"
            and isinstance(value, dict)
            and isinstance(merged_embedding.get("hnsw_config"), dict)
        ):
            merged_embedding["hnsw_config"].update(value)
            continue
        merged_embedding[key] = value
    return merged_embedding


def _merge_extra_fields(merged: dict[str, Any], extra_fields: dict[str, Any]) -> None:
    for key, value in extra_fields.items():
        if key not in LEGACY_RUNTIME_TOP_LEVEL_COMPATIBILITY_KEYS:
            raise ValueError(
                f"unsupported passthrough top-level config key: {key}. "
                "Add explicit schema support before merging it."
            )
        merged[key] = copy.deepcopy(value)
        log.info(f"  Added passthrough top-level config: {key}")
