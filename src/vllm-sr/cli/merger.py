"""Configuration merger for vLLM Semantic Router."""

import copy
from typing import Any

from cli.authoring_runtime_compile import build_first_slice_runtime_overlay
from cli.compat_blocks import dump_typed_compat_block, get_typed_compat_blocks
from cli.models import UserConfig
from cli.router_translation import (
    extract_categories_from_decisions,
    translate_complexity_signals,
    translate_context_signals,
    translate_embedding_signals,
    translate_fact_check_signals,
    translate_jailbreak_signals,
    translate_language_signals,
    translate_modality_signals,
    translate_pii_signals,
    translate_preference_signals,
    translate_role_binding_signals,
    translate_user_feedback_signals,
)
from cli.utils import getLogger

log = getLogger(__name__)

_SIGNAL_TRANSLATIONS = (
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
    ("modality", "modality_rules", translate_modality_signals, "modality signals"),
    (
        "role_bindings",
        "role_bindings",
        translate_role_binding_signals,
        "role binding signals",
    ),
    ("jailbreak", "jailbreak", translate_jailbreak_signals, "jailbreak signals"),
    ("pii", "pii", translate_pii_signals, "PII signals"),
)
_NAMED_TYPED_COMPAT_MERGES = (
    ("api", "api", "  Added api configuration"),
    ("authz", "authz", "  Added authz configuration"),
    ("bert_model", "bert_model", "  Added bert_model configuration"),
    ("classifier", "classifier", "  Added classifier configuration"),
    (
        "feedback_detector",
        "feedback_detector",
        "  Added feedback_detector configuration",
    ),
    (
        "hallucination_mitigation",
        "hallucination_mitigation",
        "  Added hallucination_mitigation configuration",
    ),
    (
        "image_gen_backends",
        "image_gen_backends",
        "  Added image_gen_backends configuration",
    ),
    ("looper", "looper", "  Added looper configuration"),
    (
        "modality_detector",
        "modality_detector",
        "  Added modality_detector configuration",
    ),
    ("model_selection", "model_selection", "  Added model_selection configuration"),
    ("observability", "observability", "  Added observability configuration"),
    ("prompt_guard", "prompt_guard", "  Added prompt_guard configuration"),
    (
        "provider_profiles",
        "provider_profiles",
        "  Added provider_profiles configuration",
    ),
    ("ratelimit", "ratelimit", "  Added ratelimit configuration"),
    ("semantic_cache", "semantic_cache", "  Added semantic_cache configuration"),
    ("response_api", "response_api", "  Added response_api configuration"),
    ("router_replay", "router_replay", "  Added router_replay configuration"),
    ("tools", "tools", "  Added tools configuration"),
    ("vector_store", "vector_store", "  Added vector_store configuration"),
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
    _merge_first_slice_runtime(merged, user_config)
    _merge_signals(merged, user_config)
    _merge_optional_blocks(merged, user_config)
    _merge_typed_compat_blocks(merged, user_config)

    log.info("Configuration merged successfully")
    return merged


def _merge_first_slice_runtime(merged: dict[str, Any], user_config: UserConfig) -> None:
    runtime_overlay = build_first_slice_runtime_overlay(user_config)
    merged.update(copy.deepcopy(runtime_overlay))

    if runtime_overlay.get("listeners"):
        log.info(f"  Added {len(runtime_overlay['listeners'])} listeners")

    if runtime_overlay.get("keyword_rules"):
        log.info(f"  Added {len(runtime_overlay['keyword_rules'])} keyword signals")

    if runtime_overlay.get("decisions") is not None:
        log.info(f"  Added {len(runtime_overlay['decisions'])} decisions")

    model_config = runtime_overlay.get("model_config")
    if model_config is not None:
        log.info(f"  Added {len(model_config)} models")

    vllm_endpoints = runtime_overlay.get("vllm_endpoints")
    if vllm_endpoints is not None:
        log.info(f"  Added {len(vllm_endpoints)} endpoints")

    external_models = runtime_overlay.get("external_models")
    if external_models:
        log.info(f"  Added {len(external_models)} external_models")


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

    for attr_name, target_key, log_message in _NAMED_TYPED_COMPAT_MERGES:
        if not _merge_named_typed_compat_block(
            merged, compat_blocks, attr_name, target_key
        ):
            continue
        log.info(log_message)

    if compat_blocks.router_options is not None:
        merged.update(dump_typed_compat_block(compat_blocks.router_options))
        log.info("  Added router option compatibility keys")

    if compat_blocks.runtime_top_level is not None:
        merged.update(dump_typed_compat_block(compat_blocks.runtime_top_level))
        log.info("  Added runtime top-level compatibility keys")


def _merge_named_typed_compat_block(
    merged: dict[str, Any],
    compat_blocks: Any,
    attr_name: str,
    target_key: str,
) -> bool:
    compat_block = getattr(compat_blocks, attr_name)
    if compat_block is None:
        return False

    merged[target_key] = dump_typed_compat_block(compat_block)
    return True


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
