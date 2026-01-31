"""Configuration merger for vLLM Semantic Router."""

import copy
from typing import Dict, Any, List

from cli.models import UserConfig, PluginType
from cli.defaults import load_embedded_defaults
from cli.utils import getLogger

log = getLogger(__name__)


def translate_keyword_signals(keywords: list) -> list:
    """
    Translate keyword signals to router format.

    Args:
        keywords: List of KeywordSignal objects

    Returns:
        list: Router keyword rules
    """
    rules = []
    for signal in keywords:
        rules.append(
            {
                "name": signal.name,
                "operator": signal.operator,
                "keywords": signal.keywords,
                "case_sensitive": signal.case_sensitive,
            }
        )
    return rules


def translate_embedding_signals(embeddings: list) -> list:
    """
    Translate embedding signals to router format.

    Args:
        embeddings: List of EmbeddingSignal objects

    Returns:
        list: Router embedding rules
    """
    rules = []
    for signal in embeddings:
        rules.append(
            {
                "name": signal.name,
                "threshold": signal.threshold,
                "candidates": signal.candidates,
                "aggregation_method": signal.aggregation_method,
            }
        )
    return rules


def translate_fact_check_signals(fact_checks: list) -> list:
    """
    Translate fact check signals to router format.

    Args:
        fact_checks: List of FactCheck objects

    Returns:
        list: Router fact check rules
    """
    rules = []
    for signal in fact_checks:
        rule = {
            "name": signal.name,
        }
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_user_feedback_signals(user_feedbacks: list) -> list:
    """
    Translate user feedback signals to router format.

    Args:
        user_feedbacks: List of UserFeedback objects

    Returns:
        list: Router user feedback rules
    """
    rules = []
    for signal in user_feedbacks:
        rule = {
            "name": signal.name,
        }
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_preference_signals(preferences: list) -> list:
    """
    Translate preference signals to router format.

    Args:
        preferences: List of Preference objects

    Returns:
        list: Router preference rules
    """
    rules = []
    for signal in preferences:
        rule = {
            "name": signal.name,
        }
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_language_signals(languages: list) -> list:
    """
    Translate language signals to router format.

    Args:
        languages: List of Language objects

    Returns:
        list: Router language rules
    """
    rules = []
    for signal in languages:
        rule = {
            "name": signal.name,
        }
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_latency_signals(latencies: list) -> list:
    """
    Translate latency signals to router format.

    Args:
        latencies: List of Latency objects

    Returns:
        list: Router latency rules
    """
    rules = []
    for signal in latencies:
        rule = {
            "name": signal.name,
        }
        # At least one of tpot_percentile or ttft_percentile should be set
        if signal.tpot_percentile is not None and signal.tpot_percentile > 0:
            rule["tpot_percentile"] = signal.tpot_percentile
        if signal.ttft_percentile is not None and signal.ttft_percentile > 0:
            rule["ttft_percentile"] = signal.ttft_percentile

        # Validate that at least one is set
        if "tpot_percentile" not in rule and "ttft_percentile" not in rule:
            log.warn(
                f"Latency signal '{signal.name}' has neither tpot_percentile nor ttft_percentile set, skipping"
            )
            continue

        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_context_signals(context_rules: list) -> list:
    """
    Translate context signals to router format.

    Args:
        context_rules: List of ContextRule objects

    Returns:
        list: Router context rules
    """
    rules = []
    for signal in context_rules:
        rule = {
            "name": signal.name,
            "min_tokens": signal.min_tokens,
            "max_tokens": signal.max_tokens,
        }
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_complexity_signals(complexity_rules: list) -> list:
    """
    Translate complexity signals to router format.

    Args:
        complexity_rules: List of ComplexityRule objects

    Returns:
        list: Router complexity rules
    """
    rules = []
    for signal in complexity_rules:
        rule = {
            "name": signal.name,
            "threshold": signal.threshold,
            "hard": {"candidates": signal.hard.candidates},
            "easy": {"candidates": signal.easy.candidates},
        }
        if signal.description:
            rule["description"] = signal.description
        if signal.composer:
            rule["composer"] = {
                "operator": signal.composer.operator,
                "conditions": [
                    {"type": c.type, "name": c.name} for c in signal.composer.conditions
                ],
            }
        rules.append(rule)
    return rules


def translate_external_models(external_models: list) -> list:
    """
    Translate external models to router format.

    Args:
        external_models: List of ExternalModel objects

    Returns:
        list: Router external model configurations
    """
    models = []
    for model in external_models:
        # Parse endpoint
        parts = model.endpoint.split(":")
        address = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8000

        config = {
            "llm_provider": model.provider,
            "model_role": model.role,
            "llm_endpoint": {
                "address": address,
                "port": port,
            },
            "llm_model_name": model.model_name,
            "llm_timeout_seconds": model.timeout_seconds,
            "parser_type": model.parser_type,
        }

        # Add access_key if provided
        if model.access_key:
            config["access_key"] = model.access_key

        models.append(config)
    return models


def translate_domains_to_categories(domains: list) -> list:
    """
    Translate domains to router categories format.

    Args:
        domains: List of Domain objects

    Returns:
        list: Router categories
    """
    categories = []
    for domain in domains:
        categories.append(
            {
                "name": domain.name,
                "description": domain.description,
                "mmlu_categories": domain.mmlu_categories,
            }
        )
    return categories


def extract_categories_from_decisions(decisions: list) -> list:
    """
    Auto-generate categories from decisions that reference domains.

    Args:
        decisions: List of Decision objects

    Returns:
        list: Auto-generated categories
    """
    categories = {}

    for decision in decisions:
        for condition in decision.rules.conditions:
            if condition.type == "domain":
                if condition.name not in categories:
                    categories[condition.name] = {
                        "name": condition.name,
                        "description": f"Auto-generated from decision: {decision.name}",
                        "mmlu_categories": [condition.name],
                    }

    return list(categories.values())


def translate_providers_to_router_format(providers) -> Dict[str, Any]:
    """
    Translate providers configuration to router format.

    Args:
        providers: Providers object

    Returns:
        dict: Router format with vllm_endpoints and model_config
    """
    # Extract endpoints from all models
    vllm_endpoints = []
    model_config = {}

    for model in providers.models:
        # Add model config
        model_config[model.name] = {
            "reasoning_family": model.reasoning_family,
            "access_key": model.access_key,
        }

        # Add api_format if provided
        if model.api_format:
            model_config[model.name]["api_format"] = model.api_format

        # Add pricing if provided
        if model.pricing:
            model_config[model.name]["pricing"] = {
                "currency": model.pricing.currency or "USD",
                "prompt_per_1m": model.pricing.prompt_per_1m or 0.0,
                "completion_per_1m": model.pricing.completion_per_1m or 0.0,
            }

        # Add endpoints for this model
        for endpoint in model.endpoints:
            # Parse endpoint: can be "host", "host:port", or "host/path" or "host:port/path"
            endpoint_str = endpoint.endpoint
            path = ""

            # Extract path if present (e.g., "host/path" or "host:port/path")
            if "/" in endpoint_str:
                # Split by first "/" to separate host[:port] from path
                parts = endpoint_str.split("/", 1)
                endpoint_str = parts[0]  # host or host:port
                path = "/" + parts[1]  # /path

            # Parse host and port
            if ":" in endpoint_str:
                host, port = endpoint_str.split(":", 1)
                port = int(port)
            else:
                host = endpoint_str
                # Use default port based on protocol
                port = 443 if endpoint.protocol == "https" else 80

            endpoint_config = {
                "name": f"{model.name}_{endpoint.name}",
                "address": host,
                "port": port,
                "weight": endpoint.weight,
                "protocol": endpoint.protocol,
                "model": model.name,
            }

            # Add path if present
            if path:
                endpoint_config["path"] = path

            vllm_endpoints.append(endpoint_config)

    # Convert ReasoningFamily Pydantic models to dicts for YAML serialization
    reasoning_families_dict = {}
    if providers.reasoning_families:
        for family_name, family_config in providers.reasoning_families.items():
            # Convert Pydantic model to dict if needed
            if hasattr(family_config, "model_dump"):
                reasoning_families_dict[family_name] = family_config.model_dump()
            elif hasattr(family_config, "dict"):
                reasoning_families_dict[family_name] = family_config.dict()
            elif isinstance(family_config, dict):
                reasoning_families_dict[family_name] = family_config
            else:
                # Fallback: convert to dict manually
                reasoning_families_dict[family_name] = {
                    "type": family_config.type,
                    "parameter": family_config.parameter,
                }

    return {
        "vllm_endpoints": vllm_endpoints,
        "model_config": model_config,
        "default_model": providers.default_model,
        "reasoning_families": reasoning_families_dict,
        "default_reasoning_effort": providers.default_reasoning_effort,
    }


def merge_configs(user_config: UserConfig, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with embedded defaults.

    Args:
        user_config: Parsed user configuration
        defaults: Embedded default configuration

    Returns:
        dict: Merged router configuration
    """
    log.info("Merging user configuration with defaults...")

    # Start with defaults
    merged = copy.deepcopy(defaults)

    # Translate signals
    if user_config.signals:
        if user_config.signals.keywords:
            merged["keyword_rules"] = translate_keyword_signals(
                user_config.signals.keywords
            )
            log.info(f"  Added {len(user_config.signals.keywords)} keyword signals")

        if user_config.signals.embeddings:
            merged["embedding_rules"] = translate_embedding_signals(
                user_config.signals.embeddings
            )
            log.info(f"  Added {len(user_config.signals.embeddings)} embedding signals")

        if user_config.signals.fact_check and len(user_config.signals.fact_check) > 0:
            merged["fact_check_rules"] = translate_fact_check_signals(
                user_config.signals.fact_check
            )
            log.info(
                f"  Added {len(user_config.signals.fact_check)} fact check signals"
            )

        if (
            user_config.signals.user_feedbacks
            and len(user_config.signals.user_feedbacks) > 0
        ):
            merged["user_feedback_rules"] = translate_user_feedback_signals(
                user_config.signals.user_feedbacks
            )
            log.info(
                f"  Added {len(user_config.signals.user_feedbacks)} user feedback signals"
            )

        if user_config.signals.preferences and len(user_config.signals.preferences) > 0:
            merged["preference_rules"] = translate_preference_signals(
                user_config.signals.preferences
            )
            log.info(
                f"  Added {len(user_config.signals.preferences)} preference signals"
            )

        if user_config.signals.language and len(user_config.signals.language) > 0:
            merged["language_rules"] = translate_language_signals(
                user_config.signals.language
            )
            log.info(f"  Added {len(user_config.signals.language)} language signals")

        if user_config.signals.latency and len(user_config.signals.latency) > 0:
            merged["latency_rules"] = translate_latency_signals(
                user_config.signals.latency
            )
            log.info(f"  Added {len(user_config.signals.latency)} latency signals")

        if user_config.signals.context and len(user_config.signals.context) > 0:
            merged["context_rules"] = translate_context_signals(
                user_config.signals.context
            )
            log.info(f"  Added {len(user_config.signals.context)} context signals")

        if user_config.signals.complexity and len(user_config.signals.complexity) > 0:
            merged["complexity_rules"] = translate_complexity_signals(
                user_config.signals.complexity
            )
            log.info(
                f"  Added {len(user_config.signals.complexity)} complexity signals"
            )

        # Translate domains to categories
        if user_config.signals.domains:
            merged["categories"] = translate_domains_to_categories(
                user_config.signals.domains
            )
            log.info(f"  Added {len(user_config.signals.domains)} domains")
        else:
            # Auto-generate categories from decisions
            merged["categories"] = extract_categories_from_decisions(
                user_config.decisions
            )
            log.info(
                f"  Auto-generated {len(merged['categories'])} categories from decisions"
            )
    else:
        # No signals, auto-generate categories
        merged["categories"] = extract_categories_from_decisions(user_config.decisions)
        log.info(
            f"  Auto-generated {len(merged['categories'])} categories from decisions"
        )

    # Add decisions (convert to dict)
    # Use mode='python' to ensure proper enum handling
    decisions_list = []
    for decision in user_config.decisions:
        decision_dict = decision.model_dump(mode="python")
        # Post-process plugins to ensure PluginType enums are converted to strings
        if "plugins" in decision_dict and decision_dict["plugins"]:
            for plugin in decision_dict["plugins"]:
                if "type" in plugin:
                    # Convert PluginType enum to string value
                    if isinstance(plugin["type"], PluginType):
                        plugin["type"] = plugin["type"].value
                    elif hasattr(plugin["type"], "value"):
                        plugin["type"] = plugin["type"].value
        decisions_list.append(decision_dict)
    merged["decisions"] = decisions_list
    log.info(f"  Added {len(user_config.decisions)} decisions")

    # Translate providers
    provider_config = translate_providers_to_router_format(user_config.providers)
    merged.update(provider_config)
    log.info(f"  Added {len(user_config.providers.models)} models")
    log.info(f"  Added {len(provider_config['vllm_endpoints'])} endpoints")

    log.info("✓ Configuration merged successfully")

    return merged
