"""Configuration merger for vLLM Semantic Router."""

import copy
from typing import Dict, Any, List

from cli.models import UserConfig, Domain
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

        # Add endpoints for this model
        for endpoint in model.endpoints:
            # Parse endpoint (host:port or just host)
            if ":" in endpoint.endpoint:
                host, port = endpoint.endpoint.split(":", 1)
                port = int(port)
            else:
                host = endpoint.endpoint
                # Use default port based on protocol
                port = 443 if endpoint.protocol == "https" else 80

            vllm_endpoints.append(
                {
                    "name": f"{model.name}_{endpoint.name}",
                    "address": host,
                    "port": port,
                    "weight": endpoint.weight,
                    "protocol": endpoint.protocol,
                    "model": model.name,
                }
            )

    return {
        "vllm_endpoints": vllm_endpoints,
        "model_config": model_config,
        "default_model": providers.default_model,
        "reasoning_families": providers.reasoning_families,
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
    merged["decisions"] = [decision.model_dump() for decision in user_config.decisions]
    log.info(f"  Added {len(user_config.decisions)} decisions")

    # Translate providers
    provider_config = translate_providers_to_router_format(user_config.providers)
    merged.update(provider_config)
    log.info(f"  Added {len(user_config.providers.models)} models")
    log.info(f"  Added {len(provider_config['vllm_endpoints'])} endpoints")

    log.info("✓ Configuration merged successfully")

    return merged
