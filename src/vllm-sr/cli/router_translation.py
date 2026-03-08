"""Router config translation helpers for the CLI merger."""

from typing import Any
from urllib.parse import urlparse

from cli.models import PluginType

DEFAULT_HTTP_PORT = 80
DEFAULT_HTTPS_PORT = 443
DEFAULT_ENDPOINT_PORT = 8000


def _condition_to_dict(condition) -> dict[str, Any]:
    """Serialize recursive condition node to router rule dict."""
    node: dict[str, Any] = {}
    ctype = getattr(condition, "type", None)
    cname = getattr(condition, "name", None)
    cop = getattr(condition, "operator", None)
    cchildren = getattr(condition, "conditions", None)

    if ctype is not None:
        node["type"] = ctype
    if cname is not None:
        node["name"] = cname
    if cop is not None:
        node["operator"] = cop
    if cchildren:
        node["conditions"] = [_condition_to_dict(child) for child in cchildren]
    return node


def _iter_condition_nodes(conditions):
    """Depth-first traversal over recursive condition trees."""
    if not conditions:
        return
    for condition in conditions:
        yield condition
        if getattr(condition, "conditions", None):
            yield from _iter_condition_nodes(condition.conditions)


def translate_keyword_signals(keywords: list) -> list:
    """Translate keyword signals to router format."""
    return [
        {
            "name": signal.name,
            "operator": signal.operator,
            "keywords": signal.keywords,
            "case_sensitive": signal.case_sensitive,
        }
        for signal in keywords
    ]


def translate_embedding_signals(embeddings: list) -> list:
    """Translate embedding signals to router format."""
    return [
        {
            "name": signal.name,
            "threshold": signal.threshold,
            "candidates": signal.candidates,
            "aggregation_method": signal.aggregation_method,
        }
        for signal in embeddings
    ]


def translate_fact_check_signals(fact_checks: list) -> list:
    """Translate fact check signals to router format."""
    return [
        _translate_named_signal(signal.name, signal.description)
        for signal in fact_checks
    ]


def translate_user_feedback_signals(user_feedbacks: list) -> list:
    """Translate user feedback signals to router format."""
    return [
        _translate_named_signal(signal.name, signal.description)
        for signal in user_feedbacks
    ]


def translate_preference_signals(preferences: list) -> list:
    """Translate preference signals to router format."""
    rules = []
    for signal in preferences:
        rule = _translate_named_signal(signal.name, signal.description)
        if signal.threshold is not None:
            rule["threshold"] = signal.threshold
        if signal.examples:
            rule["examples"] = signal.examples
        rules.append(rule)
    return rules


def translate_language_signals(languages: list) -> list:
    """Translate language signals to router format."""
    return [
        _translate_named_signal(signal.name, signal.description) for signal in languages
    ]


def translate_context_signals(context_rules: list) -> list:
    """Translate context signals to router format."""
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


def translate_listeners(listeners: list) -> list:
    """Translate listeners to router format."""
    translated = []
    for listener in listeners:
        config = {
            "name": listener.name,
            "address": listener.address,
            "port": listener.port,
        }
        if listener.timeout:
            config["timeout"] = listener.timeout
        translated.append(config)
    return translated


def translate_complexity_signals(complexity_rules: list) -> list:
    """Translate complexity signals to router format."""
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
            rule["composer"] = _condition_to_dict(signal.composer)
        rules.append(rule)
    return rules


def translate_jailbreak_signals(jailbreak_rules: list) -> list:
    """Translate jailbreak signals to router format."""
    rules = []
    for signal in jailbreak_rules:
        rule = {
            "name": signal.name,
            "threshold": signal.threshold,
        }
        if signal.method:
            rule["method"] = signal.method
        if signal.include_history:
            rule["include_history"] = signal.include_history
        if signal.jailbreak_patterns:
            rule["jailbreak_patterns"] = signal.jailbreak_patterns
        if signal.benign_patterns:
            rule["benign_patterns"] = signal.benign_patterns
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_pii_signals(pii_rules: list) -> list:
    """Translate PII signals to router format."""
    rules = []
    for signal in pii_rules:
        rule = {
            "name": signal.name,
            "threshold": signal.threshold,
        }
        if signal.pii_types_allowed:
            rule["pii_types_allowed"] = signal.pii_types_allowed
        if signal.include_history:
            rule["include_history"] = signal.include_history
        if signal.description:
            rule["description"] = signal.description
        rules.append(rule)
    return rules


def translate_external_models(external_models: list) -> list:
    """Translate external models to router format."""
    models = []
    for model in external_models:
        address, port, protocol = _parse_endpoint(model.endpoint)
        config = {
            "llm_provider": model.provider,
            "model_role": model.role,
            "llm_endpoint": {
                "address": address,
                "port": port,
                "protocol": protocol,
            },
            "llm_model_name": model.model_name,
            "llm_timeout_seconds": model.timeout_seconds,
            "parser_type": model.parser_type,
        }
        if model.access_key:
            config["access_key"] = model.access_key
        models.append(config)
    return models


def translate_decisions(decisions: list) -> list:
    """Translate decisions to a clean router-ready dict representation."""
    translated = []
    for decision in decisions:
        decision_dict = decision.model_dump(mode="python", exclude_none=True)
        for plugin in decision_dict.get("plugins", []):
            _normalize_plugin_type(plugin)
        translated.append(decision_dict)
    return translated


def extract_categories_from_decisions(decisions: list) -> list:
    """Auto-generate categories from decisions that reference domains."""
    categories = {}
    for decision in decisions:
        for condition in _iter_condition_nodes(decision.rules.conditions):
            if condition.type == "domain" and condition.name not in categories:
                categories[condition.name] = {
                    "name": condition.name,
                    "description": f"Auto-generated from decision: {decision.name}",
                    "mmlu_categories": [condition.name],
                }
    return list(categories.values())


def translate_providers_to_router_format(providers) -> dict[str, Any]:
    """Translate providers configuration to router runtime fields."""
    vllm_endpoints: list[dict[str, Any]] = []
    model_config: dict[str, dict[str, Any]] = {}

    for model in providers.models:
        model_entry, preferred_endpoints = _translate_model(model)
        for endpoint in preferred_endpoints:
            vllm_endpoints.append(endpoint)
        if preferred_endpoints:
            model_entry["preferred_endpoints"] = [
                endpoint["name"] for endpoint in preferred_endpoints
            ]
        model_config[model.name] = model_entry

    external_models = []
    if providers.external_models:
        external_models = translate_external_models(providers.external_models)

    return {
        "vllm_endpoints": vllm_endpoints,
        "model_config": model_config,
        "default_model": providers.default_model,
        "reasoning_families": _translate_reasoning_families(
            providers.reasoning_families
        ),
        "default_reasoning_effort": providers.default_reasoning_effort,
        "external_models": external_models,
    }


def _translate_named_signal(name: str, description: str | None) -> dict[str, Any]:
    rule = {"name": name}
    if description:
        rule["description"] = description
    return rule


def _parse_endpoint(endpoint_str: str) -> tuple[str, int, str]:
    """Parse an endpoint string into (address, port, protocol)."""
    if "://" in endpoint_str:
        parsed = urlparse(endpoint_str)
        address = parsed.hostname or ""
        port = parsed.port or (
            DEFAULT_HTTPS_PORT if parsed.scheme == "https" else DEFAULT_HTTP_PORT
        )
        protocol = parsed.scheme
        return address, port, protocol

    if ":" in endpoint_str:
        address, port_str = endpoint_str.rsplit(":", 1)
        port = int(port_str)
        protocol = "https" if port == DEFAULT_HTTPS_PORT else "http"
        return address, port, protocol

    return endpoint_str, DEFAULT_ENDPOINT_PORT, "http"


def _normalize_plugin_type(plugin: dict[str, Any]) -> None:
    plugin_type = plugin.get("type")
    if isinstance(plugin_type, PluginType) or hasattr(plugin_type, "value"):
        plugin["type"] = plugin_type.value


def _translate_model(model) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_entry = _build_model_entry(model)
    preferred_endpoints = [
        _translate_provider_endpoint(model.name, endpoint)
        for endpoint in model.endpoints
    ]
    return model_entry, preferred_endpoints


def _build_model_entry(model) -> dict[str, Any]:
    model_entry: dict[str, Any] = {}
    if model.reasoning_family:
        model_entry["reasoning_family"] = model.reasoning_family
    if model.access_key:
        model_entry["access_key"] = model.access_key
    if model.param_size:
        model_entry["param_size"] = model.param_size
    if model.api_format:
        model_entry["api_format"] = model.api_format
    if model.description:
        model_entry["description"] = model.description
    if model.capabilities:
        model_entry["capabilities"] = model.capabilities
    if model.quality_score is not None:
        model_entry["quality_score"] = model.quality_score
    if model.pricing:
        model_entry["pricing"] = {
            "currency": model.pricing.currency or "USD",
            "prompt_per_1m": model.pricing.prompt_per_1m or 0.0,
            "completion_per_1m": model.pricing.completion_per_1m or 0.0,
        }
    return model_entry


def _translate_provider_endpoint(model_name: str, endpoint) -> dict[str, Any]:
    host, port, path = _split_provider_endpoint(endpoint.endpoint, endpoint.protocol)
    endpoint_config = {
        "name": f"{model_name}_{endpoint.name}",
        "address": host,
        "port": port,
        "weight": endpoint.weight,
        "protocol": endpoint.protocol,
        "model": model_name,
    }
    if path:
        endpoint_config["path"] = path
    return endpoint_config


def _split_provider_endpoint(endpoint_str: str, protocol: str) -> tuple[str, int, str]:
    path = ""
    host_port = endpoint_str
    if "/" in endpoint_str:
        host_port, path_suffix = endpoint_str.split("/", 1)
        path = f"/{path_suffix}"

    if ":" in host_port:
        host, port_str = host_port.split(":", 1)
        return host, int(port_str), path

    return (
        host_port,
        DEFAULT_HTTPS_PORT if protocol == "https" else DEFAULT_HTTP_PORT,
        path,
    )


def _translate_reasoning_families(reasoning_families) -> dict[str, Any]:
    translated = {}
    if not reasoning_families:
        return translated

    for family_name, family_config in reasoning_families.items():
        if hasattr(family_config, "model_dump"):
            translated[family_name] = family_config.model_dump()
            continue
        if hasattr(family_config, "dict"):
            translated[family_name] = family_config.dict()
            continue
        if isinstance(family_config, dict):
            translated[family_name] = family_config
            continue
        translated[family_name] = {
            "type": family_config.type,
            "parameter": family_config.parameter,
        }
    return translated
