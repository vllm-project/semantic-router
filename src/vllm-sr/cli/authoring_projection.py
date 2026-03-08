"""Projection helpers from CLI UserConfig into the TD001 canonical first slice."""

from __future__ import annotations

from typing import Any

from cli.models import UserConfig


def build_first_slice_authoring_config(user_config: UserConfig) -> dict[str, Any]:
    """Project the canonical TD001 first slice from a parsed UserConfig."""

    projected: dict[str, Any] = {"version": user_config.version}

    listeners = _project_listeners(user_config)
    if listeners:
        projected["listeners"] = listeners

    signals = _project_signals(user_config)
    if signals:
        projected["signals"] = signals

    projected["providers"] = _project_providers(user_config)

    decisions = _project_decisions(user_config)
    if decisions:
        projected["decisions"] = decisions

    return projected


def _project_listeners(user_config: UserConfig) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for listener in user_config.listeners:
        entry = {
            "name": listener.name,
            "address": listener.address,
            "port": listener.port,
        }
        if listener.timeout:
            entry["timeout"] = listener.timeout
        projected.append(entry)
    return projected


def _project_signals(user_config: UserConfig) -> dict[str, Any]:
    if not user_config.signals or not user_config.signals.keywords:
        return {}

    return {
        "keywords": [
            {
                "name": signal.name,
                "operator": signal.operator,
                "keywords": list(signal.keywords),
                "case_sensitive": signal.case_sensitive,
            }
            for signal in user_config.signals.keywords
        ]
    }


def _project_providers(user_config: UserConfig) -> dict[str, Any]:
    providers = user_config.providers
    projected: dict[str, Any] = {}

    if providers.models:
        projected["models"] = [_project_model(model) for model in providers.models]
    if providers.default_model:
        projected["default_model"] = providers.default_model
    if providers.reasoning_families:
        projected["reasoning_families"] = {
            name: {
                "type": family.type,
                "parameter": family.parameter,
            }
            for name, family in providers.reasoning_families.items()
        }
    if providers.default_reasoning_effort:
        projected["default_reasoning_effort"] = providers.default_reasoning_effort

    return projected


def _project_model(model: Any) -> dict[str, Any]:
    projected = {"name": model.name}

    if model.endpoints:
        projected["endpoints"] = [
            _project_endpoint(endpoint) for endpoint in model.endpoints
        ]
    if model.access_key:
        projected["access_key"] = model.access_key
    if model.reasoning_family:
        projected["reasoning_family"] = model.reasoning_family
    if model.pricing is not None:
        pricing = {
            "currency": model.pricing.currency,
            "prompt_per_1m": model.pricing.prompt_per_1m,
            "completion_per_1m": model.pricing.completion_per_1m,
        }
        projected["pricing"] = {
            key: value for key, value in pricing.items() if value not in (None, "")
        }
    if model.param_size:
        projected["param_size"] = model.param_size
    if model.api_format:
        projected["api_format"] = model.api_format
    if model.description:
        projected["description"] = model.description
    if model.capabilities:
        projected["capabilities"] = list(model.capabilities)
    if model.quality_score is not None:
        projected["quality_score"] = model.quality_score

    return projected


def _project_endpoint(endpoint: Any) -> dict[str, Any]:
    projected = {
        "name": endpoint.name,
        "weight": endpoint.weight,
        "endpoint": endpoint.endpoint,
    }
    if endpoint.protocol and endpoint.protocol != "http":
        projected["protocol"] = endpoint.protocol
    return projected


def _project_decisions(user_config: UserConfig) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for decision in user_config.decisions:
        entry = {
            "name": decision.name,
            "rules": _project_rules(decision.rules),
        }
        if decision.description:
            entry["description"] = decision.description
        if decision.priority is not None:
            entry["priority"] = decision.priority
        if decision.modelRefs:
            entry["modelRefs"] = [
                _project_model_ref(model_ref) for model_ref in decision.modelRefs
            ]
        projected.append(entry)
    return projected


def _project_rules(rules: Any) -> dict[str, Any]:
    projected = {"operator": rules.operator}
    if rules.conditions:
        projected["conditions"] = [
            _project_condition(condition) for condition in rules.conditions
        ]
    return projected


def _project_condition(condition: Any) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    if condition.type is not None:
        projected["type"] = condition.type
    if condition.name is not None:
        projected["name"] = condition.name
    if condition.operator is not None:
        projected["operator"] = condition.operator
    if condition.conditions:
        projected["conditions"] = [
            _project_condition(child) for child in condition.conditions
        ]
    return projected


def _project_model_ref(model_ref: Any) -> dict[str, Any]:
    projected = {
        "model": model_ref.model,
        "use_reasoning": bool(model_ref.use_reasoning),
    }
    if model_ref.reasoning_effort:
        projected["reasoning_effort"] = model_ref.reasoning_effort
    if model_ref.lora_name:
        projected["lora_name"] = model_ref.lora_name
    return projected
