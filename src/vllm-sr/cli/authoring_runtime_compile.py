"""Compile the TD001 canonical first slice into CLI runtime config overlays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from cli.authoring_projection import build_first_slice_authoring_config
from cli.models import PluginType, UserConfig

_CURRENT_AUTHORING_VERSION = "v0.1"
_DEFAULT_HTTP_PORT = 80
_DEFAULT_HTTPS_PORT = 443
_DEFAULT_EXTERNAL_ENDPOINT_PORT = 8000


def build_first_slice_runtime_overlay(user_config: UserConfig) -> dict[str, Any]:
    """Compile first-slice runtime keys and re-attach current CLI extensions."""

    runtime = compile_first_slice_runtime(
        build_first_slice_authoring_config(user_config)
    )

    if runtime.get("decisions"):
        runtime["decisions"] = _reattach_decision_extensions(
            runtime["decisions"], user_config.decisions
        )

    if user_config.providers.external_models:
        runtime["external_models"] = _compile_external_models(
            user_config.providers.external_models
        )

    return runtime


def compile_first_slice_runtime(authoring_config: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the TD001 canonical first authoring slice into runtime config keys."""

    _validate_authoring_version(authoring_config.get("version"))

    runtime: dict[str, Any] = {}

    listeners = _compile_listeners(authoring_config.get("listeners"))
    if listeners:
        runtime["listeners"] = listeners

    signals = authoring_config.get("signals") or {}
    keyword_rules = _compile_keyword_rules(signals.get("keywords"))
    if keyword_rules:
        runtime["keyword_rules"] = keyword_rules

    providers = authoring_config.get("providers") or {}
    runtime.update(_compile_providers(providers))

    decisions = _compile_decisions(authoring_config.get("decisions"))
    if decisions:
        runtime["decisions"] = decisions

    return runtime


def _validate_authoring_version(version: str | None) -> None:
    if version in (None, _CURRENT_AUTHORING_VERSION):
        return
    raise ValueError(f"unsupported authoring config version: {version}")


def _compile_listeners(listeners: Any) -> list[dict[str, Any]]:
    if not listeners:
        return []

    compiled: list[dict[str, Any]] = []
    for listener in listeners:
        entry = {
            "name": listener["name"],
            "address": listener["address"],
            "port": listener["port"],
        }
        if listener.get("timeout"):
            entry["timeout"] = listener["timeout"]
        compiled.append(entry)
    return compiled


def _compile_keyword_rules(keywords: Any) -> list[dict[str, Any]]:
    if not keywords:
        return []

    compiled: list[dict[str, Any]] = []
    for keyword in keywords:
        compiled.append(
            {
                "name": keyword["name"],
                "operator": keyword["operator"],
                "keywords": list(keyword["keywords"]),
                "case_sensitive": bool(keyword.get("case_sensitive", False)),
            }
        )
    return compiled


def _compile_providers(providers: Mapping[str, Any]) -> dict[str, Any]:
    models = providers.get("models") or []

    vllm_endpoints: list[dict[str, Any]] = []
    model_config: dict[str, dict[str, Any]] = {}
    for model in models:
        model_entry, preferred_endpoints = _compile_model(model)
        if preferred_endpoints:
            model_entry["preferred_endpoints"] = [
                endpoint["name"] for endpoint in preferred_endpoints
            ]
            vllm_endpoints.extend(preferred_endpoints)
        model_config[model["name"]] = model_entry

    compiled: dict[str, Any] = {
        "vllm_endpoints": vllm_endpoints,
        "model_config": model_config,
    }
    if providers.get("default_model"):
        compiled["default_model"] = providers["default_model"]
    if providers.get("reasoning_families"):
        compiled["reasoning_families"] = _compile_reasoning_families(
            providers["reasoning_families"]
        )
    if providers.get("default_reasoning_effort"):
        compiled["default_reasoning_effort"] = providers["default_reasoning_effort"]
    return compiled


def _compile_model(
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry: dict[str, Any] = {}
    for key in (
        "reasoning_family",
        "access_key",
        "param_size",
        "api_format",
        "description",
        "capabilities",
        "quality_score",
    ):
        value = model.get(key)
        if value in (None, "", [], {}):
            continue
        entry[key] = value

    pricing = model.get("pricing")
    if pricing:
        entry["pricing"] = {
            key: value
            for key, value in pricing.items()
            if value not in (None, "", [], {})
        }

    endpoints = [
        _compile_provider_endpoint(model["name"], endpoint)
        for endpoint in model.get("endpoints", [])
    ]
    return entry, endpoints


def _compile_provider_endpoint(
    model_name: str, endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    host, port, path = _split_provider_endpoint(
        endpoint["endpoint"], endpoint.get("protocol", "http")
    )
    compiled = {
        "name": f"{model_name}_{endpoint['name']}",
        "address": host,
        "port": port,
        "weight": endpoint["weight"],
        "protocol": endpoint.get("protocol", "http"),
        "model": model_name,
    }
    if path:
        compiled["path"] = path
    return compiled


def _split_provider_endpoint(endpoint: str, protocol: str) -> tuple[str, int, str]:
    path = ""
    host_port = endpoint
    if "/" in endpoint:
        host_port, path_suffix = endpoint.split("/", 1)
        path = f"/{path_suffix}"

    if ":" in host_port:
        host, port_str = host_port.split(":", 1)
        return host, int(port_str), path

    return (
        host_port,
        _DEFAULT_HTTPS_PORT if protocol == "https" else _DEFAULT_HTTP_PORT,
        path,
    )


def _compile_reasoning_families(
    reasoning_families: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        family_name: {
            "type": family_config["type"],
            "parameter": family_config["parameter"],
        }
        for family_name, family_config in reasoning_families.items()
    }


def _compile_decisions(decisions: Any) -> list[dict[str, Any]]:
    if not decisions:
        return []

    compiled: list[dict[str, Any]] = []
    for decision in decisions:
        entry = {
            "name": decision["name"],
            "rules": _compile_rules(decision["rules"]),
            "modelRefs": [
                _compile_model_ref(model_ref)
                for model_ref in decision.get("modelRefs", [])
            ],
        }
        if decision.get("description"):
            entry["description"] = decision["description"]
        if decision.get("priority") is not None:
            entry["priority"] = decision["priority"]
        compiled.append(entry)
    return compiled


def _compile_rules(rules: Mapping[str, Any]) -> dict[str, Any]:
    compiled = {"operator": rules.get("operator") or "AND"}
    if rules.get("conditions"):
        compiled["conditions"] = [
            _compile_condition(condition) for condition in rules["conditions"]
        ]
    return compiled


def _compile_condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    compiled: dict[str, Any] = {}
    for key in ("type", "name", "operator"):
        value = condition.get(key)
        if value is not None:
            compiled[key] = value
    if condition.get("conditions"):
        compiled["conditions"] = [
            _compile_condition(child) for child in condition["conditions"]
        ]
    return compiled


def _compile_model_ref(model_ref: Mapping[str, Any]) -> dict[str, Any]:
    compiled = {
        "model": model_ref["model"],
        "use_reasoning": bool(model_ref.get("use_reasoning", False)),
    }
    if model_ref.get("reasoning_effort"):
        compiled["reasoning_effort"] = model_ref["reasoning_effort"]
    if model_ref.get("lora_name"):
        compiled["lora_name"] = model_ref["lora_name"]
    return compiled


def _reattach_decision_extensions(
    compiled_decisions: list[dict[str, Any]], source_decisions: list[Any]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for compiled, source in zip(compiled_decisions, source_decisions, strict=True):
        decision = source.model_dump(mode="python", by_alias=True, exclude_none=True)
        for plugin in decision.get("plugins", []):
            _normalize_plugin_type(plugin)
        decision.update(compiled)
        merged.append(decision)
    return merged


def _normalize_plugin_type(plugin: dict[str, Any]) -> None:
    plugin_type = plugin.get("type")
    if isinstance(plugin_type, PluginType) or hasattr(plugin_type, "value"):
        plugin["type"] = plugin_type.value


def _compile_external_models(external_models: list[Any]) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for model in external_models:
        address, port, protocol = _parse_external_endpoint(model.endpoint)
        entry = {
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
            entry["access_key"] = model.access_key
        compiled.append(entry)
    return compiled


def _parse_external_endpoint(endpoint: str) -> tuple[str, int, str]:
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        return (
            parsed.hostname or "",
            parsed.port
            or (
                _DEFAULT_HTTPS_PORT if parsed.scheme == "https" else _DEFAULT_HTTP_PORT
            ),
            parsed.scheme,
        )

    if ":" in endpoint:
        address, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
        protocol = "https" if port == _DEFAULT_HTTPS_PORT else "http"
        return address, port, protocol

    return endpoint, _DEFAULT_EXTERNAL_ENDPOINT_PORT, "http"
