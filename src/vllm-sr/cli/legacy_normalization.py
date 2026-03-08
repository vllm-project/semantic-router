"""Legacy flat-config normalization helpers for the transitional CLI parser."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

CURRENT_AUTHORING_VERSION = "v0.1"
_ROOT_PROVIDER_DEFAULT_KEYS = (
    "default_model",
    "default_reasoning_effort",
    "reasoning_families",
)
_LEGACY_SIGNAL_KEY_MAP = {
    "categories": "domains",
    "complexity_rules": "complexity",
    "context_rules": "context",
    "embedding_rules": "embeddings",
    "fact_check_rules": "fact_check",
    "jailbreak": "jailbreak",
    "keyword_rules": "keywords",
    "language_rules": "language",
    "modality_rules": "modality",
    "pii": "pii",
    "preference_rules": "preferences",
    "role_bindings": "role_bindings",
    "user_feedback_rules": "user_feedbacks",
}
_LEGACY_NORMALIZATION_KEYS = (
    *_ROOT_PROVIDER_DEFAULT_KEYS,
    *_LEGACY_SIGNAL_KEY_MAP,
    "model_config",
    "vllm_endpoints",
)


def normalize_legacy_user_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat runtime keys into the canonical nested CLI shape."""

    normalized = copy.deepcopy(data)
    if any(key in normalized for key in _LEGACY_NORMALIZATION_KEYS):
        normalized.setdefault("version", CURRENT_AUTHORING_VERSION)

    _normalize_root_provider_defaults(normalized)
    _normalize_signal_aliases(normalized)
    _normalize_provider_models(normalized)
    return normalized


def _normalize_root_provider_defaults(data: dict[str, Any]) -> None:
    if not any(key in data for key in _ROOT_PROVIDER_DEFAULT_KEYS):
        return

    providers = data.get("providers")
    if providers is None:
        providers = {}
        data["providers"] = providers
    if not isinstance(providers, dict):
        return

    for key in _ROOT_PROVIDER_DEFAULT_KEYS:
        if key not in data:
            continue
        root_value = data.pop(key)
        if key not in providers or providers[key] in (None, "", [], {}):
            providers[key] = root_value


def _normalize_signal_aliases(data: dict[str, Any]) -> None:
    if not any(key in data for key in _LEGACY_SIGNAL_KEY_MAP):
        return

    signals = data.get("signals")
    if signals is None:
        signals = {}
        data["signals"] = signals
    if not isinstance(signals, dict):
        return

    for legacy_key, nested_key in _LEGACY_SIGNAL_KEY_MAP.items():
        if legacy_key not in data:
            continue
        legacy_value = data.pop(legacy_key)
        if nested_key not in signals or signals[nested_key] in (None, [], {}):
            signals[nested_key] = legacy_value


def _normalize_provider_models(data: dict[str, Any]) -> None:
    model_config = data.get("model_config")
    vllm_endpoints = data.get("vllm_endpoints")

    if model_config is not None and not isinstance(model_config, dict):
        return
    if vllm_endpoints is not None and not isinstance(vllm_endpoints, list):
        return
    if model_config is None and vllm_endpoints is None:
        return

    providers = data.get("providers")
    if providers is None:
        providers = {}
        data["providers"] = providers
    if not isinstance(providers, dict):
        return

    legacy_models = _build_legacy_models(model_config or {}, vllm_endpoints or [])
    existing_models = providers.get("models")
    if not existing_models:
        if legacy_models:
            providers["models"] = legacy_models
        data.pop("model_config", None)
        data.pop("vllm_endpoints", None)
        return
    if not isinstance(existing_models, list):
        return

    _merge_legacy_models(existing_models, legacy_models)
    data.pop("model_config", None)
    data.pop("vllm_endpoints", None)


def _build_legacy_models(
    model_config: dict[str, Any], vllm_endpoints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    endpoints_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for endpoint in vllm_endpoints:
        model_name = endpoint.get("model")
        if not model_name:
            continue
        endpoints_by_model[model_name].append(endpoint)

    model_names = list(model_config)
    for model_name in endpoints_by_model:
        if model_name not in model_config:
            model_names.append(model_name)

    models = []
    for model_name in model_names:
        config = model_config.get(model_name) or {}
        model_entry: dict[str, Any] = {"name": model_name}
        endpoints = _build_model_endpoints(
            model_name,
            endpoints_by_model.get(model_name, []),
            config.get("preferred_endpoints") or [],
        )
        if endpoints:
            model_entry["endpoints"] = endpoints
        for key in (
            "access_key",
            "reasoning_family",
            "pricing",
            "param_size",
            "api_format",
            "description",
            "capabilities",
            "quality_score",
        ):
            value = config.get(key)
            if value is None:
                continue
            model_entry[key] = copy.deepcopy(value)
        models.append(model_entry)
    return models


def _build_model_endpoints(
    model_name: str,
    runtime_endpoints: list[dict[str, Any]],
    preferred_endpoints: list[str],
) -> list[dict[str, Any]]:
    indexed = {endpoint.get("name"): endpoint for endpoint in runtime_endpoints}
    ordered = []
    seen = set()

    for endpoint_name in preferred_endpoints:
        endpoint = indexed.get(endpoint_name)
        if endpoint is None:
            continue
        ordered.append(endpoint)
        seen.add(endpoint_name)

    for endpoint in runtime_endpoints:
        endpoint_name = endpoint.get("name")
        if endpoint_name in seen:
            continue
        ordered.append(endpoint)

    translated = []
    for endpoint in ordered:
        translated.append(_translate_runtime_endpoint(model_name, endpoint))
    return translated


def _translate_runtime_endpoint(
    model_name: str, runtime_endpoint: dict[str, Any]
) -> dict[str, Any]:
    protocol = runtime_endpoint.get("protocol") or "http"
    address = runtime_endpoint.get("address") or ""
    port = runtime_endpoint.get("port")
    path = runtime_endpoint.get("path") or ""
    endpoint_value = address
    if port is not None:
        endpoint_value = f"{endpoint_value}:{port}"
    if path:
        endpoint_value = f"{endpoint_value}{path}"

    full_name = runtime_endpoint.get("name") or model_name
    prefix = f"{model_name}_"
    endpoint_name = (
        full_name[len(prefix) :] if full_name.startswith(prefix) else full_name
    )

    translated = {
        "name": endpoint_name,
        "weight": runtime_endpoint.get("weight", 100),
        "endpoint": endpoint_value,
    }
    if protocol:
        translated["protocol"] = protocol
    return translated


def _merge_legacy_models(
    existing_models: list[dict[str, Any]], legacy_models: list[dict[str, Any]]
) -> None:
    existing_by_name = {}
    for model in existing_models:
        if isinstance(model, dict) and "name" in model:
            existing_by_name[model["name"]] = model

    for legacy_model in legacy_models:
        name = legacy_model.get("name")
        if not name:
            continue
        existing_model = existing_by_name.get(name)
        if existing_model is None:
            existing_models.append(copy.deepcopy(legacy_model))
            existing_by_name[name] = existing_models[-1]
            continue
        _merge_model_dict(existing_model, legacy_model)


def _merge_model_dict(
    existing_model: dict[str, Any], legacy_model: dict[str, Any]
) -> None:
    for key, value in legacy_model.items():
        if key == "name":
            continue
        if key == "endpoints":
            _merge_model_endpoints(existing_model, value)
            continue
        if key not in existing_model or existing_model[key] in (None, "", [], {}):
            existing_model[key] = copy.deepcopy(value)


def _merge_model_endpoints(
    existing_model: dict[str, Any], legacy_endpoints: Any
) -> None:
    if not legacy_endpoints:
        return
    existing_endpoints = existing_model.get("endpoints")
    if not existing_endpoints:
        existing_model["endpoints"] = copy.deepcopy(legacy_endpoints)
        return
    if not isinstance(existing_endpoints, list):
        return

    existing_by_name = {}
    for endpoint in existing_endpoints:
        if isinstance(endpoint, dict) and "name" in endpoint:
            existing_by_name[endpoint["name"]] = endpoint

    for legacy_endpoint in legacy_endpoints:
        name = legacy_endpoint.get("name")
        if not name:
            continue
        existing_endpoint = existing_by_name.get(name)
        if existing_endpoint is None:
            existing_endpoints.append(copy.deepcopy(legacy_endpoint))
            existing_by_name[name] = existing_endpoints[-1]
            continue
        for key, value in legacy_endpoint.items():
            if key not in existing_endpoint or existing_endpoint[key] in (
                None,
                "",
                [],
                {},
            ):
                existing_endpoint[key] = copy.deepcopy(value)
