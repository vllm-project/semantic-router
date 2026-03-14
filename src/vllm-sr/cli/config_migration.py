"""Helpers for migrating legacy CLI config layouts to canonical v0.3 YAML."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_CANONICAL_VERSION = "v0.3"
_TOP_LEVEL_KEYS = {
    "version",
    "listeners",
    "providers",
    "routing",
    "global",
    "setup",
}
_LEGACY_ROUTING_KEYS = {"signals", "decisions"}


def migrate_config_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical v0.3 config dict from legacy or mixed input data."""

    source = deepcopy(data or {})
    providers, routing, global_config = _prepare_blocks(source)
    routing_models, routing_models_by_name = _collect_routing_models(routing)

    _move_legacy_routing_blocks(source, routing)
    provider_models = _migrate_provider_models(
        providers, routing_models, routing_models_by_name
    )
    _ensure_routing_model_refs(
        providers, provider_models, routing_models, routing_models_by_name
    )
    if routing_models:
        routing["modelCards"] = routing_models
    routing.pop("models", None)

    _move_legacy_global_blocks(source, providers, global_config)
    _persist_provider_models(providers, provider_models)

    canonical: dict[str, Any] = {
        "version": _CANONICAL_VERSION,
        "listeners": _clone_list(source.get("listeners")),
        "providers": providers,
        "routing": routing,
    }
    if global_config:
        canonical["global"] = global_config
    if "setup" in source:
        canonical["setup"] = deepcopy(source["setup"])

    return canonical


def _prepare_blocks(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    providers = _as_dict(source.get("providers"))
    routing = _as_dict(source.get("routing"))
    global_config = _as_dict(source.get("global"))
    return providers, routing, global_config


def _collect_routing_models(
    routing: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    routing_models = _clone_list(routing.get("modelCards") or routing.get("models"))
    routing_models_by_name = {
        item.get("name"): item
        for item in routing_models
        if isinstance(item, dict) and item.get("name")
    }
    return routing_models, routing_models_by_name


def _move_legacy_routing_blocks(
    source: dict[str, Any], routing: dict[str, Any]
) -> None:
    if "signals" in source and "signals" not in routing:
        routing["signals"] = deepcopy(source["signals"])
    if "decisions" in source and "decisions" not in routing:
        routing["decisions"] = deepcopy(source["decisions"])


def _migrate_provider_models(
    providers: dict[str, Any],
    routing_models: list[dict[str, Any]],
    routing_models_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    provider_models = _clone_list(providers.get("models"))
    backends = _as_dict(providers.get("backends"))
    auth_profiles = _as_dict(providers.get("auth_profiles"))
    model_targets = _as_dict(providers.get("model_targets"))

    normalized_provider_models: list[dict[str, Any]] = []
    for model in provider_models:
        normalized = _normalize_existing_provider_model(
            model, routing_models, routing_models_by_name
        )
        if normalized:
            normalized_provider_models.append(normalized)

    if not normalized_provider_models and model_targets:
        normalized_provider_models.extend(
            _convert_model_targets_to_provider_models(
                model_targets, backends, auth_profiles
            )
        )

    return normalized_provider_models


def _normalize_existing_provider_model(
    model: Any,
    routing_models: list[dict[str, Any]],
    routing_models_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(model, dict):
        return None

    model_name = str(model.get("name", "")).strip()
    if not model_name:
        return None

    semantic_model = _ensure_routing_model(
        model_name, routing_models, routing_models_by_name
    )
    _populate_semantic_model_fields(semantic_model, model)
    provider_model = {"name": model_name}
    _set_if_missing(provider_model, "provider_model_id", model.get("provider_model_id"))
    _set_if_missing(provider_model, "pricing", model.get("pricing"))
    _set_if_missing(provider_model, "api_format", model.get("api_format"))
    _set_if_missing(
        provider_model, "external_model_ids", model.get("external_model_ids")
    )

    backend_refs = _clone_list(model.get("backend_refs"))
    if backend_refs:
        provider_model["backend_refs"] = backend_refs
        return provider_model

    migrated_backend_refs = _migrate_legacy_endpoints_to_backend_refs(model)
    if migrated_backend_refs:
        provider_model["backend_refs"] = migrated_backend_refs

    return provider_model


def _populate_semantic_model_fields(
    semantic_model: dict[str, Any], model: dict[str, Any]
) -> None:
    _set_if_missing(
        semantic_model, "reasoning_family_ref", model.get("reasoning_family")
    )
    _set_if_missing(semantic_model, "param_size", model.get("param_size"))
    _set_if_missing(
        semantic_model, "context_window_size", model.get("context_window_size")
    )
    _set_if_missing(semantic_model, "description", model.get("description"))
    _set_if_missing(semantic_model, "capabilities", model.get("capabilities"))
    _set_if_missing(semantic_model, "quality_score", model.get("quality_score"))
    _set_if_missing(semantic_model, "modality", model.get("modality"))
    _set_if_missing(semantic_model, "tags", model.get("tags"))


def _migrate_legacy_endpoints_to_backend_refs(
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    backend_refs: list[dict[str, Any]] = []
    for endpoint in _clone_list(model.get("endpoints", [])):
        if not isinstance(endpoint, dict):
            continue
        endpoint_value = endpoint.get("endpoint")
        if not endpoint_value:
            continue
        backend_ref = {
            "name": endpoint.get("name") or "primary",
            "endpoint": endpoint.get("endpoint"),
            "protocol": endpoint.get("protocol"),
            "weight": endpoint.get("weight"),
            "type": endpoint.get("type"),
        }
        if model.get("access_key"):
            backend_ref["api_key"] = model["access_key"]
        backend_refs.append(
            {
                key: value
                for key, value in backend_ref.items()
                if value not in (None, "", [])
            }
        )
    return backend_refs


def _ensure_routing_model_refs(
    providers: dict[str, Any],
    provider_models: list[dict[str, Any]],
    routing_models: list[dict[str, Any]],
    routing_models_by_name: dict[str, dict[str, Any]],
) -> None:
    default_model = providers.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        _ensure_routing_model(
            default_model.strip(), routing_models, routing_models_by_name
        )

    for provider_model in provider_models:
        if not isinstance(provider_model, dict):
            continue
        model_name = provider_model.get("name")
        if model_name:
            _ensure_routing_model(
                str(model_name), routing_models, routing_models_by_name
            )


def _move_legacy_global_blocks(
    source: dict[str, Any],
    providers: dict[str, Any],
    global_config: dict[str, Any],
) -> None:
    if "external_models" in providers and "external_models" not in global_config:
        global_config["external_models"] = deepcopy(providers.pop("external_models"))

    for key, value in source.items():
        if key in _TOP_LEVEL_KEYS or key in _LEGACY_ROUTING_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        if key not in global_config:
            global_config[key] = deepcopy(value)


def _persist_provider_models(
    providers: dict[str, Any],
    provider_models: list[dict[str, Any]],
) -> None:
    providers.pop("backends", None)
    providers.pop("auth_profiles", None)
    providers.pop("model_targets", None)
    if provider_models:
        providers["models"] = provider_models


def _ensure_routing_model(
    model_name: str,
    routing_models: list[dict[str, Any]],
    routing_models_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    existing = routing_models_by_name.get(model_name)
    if existing is not None:
        return existing

    created = {"name": model_name}
    routing_models.append(created)
    routing_models_by_name[model_name] = created
    return created


def _convert_model_targets_to_provider_models(
    model_targets: dict[str, Any],
    backends: dict[str, Any],
    auth_profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    provider_models: list[dict[str, Any]] = []
    for model_name, raw_target in model_targets.items():
        if not isinstance(raw_target, dict):
            continue
        provider_model: dict[str, Any] = {"name": model_name}
        _set_if_missing(
            provider_model, "provider_model_id", raw_target.get("provider_model_id")
        )
        _set_if_missing(provider_model, "pricing", raw_target.get("pricing"))
        _set_if_missing(provider_model, "api_format", raw_target.get("api_format"))
        _set_if_missing(
            provider_model, "external_model_ids", raw_target.get("external_model_ids")
        )

        auth_ref = raw_target.get("auth_ref")
        auth_profile = (
            auth_profiles.get(auth_ref) if isinstance(auth_ref, str) else None
        )

        backend_refs: list[dict[str, Any]] = []
        for backend_name in _clone_list(raw_target.get("backend_refs")):
            backend = backends.get(backend_name)
            if not isinstance(backend, dict):
                continue
            backend_ref = {"name": backend_name}
            for key in (
                "endpoint",
                "protocol",
                "weight",
                "type",
                "base_url",
                "provider",
                "auth_header",
                "auth_prefix",
                "extra_headers",
                "api_version",
                "chat_path",
            ):
                _set_if_missing(backend_ref, key, backend.get(key))
            if isinstance(auth_profile, dict):
                _set_if_missing(backend_ref, "api_key", auth_profile.get("api_key"))
                _set_if_missing(
                    backend_ref, "api_key_env", auth_profile.get("api_key_env")
                )
            backend_refs.append(backend_ref)

        if backend_refs:
            provider_model["backend_refs"] = backend_refs
        provider_models.append(provider_model)

    return provider_models


def _set_if_missing(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target:
        return
    if value in (None, "", [], {}):
        return
    target[key] = deepcopy(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _clone_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []
