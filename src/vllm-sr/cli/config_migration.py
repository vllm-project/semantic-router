"""Helpers for migrating legacy CLI config layouts to canonical v0.3 YAML."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cli.config_contract import (
    CANONICAL_TOP_LEVEL_KEYS,
    CANONICAL_VERSION,
    LEGACY_PROVIDER_DEFAULT_KEYS,
    LEGACY_PROVIDER_KEYS,
    LEGACY_ROUTING_KEYS,
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
)
from cli.config_migration_catalog import migrate_v03_catalog_contract
from cli.config_migration_global import normalize_global_layout, place_global_block
from cli.config_migration_model_runtime import migrate_prompt_guard_backend


def migrate_config_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical v0.3 config dict from legacy or mixed input data."""

    source = deepcopy(data or {})
    # Historically an omitted listener list meant that the local CLI would
    # synthesize its default listener.  An explicit empty list, in contrast,
    # is the metadata-only/external-gateway mode and must stay transport-free.
    router_owns_transport = not (
        "listeners" in source and source.get("listeners") == []
    )
    providers, routing, global_config = _prepare_blocks(source)
    routing_models, routing_models_by_name = _collect_routing_models(routing)

    _move_legacy_routing_blocks(source, routing)
    _move_legacy_flat_signal_blocks(source, routing)
    _move_legacy_provider_defaults(source, providers)
    provider_models = _migrate_provider_models(
        providers, routing_models, routing_models_by_name
    )
    _migrate_legacy_flat_model_bindings(
        source, provider_models, routing_models, routing_models_by_name
    )
    _ensure_routing_model_refs(
        providers, provider_models, routing_models, routing_models_by_name
    )
    if routing_models:
        routing["modelCards"] = routing_models
    routing.pop("models", None)

    _move_legacy_global_blocks(source, providers, global_config)
    _normalize_response_cache_contract(routing, global_config)
    _persist_provider_models(providers, provider_models)

    canonical: dict[str, Any] = {
        "version": CANONICAL_VERSION,
        "listeners": _clone_list(source.get("listeners")),
        "providers": providers,
        "routing": routing,
    }
    for key in ("evaluation", "evaluation_catalog"):
        if key in source:
            canonical[key] = deepcopy(source[key])
    for key in ("entrypoints", "recipes"):
        if key in source:
            canonical[key] = deepcopy(source[key])
    if global_config:
        canonical["global"] = global_config
    if "setup" in source:
        canonical["setup"] = deepcopy(source["setup"])
    _normalize_response_cache_plugins(canonical)
    migrate_v03_catalog_contract(
        canonical,
        router_owns_transport=router_owns_transport,
    )
    migrate_prompt_guard_backend(canonical)

    return canonical


def _normalize_response_cache_contract(
    routing: dict[str, Any], global_config: dict[str, Any]
) -> None:
    stores = _as_dict(global_config.get("stores"))
    if "response_cache" in stores and "semantic_cache" in stores:
        raise ValueError(
            "global.stores.response_cache conflicts with global.stores.semantic_cache"
        )
    if "semantic_cache" in stores:
        stores["response_cache"] = stores.pop("semantic_cache")
    if stores:
        global_config["stores"] = stores
    _normalize_response_cache_plugins(routing)


def _normalize_response_cache_plugins(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _normalize_response_cache_plugins(item)
        return
    if not isinstance(value, dict):
        return
    plugins = value.get("plugins")
    if isinstance(plugins, list):
        seen: set[str] = set()
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            plugin_type = plugin.get("type")
            if plugin_type in {
                "semantic-cache",
                "semantic_cache",
                "response-cache",
            }:
                plugin["type"] = "response_cache"
            normalized = plugin.get("type")
            if normalized in seen:
                raise ValueError(f"duplicate plugin after normalization: {normalized}")
            if isinstance(normalized, str):
                seen.add(normalized)
    for child in value.values():
        _normalize_response_cache_plugins(child)


def _prepare_blocks(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    providers = _as_dict(source.get("providers"))
    routing = _as_dict(source.get("routing"))
    global_config = normalize_global_layout(_as_dict(source.get("global")))
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


def _move_legacy_flat_signal_blocks(
    source: dict[str, Any], routing: dict[str, Any]
) -> None:
    signals = _ensure_dict(routing, "signals")
    for legacy_key, canonical_key in LEGACY_SIGNAL_KEY_TO_CANONICAL.items():
        if canonical_key in signals:
            continue
        legacy_value = _clone_list(source.get(legacy_key))
        if legacy_value:
            signals[canonical_key] = legacy_value


def _move_legacy_provider_defaults(
    source: dict[str, Any], providers: dict[str, Any]
) -> None:
    defaults = _as_dict(providers.get("defaults"))
    for key in LEGACY_PROVIDER_DEFAULT_KEYS:
        if key not in defaults:
            _set_if_missing(defaults, key, source.get(key))
    if defaults:
        providers["defaults"] = defaults


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
    access = _as_dict(model.get("access"))

    _set_if_missing(provider_model, "catalog", model.get("catalog"))
    _set_if_missing(provider_model, "reasoning", model.get("reasoning"))
    _set_if_missing(
        provider_model, "provider_model_id", access.get("provider_model_id")
    )
    _set_if_missing(provider_model, "provider_model_id", model.get("provider_model_id"))
    _set_if_missing(provider_model, "pricing", access.get("pricing"))
    _set_if_missing(provider_model, "pricing", model.get("pricing"))
    _set_if_missing(provider_model, "reliability", access.get("reliability"))
    _set_if_missing(provider_model, "reliability", model.get("reliability"))
    _set_if_missing(provider_model, "api_format", access.get("api_format"))
    _set_if_missing(provider_model, "api_format", model.get("api_format"))
    _set_if_missing(
        provider_model, "external_model_ids", access.get("external_model_ids")
    )
    _set_if_missing(
        provider_model, "external_model_ids", model.get("external_model_ids")
    )
    _set_if_missing(provider_model, "reasoning_family", model.get("reasoning_family"))

    backend_refs = _clone_list(access.get("backend_refs") or model.get("backend_refs"))
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
    _set_if_missing(semantic_model, "param_size", model.get("param_size"))
    _set_if_missing(
        semantic_model, "context_window_size", model.get("context_window_size")
    )
    _set_if_missing(semantic_model, "description", model.get("description"))
    _set_if_missing(semantic_model, "capabilities", model.get("capabilities"))
    _set_if_missing(semantic_model, "loras", model.get("loras"))
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
    defaults = _as_dict(providers.get("defaults"))
    default_model = (
        defaults.get("model")
        or defaults.get("default_model")
        or providers.get("default_model")
    )
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
    model_catalog = _ensure_dict(global_config, "model_catalog")
    if "external_models" in providers and "external" not in model_catalog:
        model_catalog["external"] = deepcopy(providers.pop("external_models"))

    for key, value in source.items():
        if (
            key in CANONICAL_TOP_LEVEL_KEYS
            or key in LEGACY_ROUTING_KEYS
            or key in LEGACY_PROVIDER_KEYS
        ):
            continue
        if key == "auto_model_names" and isinstance(value, list):
            _ensure_dict(global_config, "router").setdefault(key, deepcopy(value))
            continue
        if value in (None, "", [], {}):
            continue
        place_global_block(global_config, key, value)


def _persist_provider_models(
    providers: dict[str, Any],
    provider_models: list[dict[str, Any]],
) -> None:
    defaults = _as_dict(providers.get("defaults"))
    for key in ("default_model", "reasoning_families", "default_reasoning_effort"):
        if key in providers and key not in defaults:
            defaults[key] = deepcopy(providers[key])

    providers.pop("backends", None)
    providers.pop("auth_profiles", None)
    providers.pop("model_targets", None)
    providers.pop("default_model", None)
    providers.pop("reasoning_families", None)
    providers.pop("default_reasoning_effort", None)
    if defaults:
        providers["defaults"] = defaults
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


def _migrate_legacy_flat_model_bindings(
    source: dict[str, Any],
    provider_models: list[dict[str, Any]],
    routing_models: list[dict[str, Any]],
    routing_models_by_name: dict[str, dict[str, Any]],
) -> None:
    legacy_model_config = _as_dict(source.get("model_config"))
    if not legacy_model_config:
        return

    provider_models_by_name = {
        str(model.get("name")): model
        for model in provider_models
        if isinstance(model, dict) and model.get("name")
    }
    backend_catalog = _build_legacy_backend_catalog(source)

    for model_name, raw_entry in legacy_model_config.items():
        if not isinstance(raw_entry, dict):
            continue

        semantic_model = _ensure_routing_model(
            str(model_name), routing_models, routing_models_by_name
        )
        _populate_semantic_model_fields(semantic_model, raw_entry)

        provider_model = provider_models_by_name.get(str(model_name))
        if provider_model is None:
            provider_model = {"name": str(model_name)}
            provider_models.append(provider_model)
            provider_models_by_name[str(model_name)] = provider_model

        _set_if_missing(provider_model, "provider_model_id", raw_entry.get("model_id"))
        _set_if_missing(provider_model, "pricing", raw_entry.get("pricing"))
        _set_if_missing(provider_model, "api_format", raw_entry.get("api_format"))
        _set_if_missing(
            provider_model, "external_model_ids", raw_entry.get("external_model_ids")
        )
        _set_if_missing(
            provider_model, "reasoning_family", raw_entry.get("reasoning_family")
        )

        if "backend_refs" not in provider_model:
            backend_refs = _legacy_backend_refs_for_model(raw_entry, backend_catalog)
            if backend_refs:
                provider_model["backend_refs"] = backend_refs


def _build_legacy_backend_catalog(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = _as_dict(source.get("provider_profiles"))
    catalog: dict[str, dict[str, Any]] = {}

    for raw_endpoint in _clone_list(source.get("vllm_endpoints")):
        if not isinstance(raw_endpoint, dict):
            continue

        endpoint_name = str(raw_endpoint.get("name", "")).strip()
        if not endpoint_name:
            continue

        backend_ref: dict[str, Any] = {"name": endpoint_name}
        profile_name = raw_endpoint.get("provider_profile")
        profile = (
            profiles.get(profile_name)
            if isinstance(profile_name, str)
            and isinstance(profiles.get(profile_name), dict)
            else None
        )
        if isinstance(profile, dict):
            for key in (
                "base_url",
                "type",
                "auth_header",
                "auth_prefix",
                "extra_headers",
                "api_version",
                "chat_path",
            ):
                value = profile.get(key)
                target_key = "provider" if key == "type" else key
                _set_if_missing(backend_ref, target_key, value)

        address = raw_endpoint.get("address")
        port = raw_endpoint.get("port")
        if isinstance(address, str) and address.strip():
            endpoint_value = address.strip()
            if isinstance(port, int) and port > 0:
                endpoint_value = f"{endpoint_value}:{port}"
            _set_if_missing(backend_ref, "endpoint", endpoint_value)

        for key in ("protocol", "weight", "type", "api_key", "api_key_env"):
            _set_if_missing(backend_ref, key, raw_endpoint.get(key))

        catalog[endpoint_name] = backend_ref

    return catalog


def _legacy_backend_refs_for_model(
    legacy_model: dict[str, Any],
    backend_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    backend_refs: list[dict[str, Any]] = []
    access_key = legacy_model.get("access_key")

    for endpoint_name in _clone_list(legacy_model.get("preferred_endpoints")):
        if not isinstance(endpoint_name, str):
            continue
        backend = backend_catalog.get(endpoint_name)
        if not isinstance(backend, dict):
            continue
        backend_ref = deepcopy(backend)
        _set_if_missing(backend_ref, "api_key", access_key)
        backend_refs.append(backend_ref)

    return backend_refs


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


def _ensure_dict(target: dict[str, Any], key: str) -> dict[str, Any]:
    existing = target.get(key)
    if isinstance(existing, dict):
        return existing
    created: dict[str, Any] = {}
    target[key] = created
    return created


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
