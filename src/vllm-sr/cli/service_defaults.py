"""CLI-local canonical defaults for router service storage backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)

CANONICAL_SERVICE_DEFAULTS: dict[str, dict[str, object]] = {
    "response_api": {
        "enabled": True,
        "store_backend": "redis",
    },
    "router_replay": {
        "enabled": True,
        "store_backend": "postgres",
    },
}

CANONICAL_STORE_DEFAULTS: dict[str, dict[str, object]] = {
    "semantic_cache": {
        "enabled": True,
        "backend_type": "milvus",
    },
}

_INVALID_MAPPING = object()


def is_setup_mode_config(config: Mapping[str, Any]) -> bool:
    """Return True when the config is a setup-mode bootstrap config."""
    setup_config = config.get("setup")
    return isinstance(setup_config, Mapping) and setup_config.get("mode") is True


def detect_canonical_storage_backends(config: Mapping[str, Any]) -> set[str]:
    """Return provisionable backends implied by canonical service and store defaults.

    Setup-mode bootstrap still uses the local default sidecar stack so the
    first `vllm-sr serve` starts the same local storage dependencies that the
    activated config will later consume.
    """
    backends: set[str] = set()
    for service_key in CANONICAL_SERVICE_DEFAULTS:
        backend = effective_service_backend(config, service_key)
        if backend in {"redis", "postgres"}:
            backends.add(backend)

    store_backend = _effective_store_backend(config, "semantic_cache", "backend_type")
    if store_backend == "milvus":
        backends.add("milvus")

    return backends


def effective_service_backend(
    config: Mapping[str, Any], service_key: str
) -> str | None:
    """Return the effective store backend for a router service."""
    service_config = _merged_service_config(config, service_key)
    if service_config is None:
        return None
    if service_config.get("enabled") is False:
        return None

    backend = str(service_config.get("store_backend") or "").strip().lower()
    return backend or None


def inject_local_service_runtime_defaults(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Inject local Docker connection defaults for canonical service backends."""
    if is_setup_mode_config(config):
        return False

    services = _ensure_runtime_services_mapping(config)
    if services is None:
        return False

    changed = False
    for service_key, default_config in CANONICAL_SERVICE_DEFAULTS.items():
        changed = (
            _inject_service_runtime_defaults(
                services,
                service_key,
                default_config,
                stack_layout,
            )
            or changed
        )

    return changed


def _services_mapping(config: Mapping[str, Any]) -> Mapping[str, Any] | object:
    global_config = config.get("global")
    if global_config is None:
        return {}
    if not isinstance(global_config, Mapping):
        log.warning(
            "Skipping canonical service defaults because global is not a mapping"
        )
        return _INVALID_MAPPING

    services = global_config.get("services")
    if services is None:
        return {}
    if not isinstance(services, Mapping):
        log.warning(
            "Skipping canonical service defaults because global.services is not a mapping"
        )
        return _INVALID_MAPPING
    return services


def _merged_service_config(
    config: Mapping[str, Any], service_key: str
) -> dict[str, object] | None:
    defaults = CANONICAL_SERVICE_DEFAULTS.get(service_key)
    if defaults is None:
        return None

    services = _services_mapping(config)
    if services is _INVALID_MAPPING:
        return None

    raw_service = services.get(service_key)
    if raw_service is None:
        return dict(defaults)
    if not isinstance(raw_service, Mapping):
        log.warning(
            "Skipping canonical service defaults for global.services.%s because it is not a mapping",
            service_key,
        )
        return None

    merged = dict(defaults)
    merged.update(dict(raw_service))
    return merged


def _ensure_mapping(
    parent: dict[str, object], key: str, path: str
) -> dict[str, object] | None:
    child = parent.get(key)
    if child is None:
        mapping: dict[str, object] = {}
        parent[key] = mapping
        return mapping
    if not isinstance(child, dict):
        log.warning(
            "Skipping local service default injection because %s is not a mapping",
            path,
        )
        return None
    return child


def _ensure_runtime_services_mapping(
    config: dict[str, object],
) -> dict[str, object] | None:
    global_config = _ensure_mapping(config, "global", "global")
    if global_config is None:
        return None
    return _ensure_mapping(global_config, "services", "global.services")


def _inject_service_runtime_defaults(
    services: dict[str, object],
    service_key: str,
    default_config: dict[str, object],
    stack_layout: RuntimeStackLayout,
) -> bool:
    service_config, changed = _ensure_service_runtime_mapping(services, service_key)
    if service_config is None:
        return changed

    changed = _apply_missing_defaults(service_config, default_config) or changed
    if service_config.get("enabled") is False:
        return changed

    return (
        _inject_backend_runtime_defaults(service_key, service_config, stack_layout)
        or changed
    )


def _ensure_service_runtime_mapping(
    services: dict[str, object], service_key: str
) -> tuple[dict[str, object] | None, bool]:
    service_config = services.get(service_key)
    if service_config is None:
        service_mapping: dict[str, object] = {}
        services[service_key] = service_mapping
        return service_mapping, True
    if not isinstance(service_config, dict):
        log.warning(
            "Skipping local service default injection for global.services.%s because it is not a mapping",
            service_key,
        )
        return None, False
    return service_config, False


def _inject_backend_runtime_defaults(
    service_key: str,
    service_config: dict[str, object],
    stack_layout: RuntimeStackLayout,
) -> bool:
    backend = _normalized_backend_value(service_config.get("store_backend"))
    if not backend:
        return False

    backend_defaults = _local_backend_defaults(
        service_key=service_key,
        backend=backend,
        stack_layout=stack_layout,
    )
    if not backend_defaults:
        return False

    backend_config, changed = _ensure_backend_runtime_mapping(
        service_config,
        service_key,
        backend,
        backend_defaults,
    )
    if backend_config is None:
        return changed

    return _apply_missing_or_blank_defaults(backend_config, backend_defaults) or changed


def _ensure_backend_runtime_mapping(
    service_config: dict[str, object],
    service_key: str,
    backend: str,
    backend_defaults: dict[str, object],
) -> tuple[dict[str, object] | None, bool]:
    backend_config = service_config.get(backend)
    if backend_config is None:
        service_config[backend] = dict(backend_defaults)
        created_config = service_config.get(backend)
        if isinstance(created_config, dict):
            return created_config, True
        return None, True
    if not isinstance(backend_config, dict):
        log.warning(
            "Skipping local service default injection for global.services.%s.%s because it is not a mapping",
            service_key,
            backend,
        )
        return None, False
    return backend_config, False


def _apply_missing_defaults(
    target: dict[str, object], defaults: Mapping[str, object]
) -> bool:
    changed = False
    for key, value in defaults.items():
        if key not in target:
            target[key] = value
            changed = True
    return changed


def _apply_missing_or_blank_defaults(
    target: dict[str, object], defaults: Mapping[str, object]
) -> bool:
    changed = False
    for key, value in defaults.items():
        if key not in target or target[key] in (None, ""):
            target[key] = value
            changed = True
    return changed


def _normalized_backend_value(raw_backend: object) -> str | None:
    backend = str(raw_backend or "").strip().lower()
    return backend or None


def inject_local_store_runtime_defaults(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Inject local Docker connection defaults for canonical store backends."""
    if (
        is_setup_mode_config(config)
        or _effective_store_backend(config, "semantic_cache", "backend_type")
        != "milvus"
    ):
        return False

    stores = _ensure_stores_mapping(config)
    if stores is None:
        return False

    cache_config = stores.get("semantic_cache")
    if cache_config is None:
        cache_mapping: dict[str, object] = {}
        stores["semantic_cache"] = cache_mapping
        cache_config = cache_mapping
    elif not isinstance(cache_config, dict):
        log.warning(
            "Skipping local store default injection for global.stores.semantic_cache "
            "because it is not a mapping"
        )
        return False

    if "backend_type" not in cache_config:
        cache_config["backend_type"] = "milvus"

    connection_defaults = {
        "host": stack_layout.milvus_container_name,
        "port": 19530,
        "database": "default",
        "timeout": 30,
    }

    collection_defaults: dict[str, object] = {
        "name": "semantic_cache",
        "description": "Semantic cache for LLM request-response pairs",
        "vector_field": {
            "name": "embedding",
            "dimension": 384,
            "metric_type": "IP",
        },
        "index": {
            "type": "HNSW",
            "params": {"M": 16, "efConstruction": 64},
        },
    }

    search_defaults: dict[str, object] = {
        "params": {"ef": 64},
        "topk": 10,
    }

    development_defaults: dict[str, object] = {
        "auto_create_collection": True,
    }

    milvus_block = cache_config.get("milvus")
    if milvus_block is None:
        cache_config["milvus"] = {
            "connection": dict(connection_defaults),
            "collection": dict(collection_defaults),
            "search": dict(search_defaults),
            "development": dict(development_defaults),
        }
        return True
    if not isinstance(milvus_block, dict):
        log.warning(
            "Skipping local store default injection for "
            "global.stores.semantic_cache.milvus because it is not a mapping"
        )
        return False

    c1 = _inject_sub_block(milvus_block, "connection", connection_defaults)
    c2 = _inject_sub_block(
        milvus_block, "collection", collection_defaults, {"name": "semantic_cache"}
    )
    c3 = _inject_sub_block(milvus_block, "search", search_defaults)
    c4 = _inject_sub_block(milvus_block, "development", development_defaults)
    return c1 or c2 or c3 or c4


def _inject_sub_block(
    parent: dict[str, object],
    key: str,
    full_defaults: dict[str, object],
    backfill_defaults: dict[str, object] | None = None,
) -> bool:
    """Inject or backfill a sub-block inside the Milvus config."""
    existing = parent.get(key)
    if existing is None:
        parent[key] = dict(full_defaults)
        return True
    if isinstance(existing, dict):
        return _apply_missing_or_blank_defaults(
            existing,
            backfill_defaults if backfill_defaults is not None else full_defaults,
        )
    return False


def _effective_store_backend(
    config: Mapping[str, Any], store_key: str, backend_field: str
) -> str | None:
    """Return the effective backend for a store entry, falling back to canonical defaults."""
    defaults = CANONICAL_STORE_DEFAULTS.get(store_key)
    if defaults is None:
        return None

    stores = _stores_mapping(config)
    if stores is _INVALID_MAPPING:
        return None

    store_config = stores.get(store_key)
    if store_config is None:
        return str(defaults.get(backend_field) or "").strip().lower() or None
    if not isinstance(store_config, Mapping):
        log.warning(
            "Skipping canonical store defaults for global.stores.%s "
            "because it is not a mapping",
            store_key,
        )
        return None

    if store_config.get("enabled") is False:
        return None

    raw = store_config.get(backend_field)
    if raw is not None:
        return str(raw).strip().lower() or None
    return str(defaults.get(backend_field) or "").strip().lower() or None


def _stores_mapping(config: Mapping[str, Any]) -> Mapping[str, Any] | object:
    global_config = config.get("global")
    if global_config is None:
        return {}
    if not isinstance(global_config, Mapping):
        log.warning("Skipping canonical store defaults because global is not a mapping")
        return _INVALID_MAPPING

    stores = global_config.get("stores")
    if stores is None:
        return {}
    if not isinstance(stores, Mapping):
        log.warning(
            "Skipping canonical store defaults because global.stores is not a mapping"
        )
        return _INVALID_MAPPING
    return stores


def _ensure_stores_mapping(
    config: dict[str, object],
) -> dict[str, object] | None:
    global_config = _ensure_mapping(config, "global", "global")
    if global_config is None:
        return None
    return _ensure_mapping(global_config, "stores", "global.stores")


def _local_backend_defaults(
    *, service_key: str, backend: str, stack_layout: RuntimeStackLayout
) -> dict[str, object]:
    if backend == "redis":
        return {
            "address": f"{stack_layout.redis_container_name}:6379",
            "db": 0,
        }

    if service_key == "router_replay" and backend == "postgres":
        return {
            "host": stack_layout.postgres_container_name,
            "port": 5432,
            "database": "vsr",
            "user": "router",
            "password": "router-secret",
            "ssl_mode": "disable",
        }

    return {}
