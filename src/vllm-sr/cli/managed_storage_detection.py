"""What a canonical service/store config says about managed storage.

Every question of the form "does this config ask this stack to run a backend,
and is the endpoint one we provision" is answered here, together with the one
inventory of canonical service and store defaults those answers are measured
against. Keeping the read side apart from `service_defaults`, which writes this
stack's local endpoints back into a config, is what stops the same inventory
from being restated on both sides and drifting.

Reads only: nothing here mutates a config, and nothing here imports the
injection side.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)

CANONICAL_SERVICE_DEFAULTS: dict[str, dict[str, object]] = {
    "response_api": {
        "enabled": True,
        "store_backend": "redis",
    },
    "router_replay": {
        "enabled": False,
        "store_backend": "memory",
    },
    "startup_status": {
        "store_backend": "file",
    },
}

CANONICAL_STORE_DEFAULTS: dict[str, dict[str, object]] = {
    "response_cache": {
        "enabled": True,
        "backend_type": "memory",
    },
}

# Sentinel for "the config named this block but it is not a mapping", which is
# not the same answer as "the block is absent".
_INVALID_MAPPING = object()

# The field each backend block uses to name its endpoint.
_BACKEND_ENDPOINT_KEYS = {"postgres": "host"}
_DEFAULT_BACKEND_ENDPOINT_KEY = "address"


def is_setup_mode_config(config: Mapping[str, Any]) -> bool:
    """Return True when the config is a setup-mode bootstrap config."""
    setup_config = config.get("setup")
    return isinstance(setup_config, Mapping) and setup_config.get("mode") is True


def detect_canonical_storage_backends(
    config: Mapping[str, Any], stack_layout: RuntimeStackLayout | None = None
) -> set[str]:
    """Return provisionable backends implied by canonical service and store defaults.

    When a stack layout is supplied, canonical omitted service endpoints are
    resolved to that stack while explicit external endpoints are never
    converted into speculative local sidecars.
    """
    backends: set[str] = set()
    for service_key in CANONICAL_SERVICE_DEFAULTS:
        backend = effective_service_backend(config, service_key)
        if backend in {"redis", "postgres", "milvus"} and (
            stack_layout is None
            or _service_uses_managed_backend(config, service_key, backend, stack_layout)
        ):
            backends.add(backend)

    if _response_cache_requires_managed_milvus(config, stack_layout):
        backends.add("milvus")

    vs_metadata = _vector_store_metadata_backend(config)
    if vs_metadata == "postgres" and (
        stack_layout is None
        or _vector_metadata_uses_managed_postgres(config, stack_layout)
    ):
        backends.add("postgres")

    return backends


def _response_cache_requires_managed_milvus(
    config: Mapping[str, Any], stack_layout: RuntimeStackLayout | None
) -> bool:
    if _effective_store_backend(config, "response_cache", "backend_type") != "milvus":
        return False

    if stack_layout is None:
        return True

    host = _response_cache_milvus_connection_host(config)
    if not host:
        # Reaching this branch already means the effective backend was
        # explicitly Milvus; runtime materialization fills this stack's host.
        return True

    return _is_managed_storage_endpoint(host, "milvus", stack_layout)


def _service_uses_managed_backend(
    config: Mapping[str, Any],
    service_key: str,
    backend: str,
    stack_layout: RuntimeStackLayout,
) -> bool:
    service_config = _merged_service_config(config, service_key)
    if not isinstance(service_config, Mapping):
        return False
    return _backend_config_uses_managed_endpoint(
        service_config.get(backend), backend, stack_layout
    )


def _backend_config_uses_managed_endpoint(
    backend_config: object, backend: str, stack_layout: RuntimeStackLayout
) -> bool:
    """Return whether one backend block names a container this stack provisions.

    An omitted or blank endpoint is the request to materialize the local one,
    so it counts as managed. Anything else is taken at face value: the CLI has
    no business rewriting an endpoint it does not run.
    """

    if backend_config is None:
        return True
    if not isinstance(backend_config, Mapping):
        return False
    endpoint_key = _BACKEND_ENDPOINT_KEYS.get(backend, _DEFAULT_BACKEND_ENDPOINT_KEY)
    endpoint = backend_config.get(endpoint_key)
    if not str(endpoint or "").strip():
        return True
    return _is_managed_storage_endpoint(endpoint, backend, stack_layout)


def _vector_metadata_uses_managed_postgres(
    config: Mapping[str, Any], stack_layout: RuntimeStackLayout
) -> bool:
    stores = _stores_mapping(config)
    if stores is _INVALID_MAPPING:
        return False
    vector_config = stores.get("vector_store")
    if not isinstance(vector_config, Mapping):
        return False
    postgres_config = vector_config.get("metadata_postgres")
    if postgres_config is None:
        return True
    if not isinstance(postgres_config, Mapping):
        return False
    if not str(postgres_config.get("host") or "").strip():
        return True
    return _is_managed_storage_endpoint(
        postgres_config.get("host"), "postgres", stack_layout
    )


def _is_managed_storage_endpoint(
    endpoint: object, backend: str, stack_layout: RuntimeStackLayout
) -> bool:
    value = str(endpoint or "").strip()
    if not value:
        return False
    if "://" in value:
        host = urlparse(value).hostname or ""
    elif value.startswith("[") and "]" in value:
        host = value[1 : value.index("]")]
    elif value.count(":") == 1:
        host = value.split(":", 1)[0]
    else:
        host = value
    managed_name = getattr(stack_layout, f"{backend}_container_name")
    return host.rstrip(".").lower() in {backend, managed_name.lower()}


def _response_cache_milvus_connection_host(config: Mapping[str, Any]) -> str | None:
    stores = _stores_mapping(config)
    if stores is _INVALID_MAPPING:
        return None

    cache_config = _response_cache_mapping(stores)
    if not isinstance(cache_config, Mapping):
        return None

    milvus_config = cache_config.get("milvus")
    if not isinstance(milvus_config, Mapping):
        return None

    connection_config = milvus_config.get("connection")
    if not isinstance(connection_config, Mapping):
        return None

    host = connection_config.get("host")
    if host is None:
        return None
    return str(host).strip() or None


def _response_cache_mapping(stores: Mapping[str, Any]) -> object:
    canonical = stores.get("response_cache")
    legacy = stores.get("semantic_cache")
    if canonical is not None:
        if legacy is not None:
            log.warning(
                "Ignoring deprecated global.stores.semantic_cache because "
                "global.stores.response_cache is configured"
            )
        return canonical
    return legacy


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

    store_config = (
        _response_cache_mapping(stores)
        if store_key == "response_cache"
        else stores.get(store_key)
    )
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


def _vector_store_metadata_backend(config: Mapping[str, Any]) -> str | None:
    """Return the metadata_store value from global.stores.vector_store, if set."""
    stores = _stores_mapping(config)
    if stores is _INVALID_MAPPING:
        return None
    vs_config = stores.get("vector_store")
    if not isinstance(vs_config, Mapping):
        return None
    if vs_config.get("enabled") is False:
        return None
    raw = vs_config.get("metadata_store")
    if raw is None:
        return None
    return str(raw).strip().lower() or None


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
