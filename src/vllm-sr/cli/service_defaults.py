"""Writing this stack's local storage endpoints into a runtime config.

The read side -- which backends a config asks for, and whether an endpoint is
one this stack provisions -- lives in :mod:`cli.managed_storage_detection`.
This module only materializes: it fills the local container address, port, and
credential reference into the service and store blocks that named a managed
backend, and leaves every endpoint it does not run untouched.
"""

from __future__ import annotations

from collections.abc import Mapping

from cli.managed_storage_detection import (
    CANONICAL_SERVICE_DEFAULTS,
    _backend_config_uses_managed_endpoint,
    _effective_store_backend,
    _vector_store_metadata_backend,
    is_setup_mode_config,
)
from cli.runtime_stack import RuntimeStackLayout
from cli.storage_secrets import (
    MANAGED_POSTGRES_DATABASE,
    MANAGED_POSTGRES_USER,
    POSTGRES_PASSWORD_PLACEHOLDER,
    REDIS_PASSWORD_PLACEHOLDER,
)
from cli.utils import get_logger

log = get_logger(__name__)

LOCAL_MANAGEMENT_API_DEFAULTS: dict[str, object] = {
    "bind_address": "0.0.0.0",
    "port": 8080,
}

# Credential fields the CLI owns outright on an endpoint it provisions itself.
MANAGED_CREDENTIAL_FIELDS: tuple[str, ...] = ("password",)


def inject_local_service_runtime_defaults(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Inject local Docker connection defaults for canonical service backends."""
    if is_setup_mode_config(config):
        return False

    services = _ensure_runtime_services_mapping(config)
    if services is None:
        return False

    changed = _inject_local_management_api_defaults(services)
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


def _inject_local_management_api_defaults(services: dict[str, object]) -> bool:
    """Materialize a container-reachable listener only when intent is absent.

    Explicit management listener fields remain authoritative. The Router's
    container entrypoint marks this wildcard listener as internal; host
    publication is independently constrained to loopback by container_start.
    """

    if "management_api" in services:
        return False
    services["management_api"] = dict(LOCAL_MANAGEMENT_API_DEFAULTS)
    return True


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

    existing = service_config.get(backend)
    if isinstance(existing, Mapping) and not _backend_config_uses_managed_endpoint(
        existing, backend, stack_layout
    ):
        # An explicitly external endpoint is left exactly as written. Filling
        # this stack's local defaults into it would aim the service at a
        # container that does not hold that data, and would replace a working
        # password with a placeholder only this stack's own Router expands.
        log.debug(
            "Skipping local backend defaults for global.services.%s.%s because "
            "it names an endpoint this stack does not manage",
            service_key,
            backend,
        )
        return False

    backend_config, changed = _ensure_backend_runtime_mapping(
        service_config,
        service_key,
        backend,
        backend_defaults,
    )
    if backend_config is None:
        return changed

    changed = (
        _apply_missing_or_blank_defaults(backend_config, backend_defaults) or changed
    )
    return (
        _apply_managed_credential_defaults(
            backend_config,
            backend_defaults,
            f"global.services.{service_key}.{backend}",
        )
        or changed
    )


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


def _apply_managed_credential_defaults(
    target: dict[str, object], defaults: Mapping[str, object], where: str
) -> bool:
    """Force a managed endpoint's credential fields back to the placeholder.

    ``_apply_missing_or_blank_defaults`` keeps whatever the user wrote, which
    is right for hosts, ports, and database names. It is wrong for the
    credential of a container this CLI provisions: the CLI generates that
    value, so a surviving literal is a stale one that authenticates against
    nothing once the backend is re-keyed. Overwriting is therefore scoped to
    these fields on managed endpoints and leaves the general missing-or-blank
    semantics untouched.

    No known constant is matched by name. Recognizing one shipped default
    would still leave every other hand-written password to fail the same way,
    with no clue as to why.
    """

    changed = False
    for key in MANAGED_CREDENTIAL_FIELDS:
        if key not in defaults:
            continue
        placeholder = defaults[key]
        current = target.get(key)
        if current == placeholder:
            continue
        if current is not None and str(current).strip():
            # A user may have re-keyed their managed backend by hand. Replacing
            # that silently would hand them an authentication failure with
            # nothing pointing at the cause. The value itself is never logged.
            log.warning(
                "Replacing the configured %s on %s with this stack's generated "
                "credential, because the CLI provisions that backend and owns "
                "its credentials. Point the service at an external endpoint to "
                "keep your own value, or use `vllm-sr storage rotate` to change "
                "the generated one.",
                key,
                where,
            )
        target[key] = placeholder
        changed = True
    return changed


def _normalized_backend_value(raw_backend: object) -> str | None:
    backend = str(raw_backend or "").strip().lower()
    return backend or None


def inject_local_store_runtime_defaults(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Inject local Docker connection defaults for canonical store backends."""
    if is_setup_mode_config(config):
        return False

    wants_milvus_cache = (
        _effective_store_backend(config, "response_cache", "backend_type") == "milvus"
    )
    wants_vector_metadata_postgres = (
        _vector_store_metadata_backend(config) == "postgres"
    )
    if not wants_milvus_cache and not wants_vector_metadata_postgres:
        return False

    stores = _ensure_stores_mapping(config)
    if stores is None:
        return False

    changed = False
    if wants_milvus_cache:
        changed = (
            _inject_response_cache_milvus_defaults(stores, stack_layout) or changed
        )
    if wants_vector_metadata_postgres:
        changed = (
            _inject_vector_store_metadata_postgres_defaults(stores, stack_layout)
            or changed
        )
    return changed


def _inject_response_cache_milvus_defaults(
    stores: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    if "response_cache" not in stores and "semantic_cache" in stores:
        stores["response_cache"] = stores.pop("semantic_cache")
    cache_config = stores.get("response_cache")
    if cache_config is None:
        cache_mapping: dict[str, object] = {}
        stores["response_cache"] = cache_mapping
        cache_config = cache_mapping
    elif not isinstance(cache_config, dict):
        log.warning(
            "Skipping local store default injection for global.stores.response_cache "
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
            "dimension": 768,
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
            "global.stores.response_cache.milvus because it is not a mapping"
        )
        return False

    c1 = _inject_sub_block(milvus_block, "connection", connection_defaults)
    c2 = _inject_sub_block(
        milvus_block, "collection", collection_defaults, {"name": "semantic_cache"}
    )
    c3 = _inject_sub_block(milvus_block, "search", search_defaults)
    c4 = _inject_sub_block(milvus_block, "development", development_defaults)
    return c1 or c2 or c3 or c4


def _inject_vector_store_metadata_postgres_defaults(
    stores: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    vs_config = stores.get("vector_store")
    if not isinstance(vs_config, dict):
        log.warning(
            "Skipping local store default injection for global.stores.vector_store "
            "because it is not a mapping"
        )
        return False

    postgres_defaults = _local_postgres_defaults(stack_layout)
    metadata_config = vs_config.get("metadata_postgres")
    if metadata_config is None:
        vs_config["metadata_postgres"] = dict(postgres_defaults)
        return True
    if not isinstance(metadata_config, dict):
        log.warning(
            "Skipping local store default injection for "
            "global.stores.vector_store.metadata_postgres because it is not a mapping"
        )
        return False

    if not _backend_config_uses_managed_endpoint(
        metadata_config, "postgres", stack_layout
    ):
        log.debug(
            "Skipping local store defaults for "
            "global.stores.vector_store.metadata_postgres because it names an "
            "endpoint this stack does not manage"
        )
        return False

    changed = _apply_missing_or_blank_defaults(metadata_config, postgres_defaults)
    return (
        _apply_managed_credential_defaults(
            metadata_config,
            postgres_defaults,
            "global.stores.vector_store.metadata_postgres",
        )
        or changed
    )


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
            "password": REDIS_PASSWORD_PLACEHOLDER,
        }

    if service_key == "router_replay" and backend == "postgres":
        return _local_postgres_defaults(stack_layout)

    return {}


def _local_postgres_defaults(stack_layout: RuntimeStackLayout) -> dict[str, object]:
    # The password is a reference, never a value. Router expands ``${VAR}``
    # across the whole YAML tree at startup, so the generated runtime config
    # can name this stack's credential without ever carrying it, and the value
    # reaches only the Router container's environment.
    return {
        "host": stack_layout.postgres_container_name,
        "port": 5432,
        "database": MANAGED_POSTGRES_DATABASE,
        "user": MANAGED_POSTGRES_USER,
        "password": POSTGRES_PASSWORD_PLACEHOLDER,
        "ssl_mode": "disable",
    }
