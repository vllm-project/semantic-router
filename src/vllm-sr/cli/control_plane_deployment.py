"""Target-neutral deployment contract for optional Router capabilities.

This module owns only deployment topology.  Dynamic Models, Recipes,
Entrypoints, identities, keys, policies, and counters remain Router Management
resources and never become container or Kubernetes configuration.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from cli.bootstrap import (
    LOCAL_BOOTSTRAP_TOKEN_NAME,
    LOCAL_REPLICA_ID_ENV,
)
from cli.runtime_env_names import runtime_env_name_is_allowed
from cli.runtime_stack import RuntimeStackLayout
from cli.storage_secrets import (
    POSTGRES_PASSWORD_ENV,
    REDIS_PASSWORD_ENV,
    StorageSecrets,
)

_CONTROL_PLANE_SECRET_FILE_PATHS = (
    ("global", "stores", "management", "postgres", "dsn_file"),
    ("global", "stores", "runtime", "redis", "url_file"),
    ("global", "services", "access", "credentials", "api_key_hmac_keyring_file"),
    ("global", "services", "access", "credentials", "delegation_hmac_keyring_file"),
    ("global", "services", "access", "credentials", "reveal", "kek_keyring_file"),
    ("global", "services", "access", "tenant_context", "signing_key_file"),
    ("global", "services", "backend_credentials", "provider_kek_keyring_file"),
    ("global", "services", "routing_security", "hmac_keyring_file"),
    ("global", "services", "management_api", "tls", "certificate_file"),
    ("global", "services", "management_api", "tls", "private_key_file"),
    ("global", "services", "management_api", "tls", "client_ca_bundle_file"),
    ("global", "services", "management_api", "auth", "token_signing_keyring_file"),
    (
        "global",
        "services",
        "management_api",
        "auth",
        "service_account_hmac_keyring_file",
    ),
    ("global", "services", "management_api", "auth", "invitation_hmac_keyring_file"),
    ("global", "services", "management_api", "auth", "response_kek_keyring_file"),
    ("global", "services", "management_api", "auth", "bootstrap", "token_file"),
    ("global", "services", "management_api", "auth", "recovery", "token_file"),
)


@dataclass(frozen=True)
class StoreSecretReference:
    """One exact, non-literal store credential source."""

    kind: Literal["env", "file"]
    value: str


@dataclass(frozen=True)
class ControlPlaneStoreReferences:
    """Configured durable-authority and distributed-runtime store references."""

    postgres: StoreSecretReference | None
    valkey: StoreSecretReference | None


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Capabilities derived from typed services and stores, never a public mode."""

    file_routing: bool
    durable_management: bool
    management_api: bool
    distributed_state: bool
    native_access: bool


@dataclass(frozen=True)
class LocalControlPlanePlan:
    """Pure description of bindings a local Router process will need.

    Planning reads only the compiled bootstrap plus the explicit runtime
    environment supplied by the caller. It does not inspect the host process,
    create credential files, or touch containers. That keeps provisioning as
    the sole credential authority and makes the apply/state-commit transaction
    observable as one ordered operation.
    """

    postgres_dsn_env: str | None = None
    valkey_url_env: str | None = None
    postgres_password_env: str | None = None
    redis_password_env: str | None = None
    replica_binding: tuple[str, str] | None = None

    @property
    def required_backends(self) -> frozenset[str]:
        backends: set[str] = set()
        if self.postgres_dsn_env:
            backends.add("postgres")
        if self.valkey_url_env:
            backends.add("redis")
        return frozenset(backends)


def runtime_capabilities(config: object) -> RuntimeCapabilities:
    """Derive deployment behavior from configured components."""

    if isinstance(config, Mapping):
        global_config = config.get("global") or {}
    else:
        global_config = getattr(config, "global_", None) or {}
    if not isinstance(global_config, Mapping):
        raise ValueError("global must be a mapping")
    stores = global_config.get("stores") or {}
    services = global_config.get("services") or {}
    if not isinstance(stores, Mapping):
        raise ValueError("global.stores must be a mapping")
    if not isinstance(services, Mapping):
        raise ValueError("global.services must be a mapping")
    durable_management = stores.get("management") is not None
    distributed_state = stores.get("runtime") is not None
    management = services.get("management_api") or {}
    access = services.get("access") or {}
    if not isinstance(management, Mapping):
        raise ValueError("global.services.management_api must be a mapping")
    if not isinstance(access, Mapping):
        raise ValueError("global.services.access must be a mapping")
    management_api = management.get("enabled") is True
    native_access = access.get("enabled") is True
    if management_api and not durable_management:
        raise ValueError(
            "global.services.management_api.enabled requires global.stores.management"
        )
    if distributed_state and not durable_management:
        raise ValueError("global.stores.runtime requires global.stores.management")
    if native_access and not (durable_management and distributed_state):
        raise ValueError(
            "global.services.access.enabled requires management and runtime stores"
        )
    return RuntimeCapabilities(
        file_routing=not durable_management,
        durable_management=durable_management,
        management_api=management_api,
        distributed_state=distributed_state,
        native_access=native_access,
    )


def control_plane_store_references(
    config: Mapping[str, object],
) -> ControlPlaneStoreReferences:
    """Read configured store references without resolving secret values."""

    capabilities = runtime_capabilities(config)
    global_config = _mapping(config.get("global") or {}, "global")
    stores = _mapping(global_config.get("stores") or {}, "global.stores")
    postgres_reference = None
    valkey_reference = None
    if capabilities.durable_management:
        management = _mapping(stores.get("management"), "global.stores.management")
        postgres = _mapping(
            management.get("postgres"), "global.stores.management.postgres"
        )
        postgres_reference = _one_secret_reference(
            postgres,
            env_field="dsn_env",
            file_field="dsn_file",
            path="global.stores.management.postgres",
        )
    if capabilities.distributed_state:
        runtime = _mapping(stores.get("runtime"), "global.stores.runtime")
        redis = _mapping(runtime.get("redis"), "global.stores.runtime.redis")
        valkey_reference = _one_secret_reference(
            redis,
            env_field="url_env",
            file_field="url_file",
            path="global.stores.runtime.redis",
        )
    return ControlPlaneStoreReferences(
        postgres=postgres_reference,
        valkey=valkey_reference,
    )


def plan_local_control_plane(
    config: Mapping[str, object],
    env_vars: Mapping[str, str],
    stack_layout: RuntimeStackLayout,
) -> LocalControlPlanePlan:
    """Plan local store bindings without IO or environment mutation.

    An already-populated environment reference is an operator-supplied
    external store.  A file reference is also deployment-owned and is never
    read or rewritten by the CLI.  Only an absent environment value selects a
    local sidecar.
    """

    capabilities = runtime_capabilities(config)
    postgres_dsn_env = None
    valkey_url_env = None
    replica_binding = None
    if capabilities.durable_management:
        references = control_plane_store_references(config)
        if references.postgres is None:
            raise ValueError("management store reference is unavailable")
        if references.postgres.kind == "env" and not _environment_value(
            references.postgres.value, env_vars
        ):
            postgres_dsn_env = references.postgres.value
        if (
            references.valkey is not None
            and references.valkey.kind == "env"
            and not _environment_value(references.valkey.value, env_vars)
        ):
            valkey_url_env = references.valkey.value
        if not _environment_value(LOCAL_REPLICA_ID_ENV, env_vars):
            replica_binding = (
                LOCAL_REPLICA_ID_ENV,
                f"local-{stack_layout.stack_name}",
            )

    return LocalControlPlanePlan(
        postgres_dsn_env=postgres_dsn_env,
        valkey_url_env=valkey_url_env,
        postgres_password_env=(
            POSTGRES_PASSWORD_ENV
            if _config_references_env(config, POSTGRES_PASSWORD_ENV)
            else None
        ),
        redis_password_env=(
            REDIS_PASSWORD_ENV
            if _config_references_env(config, REDIS_PASSWORD_ENV)
            else None
        ),
        replica_binding=replica_binding,
    )


def required_storage_secret_backends(
    plan: LocalControlPlanePlan, active_backends: set[str] | frozenset[str]
) -> frozenset[str]:
    """Return local backends whose committed credentials Router must receive."""

    required = set(plan.required_backends)
    for backend, env_name in (
        ("postgres", plan.postgres_password_env),
        ("redis", plan.redis_password_env),
    ):
        if not env_name:
            continue
        if backend not in active_backends and backend not in plan.required_backends:
            raise ValueError(
                f"Router config references {env_name}, but local {backend} was not provisioned"
            )
        required.add(backend)
    return frozenset(required)


def resolve_local_router_bindings(
    plan: LocalControlPlanePlan,
    active_backends: set[str] | frozenset[str],
    secrets: StorageSecrets | None,
    stack_layout: RuntimeStackLayout,
) -> dict[str, str]:
    """Resolve one plan from already-committed storage state.

    This function is deliberately pure: provisioning loads or creates the
    authoritative state before calling it. A missing state for any planned
    local backend is an error instead of an invitation to mint another key.
    """

    secret_backends = required_storage_secret_backends(plan, active_backends)
    if secret_backends and secrets is None:
        raise ValueError("local storage bindings require committed credential state")

    bindings: dict[str, str] = {}
    if secrets is not None:
        if plan.postgres_password_env and "postgres" in secret_backends:
            bindings[plan.postgres_password_env] = secrets.postgres.password
        if plan.redis_password_env and "redis" in secret_backends:
            bindings[plan.redis_password_env] = secrets.redis.password
        if plan.postgres_dsn_env:
            bindings[plan.postgres_dsn_env] = (
                f"postgresql://{quote(secrets.postgres.user, safe='')}:"
                f"{quote(secrets.postgres.password, safe='')}@"
                f"{stack_layout.postgres_container_name}:5432/"
                f"{quote(secrets.postgres.database, safe='')}?sslmode=disable"
            )
        if plan.valkey_url_env:
            bindings[plan.valkey_url_env] = (
                f"redis://:{quote(secrets.redis.password, safe='')}@"
                f"{stack_layout.redis_container_name}:6379/0"
            )
    if plan.replica_binding is not None:
        name, value = plan.replica_binding
        bindings[name] = value
    return bindings


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _one_secret_reference(
    config: Mapping[str, object],
    *,
    env_field: str,
    file_field: str,
    path: str,
) -> StoreSecretReference:
    env_name = config.get(env_field)
    file_name = config.get(file_field)
    env_name = env_name if isinstance(env_name, str) else ""
    file_name = file_name if isinstance(file_name, str) else ""
    if bool(env_name) == bool(file_name):
        raise ValueError(f"{path} requires exactly one of {env_field} or {file_field}")
    if env_name:
        if not runtime_env_name_is_allowed(env_name):
            raise ValueError(
                f"{path}.{env_field} must be an uppercase, non-reserved "
                "environment variable name"
            )
        return StoreSecretReference("env", env_name)
    if (
        file_name != file_name.strip()
        or not os.path.isabs(file_name)
        or os.path.normpath(file_name) != file_name
    ):
        raise ValueError(f"{path}.{file_field} must be an absolute canonical path")
    return StoreSecretReference("file", file_name)


def local_control_plane_secret_mounts(
    config: Mapping[str, object],
) -> tuple[str, ...]:
    """Return exact read-only mounts for configured control-plane references."""

    capabilities = runtime_capabilities(config)
    if not capabilities.durable_management:
        return ()
    references = control_plane_store_references(config)
    paths = {
        reference.value
        for reference in (references.postgres, references.valkey)
        if reference is not None and reference.kind == "file"
    }
    paths.update(
        value
        for path in _CONTROL_PLANE_SECRET_FILE_PATHS
        if (value := _string_at_path(config, path))
    )
    backend_credentials = _mapping_at_path(
        config, ("global", "services", "backend_credentials")
    )
    if backend_credentials is not None:
        paths.update(
            value
            for name, entry in backend_credentials.items()
            if name not in {"provider_kek_keyring_file", "provider_kek_keyring_env"}
            and isinstance(entry, Mapping)
            and (value := entry.get("secret_file"))
            and isinstance(value, str)
        )
    bootstrap_token = _bootstrap_token_file(config)
    mounts: set[str] = set()
    for path in paths:
        if path == bootstrap_token:
            token_directory = Path(path).parent
            if not token_directory.is_dir():
                raise ValueError(
                    f"bootstrap token directory does not exist: {token_directory}"
                )
            mounts.add(f"{token_directory}:{token_directory}:ro")
            continue
        if not Path(path).is_file():
            raise ValueError(f"control-plane secret file does not exist: {path}")
        mounts.add(f"{path}:{path}:ro")
    return tuple(sorted(mounts))


def _mapping_at_path(
    config: Mapping[str, object], path: tuple[str, ...]
) -> Mapping[str, object] | None:
    node: object = config
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node if isinstance(node, Mapping) else None


def _string_at_path(config: Mapping[str, object], path: tuple[str, ...]) -> str:
    node: object = config
    for key in path:
        if not isinstance(node, Mapping):
            return ""
        node = node.get(key)
    return node if isinstance(node, str) and node else ""


def _bootstrap_token_file(config: Mapping[str, object]) -> str:
    node: object = config.get("global")
    for key in ("services", "management_api", "auth", "bootstrap"):
        if not isinstance(node, Mapping):
            return ""
        node = node.get(key)
    if not isinstance(node, Mapping):
        return ""
    bootstrap = node
    value = bootstrap.get("token_file")
    if not isinstance(value, str) or not value:
        return ""
    if Path(value).name != LOCAL_BOOTSTRAP_TOKEN_NAME:
        return ""
    return value


def _environment_value(name: str, env_vars: Mapping[str, str]) -> str:
    configured = env_vars.get(name)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return ""


def _config_references_env(config: Mapping[str, object], name: str) -> bool:
    """Return whether a runtime value references one reserved local secret."""

    placeholder = f"${{{name}}}"
    pending: list[object] = [config]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str) and placeholder in node:
            return True
    return False
