"""Final v0.3 global-module structure shared by CLI authoring paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MISSING = object()

# These are the structured global boundaries whose children configure the
# Router-native Management and access services. Other established global
# modules retain their dedicated validators and typed Router decoder.
_GLOBAL_OBJECT_FIELDS: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (
        (),
        frozenset(
            {"billing", "router", "services", "stores", "integrations", "model_catalog"}
        ),
    ),
    (("billing",), frozenset({"currency"})),
    (
        ("services",),
        frozenset(
            {
                "api",
                "response_api",
                "agent",
                "observability",
                "management_api",
                "access",
                "backend_credentials",
                "backend_egress",
                "backend_dispatch",
                "routing_security",
                "router_replay",
                "startup_status",
            }
        ),
    ),
    (
        ("stores",),
        frozenset(
            {"response_cache", "memory", "vector_store", "management", "runtime"}
        ),
    ),
    (("integrations",), frozenset({"tools", "looper"})),
    (
        ("model_catalog",),
        frozenset({"embeddings", "system", "external", "kbs", "modules"}),
    ),
    (("stores", "management"), frozenset({"postgres"})),
    (
        ("stores", "management", "postgres"),
        frozenset({"dsn_file", "dsn_env", "max_connections"}),
    ),
    (("stores", "runtime"), frozenset({"redis"})),
    (
        ("stores", "runtime", "redis"),
        frozenset({"url_file", "url_env", "key_prefix"}),
    ),
    (
        ("services", "access"),
        frozenset(
            {"enabled", "credentials", "tenant_context", "enforcement", "usage_storage"}
        ),
    ),
    (
        ("services", "access", "credentials"),
        frozenset(
            {
                "api_key_hmac_keyring_file",
                "api_key_hmac_keyring_env",
                "delegation_hmac_keyring_file",
                "delegation_hmac_keyring_env",
                "reveal",
            }
        ),
    ),
    (
        ("services", "access", "credentials", "reveal"),
        frozenset({"enabled", "kek_keyring_file", "kek_keyring_env"}),
    ),
    (
        ("services", "access", "tenant_context"),
        frozenset({"signing_key_file", "signing_key_env", "max_start_age"}),
    ),
    (
        ("services", "access", "enforcement"),
        frozenset(
            {
                "failure_mode",
                "request_accounting",
                "token_accounting",
                "unknown_usage_action",
                "settle_on",
                "deduplicate_by",
                "max_usage_backlog",
            }
        ),
    ),
    (
        ("services", "access", "usage_storage"),
        frozenset({"create_ahead_months", "maintenance_interval", "raw_retention"}),
    ),
    (("services", "backend_egress"), frozenset({"policy_file"})),
    (
        ("services", "backend_dispatch"),
        frozenset(
            {
                "bind_address",
                "port",
                "audience",
                "capability_ttl",
                "max_request_body_bytes",
            }
        ),
    ),
    (
        ("services", "management_api"),
        frozenset(
            {"enabled", "bind_address", "port", "remote_exposure", "auth", "tls"}
        ),
    ),
    (
        ("services", "management_api", "auth"),
        frozenset(
            {
                "mode",
                "tokens",
                "roles",
                "token_signing_keyring_file",
                "token_signing_keyring_env",
                "service_account_hmac_keyring_file",
                "service_account_hmac_keyring_env",
                "invitation_hmac_keyring_file",
                "invitation_hmac_keyring_env",
                "response_kek_keyring_file",
                "response_kek_keyring_env",
                "bootstrap",
                "recovery",
            }
        ),
    ),
    (
        ("services", "management_api", "tls"),
        frozenset(
            {
                "certificate_file",
                "certificate_env",
                "private_key_file",
                "private_key_env",
                "client_ca_bundle_file",
                "client_ca_bundle_env",
            }
        ),
    ),
    (
        ("services", "management_api", "auth", "bootstrap"),
        frozenset({"token_file", "token_env", "disable_after_first_cluster_admin"}),
    ),
    (
        ("services", "management_api", "auth", "recovery"),
        frozenset({"enabled", "token_file", "token_env", "loopback_only"}),
    ),
    (
        ("services", "routing_security"),
        frozenset({"hmac_keyring_file", "hmac_keyring_env"}),
    ),
)

_BACKEND_CREDENTIAL_INFRASTRUCTURE_FIELDS = frozenset(
    {"provider_kek_keyring_file", "provider_kek_keyring_env"}
)
_BACKEND_CREDENTIAL_FIELDS = frozenset(
    {"credential_adapter_id", "secret_file", "secret_env"}
)


def validate_global_structure(global_config: Mapping[str, Any]) -> None:
    """Reject fields and object shapes outside the final v0.3 global contract."""

    for segments, allowed in _GLOBAL_OBJECT_FIELDS:
        value = _value_at(global_config, segments)
        if value is _MISSING:
            continue
        path = _path(segments)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        unknown = sorted(set(value) - allowed)
        if unknown:
            qualified = ", ".join(f"{path}.{field}" for field in unknown)
            raise ValueError(f"unsupported fields in {path}: {qualified}")

    credentials = _value_at(global_config, ("services", "backend_credentials"))
    if credentials is _MISSING:
        return
    path = "global.services.backend_credentials"
    if not isinstance(credentials, Mapping):
        raise ValueError(f"{path} must be an object")
    for name, definition in credentials.items():
        if name in _BACKEND_CREDENTIAL_INFRASTRUCTURE_FIELDS:
            continue
        definition_path = f"{path}.{name}"
        if not isinstance(definition, Mapping):
            raise ValueError(f"{definition_path} must be an object")
        unknown = sorted(set(definition) - _BACKEND_CREDENTIAL_FIELDS)
        if unknown:
            qualified = ", ".join(f"{definition_path}.{field}" for field in unknown)
            raise ValueError(f"unsupported fields in {definition_path}: {qualified}")


def _value_at(root: Mapping[str, Any], segments: tuple[str, ...]) -> Any:
    value: Any = root
    for segment in segments:
        if not isinstance(value, Mapping) or segment not in value:
            return _MISSING
        value = value[segment]
    return value


def _path(segments: tuple[str, ...]) -> str:
    return ".".join(("global", *segments))
