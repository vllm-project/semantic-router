"""Stable Router backend-dispatch contract for the generated Envoy data path."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
import re
import uuid

from cli.control_plane_deployment import (
    MANAGED_MODE,
    control_plane_mode,
    managed_store_references,
)
from cli.models import UserConfig
from cli.validation_error import ValidationError

BACKEND_DISPATCH_ADDRESS_ENV = "ENVOY_BACKEND_DISPATCH_ADDRESS"

# These headers are produced only inside the trusted Router data path. They
# must be absent when ExtProc observes a public request.
INTERNAL_REQUEST_HEADERS = (
    "x-vsr-dispatch-capability",
    "x-vsr-destination-endpoint",
    "x-authz-user-id",
    "x-authz-user-groups",
    "x-authz-team-id",
    "x-authz-tenant-id",
    "x-vllm-sr-api-key-id",
    "x-vllm-sr-user-id",
    "x-vllm-sr-team-id",
    "x-vsr-internal-auth",
    "x-vsr-looper-request",
    "x-vsr-looper-iteration",
    "x-vsr-looper-decision",
    "x-vsr-fusion-depth",
    "x-vsr-selected-recipe",
    "x-vsr-routing-namespace",
    "x-vsr-routing-quota-partition",
    "x-vsr-routing-publication",
    "x-vsr-routing-runtime-epoch",
    "x-vsr-routing-snapshot-revision",
    "x-vsr-routing-digest",
)

_MAX_PORT = 65_535
_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class BackendDispatchEndpoint:
    """Exact private Router endpoint consumed by the Envoy gateway."""

    address: str
    port: int
    cluster_type: str
    address_is_domain: bool


def validate_envoy_dispatch_contract(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate control-plane isolation and the common Envoy dispatch boundary."""

    try:
        mode = control_plane_mode(config)
    except ValueError as error:
        return [ValidationError(str(error), field="global.control_plane.mode")]
    errors: list[ValidationError] = []
    if mode == MANAGED_MODE:
        _validate_managed_authority_boundary(config, errors)
        _validate_managed_stores(config, errors)
        _validate_public_namespace(config, errors)
    else:
        errors.extend(_validate_standalone_boundary(config))
    dispatch = _backend_dispatch_config(config, errors)
    if dispatch is not None:
        _validate_bind_address(dispatch, errors)
        _validate_port(dispatch, errors)
    return errors


def _validate_standalone_boundary(config: UserConfig) -> list[ValidationError]:
    """Keep managed identity state and its public namespace out of standalone."""

    errors: list[ValidationError] = []
    global_config = config.global_ or {}
    control_plane = global_config.get("control_plane") or {}
    if isinstance(control_plane, dict) and control_plane.get("public_namespace_id"):
        errors.append(
            ValidationError(
                "public namespace is available only in managed routing-only mode",
                field="global.control_plane.public_namespace_id",
            )
        )
    stores = global_config.get("stores") or {}
    if isinstance(stores, dict):
        for name in ("access", "access_runtime"):
            if stores.get(name) is not None:
                errors.append(
                    ValidationError(
                        "standalone mode does not use managed access stores",
                        field=f"global.stores.{name}",
                    )
                )
    services = global_config.get("services") or {}
    access = {}
    if isinstance(services, dict):
        access = services.get("access") or {}
    if isinstance(access, dict) and access.get("enabled") is True:
        errors.append(
            ValidationError(
                "standalone mode does not enable managed access",
                field="global.services.access.enabled",
            )
        )
    return errors


def _validate_managed_authority_boundary(
    config: UserConfig, errors: list[ValidationError]
) -> None:
    """Keep mutable routing resources out of managed bootstrap YAML."""

    if config.models:
        errors.append(
            ValidationError(
                "managed mode obtains Models through the Management API",
                field="models",
            )
        )
    if config.recipes:
        errors.append(
            ValidationError(
                "managed mode obtains Recipes through the Management API",
                field="recipes",
            )
        )
    if config.entrypoints:
        errors.append(
            ValidationError(
                "managed mode obtains Entrypoints through the Management API",
                field="entrypoints",
            )
        )


def _validate_managed_stores(config: UserConfig, errors: list[ValidationError]) -> None:
    global_config = config.global_ or {}
    stores = global_config.get("stores")
    if not isinstance(stores, dict):
        errors.append(
            ValidationError(
                "managed mode requires a stores mapping",
                field="global.stores",
            )
        )
        return
    access = stores.get("access")
    valid = True
    if not isinstance(access, dict) or access.get("type") != "postgres":
        valid = False
        errors.append(
            ValidationError(
                "managed mode requires PostgreSQL desired state",
                field="global.stores.access.type",
            )
        )
    runtime = stores.get("access_runtime")
    if not isinstance(runtime, dict) or runtime.get("type") != "redis":
        valid = False
        errors.append(
            ValidationError(
                "managed mode requires a Valkey runtime store",
                field="global.stores.access_runtime.type",
            )
        )
    if not valid:
        return
    try:
        managed_store_references({"global": global_config})
    except ValueError as error:
        errors.append(ValidationError(str(error), field="global.stores"))


def _validate_public_namespace(
    config: UserConfig, errors: list[ValidationError]
) -> None:
    global_config = config.global_ or {}
    control_plane = global_config.get("control_plane") or {}
    services = global_config.get("services") or {}
    access = {}
    if isinstance(services, dict):
        access = services.get("access") or {}
    if not isinstance(access, dict):
        access = {}
    access_enabled = bool(access.get("enabled", False))
    namespace = control_plane.get("public_namespace_id", "")
    if access_enabled:
        if namespace:
            errors.append(
                ValidationError(
                    "managed access derives namespace from the authenticated key",
                    field="global.control_plane.public_namespace_id",
                )
            )
        return
    try:
        parsed = uuid.UUID(namespace)
    except (AttributeError, TypeError, ValueError):
        parsed = None
    if parsed is None or str(parsed) != namespace:
        errors.append(
            ValidationError(
                "managed routing-only requires one canonical namespace UUID",
                field="global.control_plane.public_namespace_id",
            )
        )


def resolve_backend_dispatch_endpoint(config: UserConfig) -> BackendDispatchEndpoint:
    """Resolve Envoy's one stable Router upstream in either control-plane mode."""

    errors = validate_envoy_dispatch_contract(config)
    if errors:
        raise ValueError("; ".join(str(error) for error in errors))
    dispatch = (config.global_ or {})["services"]["backend_dispatch"]
    bind_address = dispatch["bind_address"]
    port = dispatch["port"]

    configured_override = os.getenv(BACKEND_DISPATCH_ADDRESS_ENV)
    if configured_override is None:
        address = bind_address
        if ipaddress.ip_address(address).is_unspecified:
            raise ValueError(
                "global.services.backend_dispatch.bind_address is a bind-only "
                f"address; set {BACKEND_DISPATCH_ADDRESS_ENV} for the Envoy-to-Router network"
            )
    else:
        address = _validated_upstream_address(configured_override)

    address_is_domain = not _is_ip_address(address)
    return BackendDispatchEndpoint(
        address=address,
        port=port,
        cluster_type="LOGICAL_DNS" if address_is_domain else "STATIC",
        address_is_domain=address_is_domain,
    )


def validate_networked_backend_dispatch(config: UserConfig, target: str) -> None:
    """Require a listener reachable from a separate gateway container or Pod."""

    errors = validate_envoy_dispatch_contract(config)
    if errors:
        raise ValueError("; ".join(str(error) for error in errors))
    dispatch = (config.global_ or {})["services"]["backend_dispatch"]
    address = ipaddress.ip_address(dispatch["bind_address"])
    if not address.is_unspecified:
        raise ValueError(
            "global.services.backend_dispatch.bind_address must be a wildcard "
            f"address for the {target} split data path"
        )


def _backend_dispatch_config(
    config: UserConfig, errors: list[ValidationError]
) -> dict | None:
    global_config = config.global_ or {}
    services = global_config.get("services")
    if not isinstance(services, dict):
        errors.append(
            ValidationError(
                "Envoy requires a services mapping",
                field="global.services",
            )
        )
        return None
    dispatch = services.get("backend_dispatch")
    if not isinstance(dispatch, dict):
        errors.append(
            ValidationError(
                "Envoy requires a backend dispatch mapping",
                field="global.services.backend_dispatch",
            )
        )
        return None
    return dispatch


def _validate_bind_address(dispatch: dict, errors: list[ValidationError]) -> None:
    address = dispatch.get("bind_address")
    if not isinstance(address, str) or address != address.strip():
        errors.append(
            ValidationError(
                "bind_address must be a canonical IP address",
                field="global.services.backend_dispatch.bind_address",
            )
        )
        return
    try:
        ipaddress.ip_address(address)
    except ValueError:
        errors.append(
            ValidationError(
                "bind_address must be a canonical IP address",
                field="global.services.backend_dispatch.bind_address",
            )
        )


def _validate_port(dispatch: dict, errors: list[ValidationError]) -> None:
    port = dispatch.get("port")
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= _MAX_PORT
    ):
        errors.append(
            ValidationError(
                "port must be an integer between 1 and 65535",
                field="global.services.backend_dispatch.port",
            )
        )


def _validated_upstream_address(raw_address: str) -> str:
    if raw_address != raw_address.strip() or not raw_address:
        raise ValueError(
            f"{BACKEND_DISPATCH_ADDRESS_ENV} must be an IP address or DNS name"
        )
    if _is_ip_address(raw_address):
        if ipaddress.ip_address(raw_address).is_unspecified:
            raise ValueError(
                f"{BACKEND_DISPATCH_ADDRESS_ENV} must identify a routable Router endpoint"
            )
        return raw_address

    if len(raw_address) > 253:
        raise ValueError(
            f"{BACKEND_DISPATCH_ADDRESS_ENV} must be an IP address or DNS name"
        )
    labels = raw_address.rstrip(".").split(".")
    if not labels or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError(
            f"{BACKEND_DISPATCH_ADDRESS_ENV} must be an IP address or DNS name"
        )
    return raw_address


def _is_ip_address(address: str) -> bool:
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return False
