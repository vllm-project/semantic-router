"""Stable Router backend-dispatch contract for the generated Envoy data path."""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass

from cli.config_contract import DEFAULT_BACKEND_DISPATCH
from cli.control_plane_deployment import (
    control_plane_store_references,
    runtime_capabilities,
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
_MAX_DNS_NAME_LENGTH = 253
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
    """Validate derived capabilities and the common Envoy dispatch boundary."""

    try:
        capabilities = runtime_capabilities(config)
    except ValueError as error:
        return [ValidationError(str(error), field="global")]
    errors: list[ValidationError] = []
    if capabilities.durable_management:
        _validate_control_plane_stores(config, errors)
    dispatch = _backend_dispatch_config(config, errors)
    if dispatch is not None:
        _validate_bind_address(dispatch, errors)
        _validate_port(dispatch, errors)
    return errors


def _validate_control_plane_stores(
    config: UserConfig, errors: list[ValidationError]
) -> None:
    global_config = config.global_ or {}
    stores = global_config.get("stores")
    if not isinstance(stores, dict):
        errors.append(
            ValidationError(
                "Management capabilities require a stores mapping",
                field="global.stores",
            )
        )
        return
    management = stores.get("management")
    if not isinstance(management, dict) or not isinstance(
        management.get("postgres"), dict
    ):
        errors.append(
            ValidationError(
                "Management capabilities require PostgreSQL desired state",
                field="global.stores.management.postgres",
            )
        )
        return
    try:
        control_plane_store_references({"global": global_config})
    except ValueError as error:
        errors.append(ValidationError(str(error), field="global.stores"))


def resolve_backend_dispatch_endpoint(config: UserConfig) -> BackendDispatchEndpoint:
    """Resolve Envoy's one stable Router upstream."""

    errors = validate_envoy_dispatch_contract(config)
    if errors:
        raise ValueError("; ".join(str(error) for error in errors))
    dispatch = _backend_dispatch_config(config, []) or dict(DEFAULT_BACKEND_DISPATCH)
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
    dispatch = _backend_dispatch_config(config, []) or dict(DEFAULT_BACKEND_DISPATCH)
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
    if services is None:
        return dict(DEFAULT_BACKEND_DISPATCH)
    if not isinstance(services, dict):
        errors.append(
            ValidationError("services must be a mapping", field="global.services")
        )
        return None
    dispatch = services.get("backend_dispatch")
    if dispatch is None:
        return dict(DEFAULT_BACKEND_DISPATCH)
    if not isinstance(dispatch, dict):
        errors.append(
            ValidationError(
                "backend_dispatch must be a mapping",
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

    if len(raw_address) > _MAX_DNS_NAME_LENGTH:
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
