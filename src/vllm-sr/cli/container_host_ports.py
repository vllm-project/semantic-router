"""Validated host-port publication policy for the local container runtime."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cli.consts import DEFAULT_ENVOY_PORT
from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)

DEFAULT_INTERNAL_BIND_ADDRESS = "127.0.0.1"
INTERNAL_BIND_ADDRESS_ENV = "VLLM_SR_INTERNAL_BIND_ADDRESS"
MAX_TCP_PORT = 65535

_STACK_HOST_PORT_FIELDS = (
    ("router extproc", "router_port"),
    ("router metrics", "metrics_port"),
    ("dashboard", "dashboard_port"),
    ("router API", "api_port"),
    ("Fleet Sim", "fleet_sim_port"),
    ("Jaeger OTLP", "jaeger_otlp_port"),
    ("Jaeger UI", "jaeger_ui_port"),
    ("Prometheus", "prometheus_port"),
    ("Grafana", "grafana_port"),
    ("Redis", "redis_port"),
    ("Postgres", "postgres_port"),
    ("Milvus", "milvus_port"),
)


@dataclass(frozen=True)
class ListenerHostPublication:
    """One normalized Envoy listener publication."""

    name: str
    bind_address: str
    host_port: int
    container_port: int


@dataclass(frozen=True)
class HostPortPublicationPlan:
    """Immutable preflight result shared by every startup stage."""

    internal_bind_address: str
    listeners: tuple[ListenerHostPublication, ...]

    @property
    def primary_listener_port(self) -> int:
        return self.listeners[0].container_port


def normalize_bind_address(value: Any, *, source: str) -> str:
    """Return a canonical IP literal suitable for container port publishing."""
    raw_value = value.strip() if isinstance(value, str) else ""
    if not raw_value or "%" in raw_value:
        raise ValueError(
            f"{source} must be an IPv4 or IPv6 address literal; got {value!r}"
        )

    try:
        address = ipaddress.ip_address(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{source} must be an IPv4 or IPv6 address literal; got {value!r}"
        ) from exc

    if address.is_multicast:
        raise ValueError(f"{source} cannot be a multicast address; got {value!r}")
    return str(address)


def resolve_internal_bind_address() -> str:
    """Resolve the host bind used by internal and supporting services."""
    configured = os.getenv(INTERNAL_BIND_ADDRESS_ENV)
    if configured is None:
        return DEFAULT_INTERNAL_BIND_ADDRESS

    bind_address = normalize_bind_address(configured, source=INTERNAL_BIND_ADDRESS_ENV)
    if not ipaddress.ip_address(bind_address).is_loopback:
        log.warning(
            "%s=%s publishes internal, data, and observability ports beyond "
            "the loopback interface. Only use this on a trusted network with "
            "host firewall controls.",
            INTERNAL_BIND_ADDRESS_ENV,
            bind_address,
        )
    return bind_address


def resolve_effective_internal_bind_address(
    preflight_address: str | None = None,
) -> str:
    """Use an immutable preflight value, or resolve policy for direct callers."""
    if preflight_address is None:
        return resolve_internal_bind_address()
    return normalize_bind_address(
        preflight_address, source="preflight internal bind address"
    )


def build_host_port_publication_plan(
    stack_layout: RuntimeStackLayout,
    listeners: Sequence[Mapping[str, Any]],
) -> HostPortPublicationPlan:
    """Validate and normalize every host publication before startup mutates state."""
    internal_bind_address = resolve_internal_bind_address()
    internal_publications: list[tuple[str, str, int]] = []
    for label, field_name in _STACK_HOST_PORT_FIELDS:
        port = _validate_port(
            getattr(stack_layout, field_name), source=f"{label} host port"
        )
        internal_publications.append((label, internal_bind_address, port))

    normalized_listeners = tuple(
        _normalize_listener_publication(listener, index, stack_layout.port_offset)
        for index, listener in enumerate(listeners)
    )
    if not normalized_listeners:
        raise ValueError(
            "At least one listener is required for local container startup"
        )
    _validate_unique_listener_container_ports(normalized_listeners)
    _validate_host_publication_conflicts(
        (
            *internal_publications,
            *(
                (f"listener[{index}]", listener.bind_address, listener.host_port)
                for index, listener in enumerate(normalized_listeners)
            ),
        )
    )

    return HostPortPublicationPlan(
        internal_bind_address=internal_bind_address,
        listeners=normalized_listeners,
    )


def format_port_mapping(bind_address: str, host_port: int, container_port: int) -> str:
    """Format an explicitly bound Docker/Podman publish specification."""
    normalized = normalize_bind_address(bind_address, source="host bind address")
    normalized_host_port = _validate_port(host_port, source="host port")
    normalized_container_port = _validate_port(container_port, source="container port")
    formatted_address = f"[{normalized}]" if ":" in normalized else normalized
    return f"{formatted_address}:{normalized_host_port}:{normalized_container_port}"


def _normalize_listener_publication(
    listener: Mapping[str, Any], index: int, port_offset: int
) -> ListenerHostPublication:
    if not isinstance(listener, Mapping):
        raise ValueError(f"listener[{index}] must be a mapping")

    name = str(listener.get("name") or f"listener[{index}]")
    bind_address = normalize_bind_address(
        listener.get("address"), source=f"listener {name!r} address"
    )
    container_port = _validate_port(
        listener.get("port"), source=f"listener {name!r} container port"
    )
    host_port = _validate_port(
        container_port + port_offset,
        source=f"listener {name!r} host port after offset {port_offset}",
    )
    return ListenerHostPublication(
        name=name,
        bind_address=bind_address,
        host_port=host_port,
        container_port=container_port,
    )


def _validate_port(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{source} must be an integer in range 1..65535; got {value!r}"
        )
    if not 1 <= value <= MAX_TCP_PORT:
        raise ValueError(f"{source} must be in range 1..65535; got {value}")
    return value


def _validate_unique_listener_container_ports(
    listeners: Sequence[ListenerHostPublication],
) -> None:
    """Reject split listeners that collide after binding to 0.0.0.0."""
    owner_by_port = {DEFAULT_ENVOY_PORT: "Envoy admin"}
    for index, listener in enumerate(listeners):
        label = f"listener[{index}]"
        previous_label = owner_by_port.get(listener.container_port)
        if previous_label is not None:
            raise ValueError(
                "split Envoy listener container port conflict: "
                f"{previous_label} and {label} both use "
                f"port {listener.container_port}"
            )
        owner_by_port[listener.container_port] = label


def _validate_host_publication_conflicts(
    publications: Sequence[tuple[str, str, int]],
) -> None:
    """Reject host publications whose address/port bindings overlap."""
    for index, (label, address, port) in enumerate(publications):
        for other_label, other_address, other_port in publications[:index]:
            if port != other_port or not _bind_addresses_overlap(
                address, other_address
            ):
                continue
            raise ValueError(
                "host port publication conflict: "
                f"{other_label} ({other_address}:{other_port}) overlaps "
                f"{label} ({address}:{port})"
            )


def _bind_addresses_overlap(left: str, right: str) -> bool:
    left_address = ipaddress.ip_address(left)
    right_address = ipaddress.ip_address(right)
    if left_address == right_address:
        return True
    # An unspecified bind may accept traffic for every local interface. Treat
    # it as overlapping even across IP versions because dual-stack behavior is
    # host/runtime dependent and preflight must fail deterministically.
    return left_address.is_unspecified or right_address.is_unspecified
