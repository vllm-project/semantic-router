"""Network-isolation checks for locally managed storage containers."""

from __future__ import annotations

import ipaddress


def validate_storage_port_isolation(inspection: object) -> None:
    """Reject managed storage networking that can expose host ports publicly."""

    if not isinstance(inspection, dict) or set(inspection) != {
        "network_mode",
        "publish_all_ports",
        "configured",
        "actual",
    }:
        raise ValueError("storage port isolation inspection is invalid")
    _validate_storage_network_mode(
        inspection["network_mode"], inspection["publish_all_ports"]
    )
    for source in ("configured", "actual"):
        _validate_storage_port_bindings(inspection[source], source)


def _validate_storage_network_mode(
    network_mode: object, publish_all_ports: object
) -> None:
    if not isinstance(network_mode, str) or not isinstance(publish_all_ports, bool):
        raise ValueError("storage port isolation inspection is invalid")
    normalized_mode = network_mode.strip().lower()
    if normalized_mode == "host":
        raise ValueError("host network mode bypasses loopback port isolation")
    if normalized_mode.startswith("container:"):
        raise ValueError("container network mode bypasses loopback port isolation")
    if publish_all_ports:
        raise ValueError("PublishAllPorts bypasses explicit loopback bindings")


def _validate_storage_port_bindings(ports: object, source: str) -> None:
    if ports is None:
        return
    if not isinstance(ports, dict):
        raise ValueError(f"inspection returned invalid {source} port bindings")
    for container_port, published in ports.items():
        if not isinstance(container_port, str) or not container_port:
            raise ValueError(f"inspection returned invalid {source} port bindings")
        if published is None:
            continue
        if not isinstance(published, list):
            raise ValueError(f"inspection returned invalid {source} port bindings")
        for binding in published:
            _validate_loopback_storage_binding(binding, source, container_port)


def _validate_loopback_storage_binding(
    binding: object, source: str, container_port: str
) -> None:
    if not isinstance(binding, dict) or set(binding) != {"HostIp", "HostPort"}:
        raise ValueError(f"inspection returned invalid {source} port bindings")
    host_ip = binding["HostIp"]
    host_port = binding["HostPort"]
    if (
        not isinstance(host_ip, str)
        or not isinstance(host_port, str)
        or not host_port.isdigit()
    ):
        raise ValueError(f"inspection returned invalid {source} port bindings")
    try:
        is_loopback = ipaddress.ip_address(host_ip.strip()).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError(
            f"{source} published port {container_port} uses "
            f"non-loopback host address {host_ip!r}"
        )
