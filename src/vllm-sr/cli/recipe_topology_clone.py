"""Pure clone-command and isolation contracts for Recipe topology changes."""

from __future__ import annotations

import ipaddress
import os
import sys
from typing import Any

from cli.recipe_topology_contract import (
    _DIGEST,
    _PUBLISHABLE_LISTENER_ADDRESSES,
    MANAGEMENT_CREDENTIAL_ENV,
    TopologyReconcileError,
    _valid_port,
)
from cli.recipe_topology_storage import validate_cloneable_container_snapshot


def _clone_snapshot_contract(
    snapshot: dict[str, Any], transition: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        validate_cloneable_container_snapshot(snapshot)
    except ValueError as error:
        raise TopologyReconcileError(
            f"managed {transition['service']} container is not safely cloneable: {error}"
        ) from error
    config = snapshot.get("Config") or {}
    host = snapshot.get("HostConfig") or {}
    networks = (snapshot.get("NetworkSettings") or {}).get("Networks") or {}
    image_id = _immutable_image_id(snapshot)
    return config, host, networks, image_id


def _base_clone_command(
    transition: dict[str, Any],
    snapshot: dict[str, Any],
    config: dict[str, Any],
    host: dict[str, Any],
    networks: dict[str, Any],
) -> tuple[list[str], str]:
    command = ["run", "-d", "--name", transition["name"]]
    _append_host_config(command, host)
    _append_container_identity(command, config, preserve_hostname=True)
    for exposed_port in sorted((config.get("ExposedPorts") or {}).keys()):
        command.extend(["--expose", exposed_port])
    primary_network = str(host.get("NetworkMode") or "").strip()
    # Docker reports the implicit default network mode as `default` while the
    # resulting endpoint is named `bridge`. Treat them as the same primary
    # attachment so the clone does not try to connect `bridge` a second time.
    primary_endpoint_network = (
        "bridge" if primary_network in {"", "default", "bridge"} else primary_network
    )
    if primary_network and primary_network not in {"default", "bridge"}:
        command.extend(["--network", primary_network])
        endpoint = networks.get(primary_endpoint_network) or {}
        for alias in _preserved_network_aliases(endpoint, transition, snapshot):
            command.extend(["--network-alias", alias])
    command.extend(["--volumes-from", transition["backup_name"]])
    return command, primary_endpoint_network


def _clone_port_bindings(
    host: dict[str, Any],
    listeners: list[dict[str, Any]] | None,
    management: tuple[int, int] | None,
) -> dict[str, Any]:
    bindings = host.get("PortBindings") or {}
    if listeners is not None:
        bindings = {}
        for listener in listeners:
            container_port = f"{listener['port']}/tcp"
            bindings.setdefault(container_port, []).append(
                {
                    "HostIp": _listener_host_address(listener),
                    "HostPort": str(listener["host_port"]),
                }
            )
    elif management is not None:
        bindings = _rewrite_router_management_bindings(host, management)
    return bindings


def _validate_clone_replacement(
    replacement: dict[str, Any],
    image_id: str,
    management: tuple[int, int] | None,
) -> None:
    replacement_image_id = _immutable_image_id(replacement)
    if replacement_image_id != image_id:
        raise TopologyReconcileError(
            "replacement container changed its immutable image identity"
        )
    if management is not None:
        _validate_router_management_isolation(
            replacement, management, expected_host_ip="127.0.0.1"
        )


def _preserved_network_aliases(
    endpoint: object,
    transition: dict[str, Any],
    snapshot: dict[str, Any],
) -> list[str]:
    transient_aliases = {
        transition["backup_name"],
        transition["name"],
        snapshot.get("Id", "")[:12],
    }
    if not isinstance(endpoint, dict):
        return []
    return [
        alias
        for alias in endpoint.get("Aliases") or []
        if isinstance(alias, str) and alias not in transient_aliases
    ]


def _append_container_identity(
    command: list[str], config: dict[str, Any], *, preserve_hostname: bool
) -> None:
    fields = [
        ("User", "--user"),
        ("WorkingDir", "--workdir"),
        ("StopSignal", "--stop-signal"),
        ("Domainname", "--domainname"),
    ]
    if preserve_hostname:
        fields.insert(0, ("Hostname", "--hostname"))
    for key, flag in fields:
        value = config.get(key)
        if isinstance(value, str) and value:
            command.extend([flag, value])


def _append_host_config(command: list[str], host: dict[str, Any]) -> None:
    _append_restart_policy(command, host)
    _append_ulimits(command, host)
    _append_devices(command, host)
    _append_list_host_options(command, host)
    _append_boolean_host_options(command, host)
    _append_namespace_host_options(command, host)
    _append_runtime_host_options(command, host)
    _append_log_config(command, host)


def _append_restart_policy(command: list[str], host: dict[str, Any]) -> None:
    restart = host.get("RestartPolicy") or {}
    restart_name = str(restart.get("Name") or "").strip()
    if restart_name and restart_name != "no":
        policy = restart_name
        maximum = restart.get("MaximumRetryCount")
        if restart_name == "on-failure" and isinstance(maximum, int) and maximum > 0:
            policy += f":{maximum}"
        command.extend(["--restart", policy])


def _append_ulimits(command: list[str], host: dict[str, Any]) -> None:
    for item in host.get("Ulimits") or []:
        if isinstance(item, dict) and item.get("Name"):
            command.extend(
                [
                    "--ulimit",
                    f"{item['Name']}={item.get('Soft', 0)}:{item.get('Hard', 0)}",
                ]
            )


def _append_devices(command: list[str], host: dict[str, Any]) -> None:
    for device in host.get("Devices") or []:
        if not isinstance(device, dict):
            continue
        source, target = device.get("PathOnHost"), device.get("PathInContainer")
        permissions = device.get("CgroupPermissions") or "rwm"
        if source and target:
            command.extend(["--device", f"{source}:{target}:{permissions}"])


def _append_list_host_options(command: list[str], host: dict[str, Any]) -> None:
    for key, flag in (
        ("GroupAdd", "--group-add"),
        ("CapAdd", "--cap-add"),
        ("CapDrop", "--cap-drop"),
        ("SecurityOpt", "--security-opt"),
        ("Dns", "--dns"),
        ("DnsSearch", "--dns-search"),
        ("ExtraHosts", "--add-host"),
    ):
        for value in host.get(key) or []:
            command.extend([flag, str(value)])


def _append_boolean_host_options(command: list[str], host: dict[str, Any]) -> None:
    if host.get("Privileged"):
        command.append("--privileged")
    if host.get("ReadonlyRootfs"):
        command.append("--read-only")


def _append_namespace_host_options(command: list[str], host: dict[str, Any]) -> None:
    for key, flag in (("IpcMode", "--ipc"), ("PidMode", "--pid")):
        value = str(host.get(key) or "").strip()
        if value:
            command.extend([flag, value])
    if isinstance(host.get("ShmSize"), int) and host["ShmSize"] > 0:
        command.extend(["--shm-size", str(host["ShmSize"])])


def _append_runtime_host_options(command: list[str], host: dict[str, Any]) -> None:
    runtime = str(host.get("Runtime") or "").strip()
    if runtime:
        command.extend(["--runtime", runtime])
    cgroupns = str(host.get("CgroupnsMode") or "").strip()
    if cgroupns:
        command.extend(["--cgroupns", cgroupns])


def _append_log_config(command: list[str], host: dict[str, Any]) -> None:
    log_config = host.get("LogConfig") or {}
    log_driver = str(log_config.get("Type") or "").strip()
    if log_driver:
        command.extend(["--log-driver", log_driver])
        for key, value in sorted((log_config.get("Config") or {}).items()):
            command.extend(["--log-opt", f"{key}={value}"])


def _rewrite_router_management_bindings(
    host: dict[str, Any], management: tuple[int, int]
) -> dict[str, Any]:
    _validate_router_management_host_mode(host)
    bindings = _decode_router_port_bindings(host.get("PortBindings"), "configured")
    container_port, host_port = management
    management_port = f"{container_port}/tcp"
    rendered_host_port = str(host_port)
    rewritten: dict[str, Any] = {}
    for port, published in bindings.items():
        if port == management_port:
            continue
        if any(binding["HostPort"] == rendered_host_port for binding in published):
            raise TopologyReconcileError(
                "Router management host port is shared with another container port"
            )
        rewritten[port] = published
    rewritten[management_port] = [
        {"HostIp": "127.0.0.1", "HostPort": rendered_host_port}
    ]
    return rewritten


def _validate_router_management_isolation(
    snapshot: dict[str, Any],
    management: tuple[int, int],
    *,
    expected_host_ip: str | None = None,
) -> None:
    host = snapshot.get("HostConfig")
    network = snapshot.get("NetworkSettings")
    if not isinstance(host, dict) or not isinstance(network, dict):
        raise TopologyReconcileError(
            "Router management port isolation inspection is invalid"
        )
    _validate_router_management_host_mode(host)
    configured = _decode_router_port_bindings(host.get("PortBindings"), "configured")
    actual = _decode_router_port_bindings(network.get("Ports"), "actual")
    configured_binding = _exact_router_management_binding(
        configured, management, expected_host_ip=expected_host_ip
    )
    actual_binding = _exact_router_management_binding(
        actual, management, expected_host_ip=expected_host_ip
    )
    if configured_binding != actual_binding:
        raise TopologyReconcileError(
            "configured and actual Router management publications do not match"
        )


def _validate_router_management_host_mode(host: dict[str, Any]) -> None:
    network_mode = host.get("NetworkMode")
    publish_all_ports = host.get("PublishAllPorts")
    if not isinstance(network_mode, str) or not isinstance(publish_all_ports, bool):
        raise TopologyReconcileError(
            "Router management port isolation inspection is invalid"
        )
    normalized_mode = network_mode.strip().lower()
    if normalized_mode == "host":
        raise TopologyReconcileError(
            "host network mode bypasses Router management loopback isolation"
        )
    if normalized_mode.startswith("container:"):
        raise TopologyReconcileError(
            "container network mode bypasses Router management loopback isolation"
        )
    if publish_all_ports:
        raise TopologyReconcileError(
            "PublishAllPorts bypasses Router management loopback isolation"
        )


def _decode_router_port_bindings(
    value: object, label: str
) -> dict[str, list[dict[str, str]]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TopologyReconcileError(
            f"{label} Router management port bindings are invalid"
        )
    decoded: dict[str, list[dict[str, str]]] = {}
    for port, published in value.items():
        if not isinstance(port, str) or not port:
            raise TopologyReconcileError(
                f"{label} Router management port bindings are invalid"
            )
        if published is None:
            decoded[port] = []
            continue
        if not isinstance(published, list):
            raise TopologyReconcileError(
                f"{label} Router management port bindings are invalid"
            )
        entries: list[dict[str, str]] = []
        for binding in published:
            if (
                not isinstance(binding, dict)
                or set(binding) != {"HostIp", "HostPort"}
                or not isinstance(binding.get("HostIp"), str)
                or not isinstance(binding.get("HostPort"), str)
                or not binding["HostPort"].isdigit()
                or not _valid_port(int(binding["HostPort"]))
            ):
                raise TopologyReconcileError(
                    f"{label} Router management port bindings are invalid"
                )
            entries.append(binding)
        decoded[port] = entries
    return decoded


def _exact_router_management_binding(
    bindings: dict[str, list[dict[str, str]]],
    management: tuple[int, int],
    *,
    expected_host_ip: str | None,
) -> tuple[str, int]:
    container_port, host_port = management
    management_port = f"{container_port}/tcp"
    published = bindings.get(management_port)
    if published is None or len(published) != 1:
        raise TopologyReconcileError(
            "expected exactly one Router management port binding"
        )
    binding = published[0]
    if int(binding["HostPort"]) != host_port:
        raise TopologyReconcileError(
            "Router management host port does not match the managed stack"
        )
    try:
        host_ip = ipaddress.ip_address(binding["HostIp"].strip())
    except ValueError as error:
        raise TopologyReconcileError(
            "Router management host address is not loopback"
        ) from error
    if not host_ip.is_loopback:
        raise TopologyReconcileError("Router management host address is not loopback")
    if expected_host_ip is not None and host_ip != ipaddress.ip_address(
        expected_host_ip
    ):
        raise TopologyReconcileError(
            "replacement Router management host address is not the managed loopback"
        )
    rendered_host_port = str(host_port)
    for port, entries in bindings.items():
        if port == management_port:
            continue
        if any(binding["HostPort"] == rendered_host_port for binding in entries):
            raise TopologyReconcileError(
                "Router management host port is shared with another container port"
            )
    return str(host_ip), host_port


def _append_port_bindings(command: list[str], bindings: dict[str, Any]) -> None:
    for container_port, host_bindings in sorted(bindings.items()):
        for binding in host_bindings or []:
            host_ip = str((binding or {}).get("HostIp") or "")
            host_port = str((binding or {}).get("HostPort") or "")
            if not host_port.isdigit():
                raise TopologyReconcileError("preserved port binding is invalid")
            mapping = f"{host_port}:{container_port}"
            if host_ip:
                rendered_host_ip = f"[{host_ip}]" if ":" in host_ip else host_ip
                mapping = f"{rendered_host_ip}:{mapping}"
            command.extend(["-p", mapping])


def _listener_host_address(listener: dict[str, Any]) -> str:
    address = str(listener.get("address") or "").strip()
    if address in _PUBLISHABLE_LISTENER_ADDRESSES:
        return address
    raise TopologyReconcileError(
        "listener address must be an explicit wildcard or loopback IP for host publication"
    )


def _normalized_environment(values: list[object], credential: str | None) -> list[str]:
    environment: dict[str, str] = {}
    for item in values:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if not key or "\n" in key or "\n" in value or "\x00" in item:
            raise TopologyReconcileError("preserved container environment is invalid")
        environment[key] = value
    if credential is not None:
        environment.pop(MANAGEMENT_CREDENTIAL_ENV, None)
        if credential:
            environment[MANAGEMENT_CREDENTIAL_ENV] = credential
    return [f"{key}={environment[key]}" for key in sorted(environment)]


def _open_environment_memfd(values: list[str]) -> int:
    """Return a rewound anonymous fd suitable for Docker's --env-file."""

    create_memfd = getattr(os, "memfd_create", None)
    if not sys.platform.startswith("linux") or create_memfd is None:
        raise TopologyReconcileError(
            "managed topology reconciliation requires Linux memfd support"
        )
    descriptor = -1
    try:
        descriptor = create_memfd(
            "vllm-sr-recipe-env", flags=getattr(os, "MFD_CLOEXEC", 0)
        )
        os.fchmod(descriptor, 0o600)
        encoded = ("\n".join(values) + "\n").encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short write to environment memfd")
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except (AttributeError, OSError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise TopologyReconcileError(
            "managed container environment could not be staged anonymously"
        ) from error


def _immutable_image_id(snapshot: dict[str, Any]) -> str:
    image_id = snapshot.get("Image")
    if not isinstance(image_id, str) or not _DIGEST.fullmatch(image_id):
        raise TopologyReconcileError(
            "preserved container immutable image identity is invalid"
        )
    return image_id
