"""Data-preserving contracts for managed storage container repair."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, fields
from typing import Any

from cli.recipe_topology_host_options import _reject_unreplayed_host_options

_DATA_DESTINATION = {
    "redis": "/data",
    "postgres": "/var/lib/postgresql/data",
    "milvus": "/var/lib/milvus",
}
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class StorageCloneContract:
    """Runtime state that a repaired storage container must retain exactly."""

    image_id: str
    process: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    hostname: str
    user: str
    working_directory: str
    stop_signal: str
    labels: tuple[tuple[str, str], ...]
    mounts: tuple[tuple[object, ...], ...]
    port_bindings: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    restart_policy: tuple[str, int]
    network_mode: str
    networks: tuple[tuple[str, tuple[str, ...]], ...]
    host_options: tuple[tuple[str, object], ...]


def validate_storage_port_isolation(inspection: object) -> None:
    """Reject managed storage networking that can expose host ports publicly."""

    if not isinstance(inspection, dict) or set(inspection) != {
        "network_mode",
        "publish_all_ports",
        "configured",
        "actual",
    }:
        raise ValueError("storage port isolation inspection is invalid")
    network_mode = inspection["network_mode"]
    publish_all_ports = inspection["publish_all_ports"]
    _validate_storage_network_mode(network_mode, publish_all_ports)
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


def storage_clone_contract(
    snapshot: dict[str, Any],
    service: str,
    *,
    transient_names: set[str] | None = None,
) -> StorageCloneContract:
    """Return the secret-safe, comparable repair contract for one snapshot."""

    if service not in _DATA_DESTINATION:
        raise ValueError("unsupported managed storage service")
    config = _mapping(snapshot.get("Config"), "container config")
    host = _mapping(snapshot.get("HostConfig"), "container host config")
    network_settings = _mapping(
        snapshot.get("NetworkSettings"), "container network settings"
    )
    validate_cloneable_container_snapshot(snapshot)

    image_id = snapshot.get("Image")
    if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("container immutable image identity is missing or invalid")

    mounts = _mount_contract(snapshot.get("Mounts"), service)
    environment = _environment_contract(config.get("Env"))
    labels = _string_mapping_contract(config.get("Labels"), "container labels")
    process = _string_sequence(config.get("Entrypoint"), "container entrypoint") + (
        _string_sequence(config.get("Cmd"), "container command")
    )
    ports = _port_binding_contract(host.get("PortBindings"))
    restart = _restart_policy_contract(host.get("RestartPolicy"))
    networks = _network_contract(
        network_settings.get("Networks"),
        snapshot,
        transient_names=transient_names or set(),
    )
    host_options = _host_options_contract(host)

    return StorageCloneContract(
        image_id=image_id,
        process=process,
        environment=environment,
        hostname=_optional_string(config.get("Hostname"), "container hostname"),
        user=_optional_string(config.get("User"), "container user"),
        working_directory=_optional_string(
            config.get("WorkingDir"), "container working directory"
        ),
        stop_signal=_optional_string(config.get("StopSignal"), "container stop signal"),
        labels=labels,
        mounts=mounts,
        port_bindings=ports,
        restart_policy=restart,
        network_mode=_optional_string(host.get("NetworkMode"), "network mode"),
        networks=networks,
        host_options=host_options,
    )


def validate_cloneable_container_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail before mutation when a container option cannot be replayed exactly."""

    config = _mapping(snapshot.get("Config"), "container config")
    host = _mapping(snapshot.get("HostConfig"), "container host config")
    network_settings = _mapping(
        snapshot.get("NetworkSettings"), "container network settings"
    )
    _validate_cloneable_config(config)
    _validate_cloneable_mounts(snapshot.get("Mounts"))
    _network_contract(network_settings.get("Networks"), snapshot, transient_names=set())
    _port_binding_contract(host.get("PortBindings"))
    _restart_policy_contract(host.get("RestartPolicy"))
    _optional_string(host.get("NetworkMode"), "network mode")
    _host_options_contract(host)


def _validate_cloneable_config(config: dict[str, Any]) -> None:
    _environment_contract(config.get("Env"))
    _string_mapping_contract(config.get("Labels"), "container labels")
    _string_sequence(config.get("Entrypoint"), "container entrypoint")
    _string_sequence(config.get("Cmd"), "container command")
    _validate_cloneable_config_strings(config)
    _reject_uncloneable_config_options(config)
    _empty_object_mapping(config.get("ExposedPorts"), "container exposed ports")
    _empty_object_mapping(config.get("Volumes"), "container volume declarations")
    _reject_unknown_config_options(config)


def _validate_cloneable_config_strings(config: dict[str, Any]) -> None:
    for key, label in (
        ("Hostname", "container hostname"),
        ("Domainname", "container domain name"),
        ("User", "container user"),
        ("WorkingDir", "container working directory"),
        ("StopSignal", "container stop signal"),
    ):
        _optional_string(config.get(key), label)


def _reject_uncloneable_config_options(config: dict[str, Any]) -> None:
    if config.get("Healthcheck") not in (None, {}):
        raise ValueError("custom Healthcheck is not safely cloneable")
    for key in ("Tty", "OpenStdin", "StdinOnce", "NetworkDisabled", "AttachStdin"):
        if config.get(key) not in (None, False):
            raise ValueError(f"non-default {key} is not safely cloneable")
    # `docker run -d` normally reports both output streams as attached. False
    # cannot be reproduced by the clone command and therefore fails closed.
    for key in ("AttachStdout", "AttachStderr"):
        if config.get(key) not in (None, True):
            raise ValueError(f"non-default {key} is not safely cloneable")
    # Docker exposes StopTimeout as an optional value. An explicit zero is a
    # real override, not proof that the option was omitted, and the clone path
    # does not currently replay it.
    if config.get("StopTimeout") is not None:
        raise ValueError("non-default StopTimeout is not safely cloneable")
    for key in ("OnBuild", "Shell"):
        if config.get(key) not in (None, []):
            raise ValueError(f"non-default {key} is not safely cloneable")
    if config.get("MacAddress") not in (None, ""):
        raise ValueError("non-default MacAddress is not safely cloneable")
    if config.get("ArgsEscaped") not in (None, False):
        raise ValueError("non-default ArgsEscaped is not safely cloneable")


def _reject_unknown_config_options(config: dict[str, Any]) -> None:
    covered_config_fields = {
        "ArgsEscaped",
        "AttachStderr",
        "AttachStdin",
        "AttachStdout",
        "Cmd",
        "Domainname",
        "Entrypoint",
        "Env",
        "ExposedPorts",
        "Healthcheck",
        "Hostname",
        "Image",
        "Labels",
        "MacAddress",
        "NetworkDisabled",
        "OnBuild",
        "OpenStdin",
        "Shell",
        "StdinOnce",
        "StopSignal",
        "StopTimeout",
        "Tty",
        "User",
        "Volumes",
        "WorkingDir",
    }
    for key, value in config.items():
        if key not in covered_config_fields and value not in (
            None,
            False,
            0,
            "",
            [],
            {},
        ):
            raise ValueError(f"non-default Config.{key} is not safely cloneable")


def assert_storage_contract_preserved(
    expected: StorageCloneContract, actual: StorageCloneContract
) -> None:
    """Fail without rendering environment values when a clone changed state."""

    changed = [
        field.name
        for field in fields(StorageCloneContract)
        if getattr(expected, field.name) != getattr(actual, field.name)
    ]
    if changed:
        raise ValueError(
            "storage replacement changed protected runtime fields: "
            + ", ".join(changed)
        )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _optional_string(value: object, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or "\x00" in value or "\n" in value:
        raise ValueError(f"{label} is invalid")
    return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or "\x00" in item for item in value
    ):
        raise ValueError(f"{label} is invalid")
    return tuple(value)


def _environment_contract(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("container environment is invalid")
    environment: dict[str, str] = {}
    for item in value:
        if not isinstance(item, str) or "=" not in item or "\x00" in item:
            raise ValueError("container environment is invalid")
        key, item_value = item.split("=", 1)
        if not key or "\n" in key or "\n" in item_value:
            raise ValueError("container environment is invalid")
        environment[key] = item_value
    return tuple(sorted(environment.items()))


def _string_mapping_contract(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(item, str)
        or "\x00" in key
        or "\x00" in item
        for key, item in value.items()
    ):
        raise ValueError(f"{label} is invalid")
    return tuple(sorted(value.items()))


def _empty_object_mapping(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key or item not in (None, {})
        for key, item in value.items()
    ):
        raise ValueError(f"{label} are invalid")
    return tuple(sorted(value))


def _mount_contract(value: object, service: str) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, list):
        raise ValueError("container mounts are invalid")
    mounts: list[tuple[object, ...]] = []
    data_destination = _DATA_DESTINATION[service]
    durable_data_mounts = 0
    for item in value:
        mount = _mapping(item, "container mount")
        mount_type = _optional_string(mount.get("Type"), "mount type")
        source = _optional_string(mount.get("Source"), "mount source")
        name = _optional_string(mount.get("Name"), "mount name")
        destination = _optional_string(mount.get("Destination"), "mount destination")
        if mount_type not in {"bind", "volume"} or not source or not destination:
            raise ValueError("only bind and volume mounts are safely cloneable")
        if mount_type == "volume" and not name:
            raise ValueError("volume mount identity is missing")
        if mount_type == "bind" and name:
            raise ValueError("bind mount has an invalid volume identity")
        read_write = mount.get("RW")
        if not isinstance(read_write, bool):
            raise ValueError("mount access mode is invalid")
        if destination == data_destination:
            durable_data_mounts += 1
            if not read_write:
                raise ValueError("managed storage data mount is read-only")
        mounts.append(
            (
                mount_type,
                source,
                name,
                destination,
                read_write,
                _optional_string(mount.get("Propagation"), "mount propagation"),
                _optional_string(mount.get("Driver"), "mount driver"),
                _optional_string(mount.get("Mode"), "mount mode"),
            )
        )
    if durable_data_mounts != 1:
        raise ValueError("managed storage data mount identity is ambiguous or missing")
    return tuple(sorted(mounts))


def _validate_cloneable_mounts(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("container mounts are invalid")
    destinations: set[str] = set()
    for item in value:
        mount = _mapping(item, "container mount")
        mount_type = _optional_string(mount.get("Type"), "mount type")
        source = _optional_string(mount.get("Source"), "mount source")
        destination = _optional_string(mount.get("Destination"), "mount destination")
        if mount_type not in {"bind", "volume"} or not source or not destination:
            raise ValueError("only bind and volume mounts are safely cloneable")
        if destination in destinations:
            raise ValueError("container mount destinations are ambiguous")
        destinations.add(destination)
        if not isinstance(mount.get("RW"), bool):
            raise ValueError("mount access mode is invalid")


def _port_binding_contract(
    value: object,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("container port bindings are invalid")
    result: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for container_port, bindings in value.items():
        if not isinstance(container_port, str) or not isinstance(bindings, list):
            raise ValueError("container port bindings are invalid")
        normalized: list[tuple[str, str]] = []
        for item in bindings:
            binding = _mapping(item, "container port binding")
            host_ip = _optional_string(binding.get("HostIp"), "host port address")
            host_port = _optional_string(binding.get("HostPort"), "host port")
            if not host_port.isdigit():
                raise ValueError("container host port is invalid")
            normalized.append((host_ip, host_port))
        result.append((container_port, tuple(sorted(normalized))))
    return tuple(sorted(result))


def _restart_policy_contract(value: object) -> tuple[str, int]:
    if value is None:
        return ("", 0)
    restart = _mapping(value, "container restart policy")
    name = _optional_string(restart.get("Name"), "restart policy name")
    maximum = restart.get("MaximumRetryCount", 0)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
        raise ValueError("restart retry count is invalid")
    return name, maximum


def _network_contract(
    value: object,
    snapshot: dict[str, Any],
    *,
    transient_names: set[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    networks = _mapping(value, "container networks")
    transient = set(transient_names)
    container_id = snapshot.get("Id")
    if isinstance(container_id, str) and container_id:
        transient.add(container_id[:12])
    result: list[tuple[str, tuple[str, ...]]] = []
    for network_name, raw_endpoint in networks.items():
        if not isinstance(network_name, str) or not network_name:
            raise ValueError("container network name is invalid")
        aliases = _network_endpoint_aliases(raw_endpoint, transient)
        result.append(
            (
                network_name,
                aliases,
            )
        )
    if not result:
        raise ValueError("container has no managed network attachment")
    return tuple(sorted(result))


def _network_endpoint_aliases(
    raw_endpoint: object, transient: set[str]
) -> tuple[str, ...]:
    endpoint = _mapping(raw_endpoint, "container network endpoint")
    _reject_uncloneable_network_endpoint_options(endpoint)
    aliases = _network_endpoint_names(
        endpoint.get("Aliases"), "container network aliases are invalid"
    )
    _network_endpoint_names(
        endpoint.get("DNSNames"), "container network DNS names are invalid"
    )
    _reject_unknown_network_endpoint_options(endpoint)
    return tuple(sorted(alias for alias in aliases if alias not in transient))


def _reject_uncloneable_network_endpoint_options(
    endpoint: dict[str, Any],
) -> None:
    if endpoint.get("IPAMConfig") not in (None, {}) or endpoint.get(
        "DriverOpts"
    ) not in (None, {}):
        raise ValueError("static network endpoint options are not safely cloneable")
    for key in ("Links", "LinkLocalIPs"):
        if endpoint.get(key) not in (None, []):
            raise ValueError(
                f"non-default network endpoint {key} is not safely cloneable"
            )
    if endpoint.get("GwPriority") not in (None, 0):
        raise ValueError(
            "non-default network endpoint GwPriority is not safely cloneable"
        )


def _network_endpoint_names(value: object, error_message: str) -> list[str]:
    names = value or []
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError(error_message)
    return names


def _reject_unknown_network_endpoint_options(endpoint: dict[str, Any]) -> None:
    covered_endpoint_fields = {
        "Aliases",
        "DNSNames",
        "DriverOpts",
        "EndpointID",
        "Gateway",
        "GlobalIPv6Address",
        "GlobalIPv6PrefixLen",
        "GwPriority",
        "IPAMConfig",
        "IPAddress",
        "IPPrefixLen",
        "IPv6Gateway",
        "LinkLocalIPs",
        "Links",
        "MacAddress",
        "NetworkID",
    }
    for key, item in endpoint.items():
        if key not in covered_endpoint_fields and item not in (
            None,
            False,
            0,
            "",
            [],
            {},
        ):
            raise ValueError(
                f"non-default network endpoint {key} is not safely cloneable"
            )


def _host_options_contract(host: dict[str, Any]) -> tuple[tuple[str, object], ...]:
    _reject_unreplayed_host_options(host)
    _validate_replayed_host_options(host)
    options: list[tuple[str, object]] = []
    for key in (
        "Ulimits",
        "Devices",
        "DeviceRequests",
        "GroupAdd",
        "CapAdd",
        "CapDrop",
        "SecurityOpt",
        "Dns",
        "DnsSearch",
        "ExtraHosts",
    ):
        options.append((key, _freeze(host.get(key) or [])))
    for key in ("Privileged", "ReadonlyRootfs"):
        options.append((key, bool(host.get(key))))
    for key in ("IpcMode", "PidMode"):
        options.append((key, _optional_string(host.get(key), key)))
    shm_size = host.get("ShmSize", 0)
    if not isinstance(shm_size, int) or isinstance(shm_size, bool) or shm_size < 0:
        raise ValueError("container shared-memory size is invalid")
    options.append(("ShmSize", shm_size))
    return tuple(options)


def _validate_replayed_host_options(host: dict[str, Any]) -> None:
    _validate_ulimits(host.get("Ulimits"))
    _validate_devices(host.get("Devices"))
    _validate_replayed_list_options(host)
    _validate_replayed_namespace_options(host)


def _validate_ulimits(value: object) -> None:
    ulimits = value or []
    if not isinstance(ulimits, list):
        raise ValueError("container Ulimits are invalid")
    for item in ulimits:
        limit = _mapping(item, "container ulimit")
        if set(limit) != {"Name", "Soft", "Hard"}:
            raise ValueError("container Ulimits are invalid")
        name = limit.get("Name")
        soft, hard = limit.get("Soft"), limit.get("Hard")
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or not isinstance(soft, int)
            or isinstance(soft, bool)
            or not isinstance(hard, int)
            or isinstance(hard, bool)
        ):
            raise ValueError("container Ulimits are invalid")


def _validate_devices(value: object) -> None:
    devices = value or []
    if not isinstance(devices, list):
        raise ValueError("container Devices are invalid")
    for item in devices:
        device = _mapping(item, "container device")
        if set(device) != {"PathOnHost", "PathInContainer", "CgroupPermissions"}:
            raise ValueError("container Devices are invalid")
        source = _optional_string(device.get("PathOnHost"), "device host path")
        target = _optional_string(
            device.get("PathInContainer"), "device container path"
        )
        permissions = _optional_string(
            device.get("CgroupPermissions"), "device permissions"
        )
        if not source or not target or not permissions or set(permissions) - set("rwm"):
            raise ValueError("container Devices are invalid")


def _validate_replayed_list_options(host: dict[str, Any]) -> None:
    for key in (
        "GroupAdd",
        "CapAdd",
        "CapDrop",
        "SecurityOpt",
        "Dns",
        "DnsSearch",
        "ExtraHosts",
    ):
        _string_sequence(host.get(key), key)
    for key in ("Privileged", "ReadonlyRootfs"):
        if host.get(key) not in (None, False, True):
            raise ValueError(f"container {key} is invalid")


def _validate_replayed_namespace_options(host: dict[str, Any]) -> None:
    for key in ("IpcMode", "PidMode"):
        value = _optional_string(host.get(key), key)
        if value.startswith("container:"):
            raise ValueError(f"container-scoped {key} is not safely cloneable")
    for key in ("Runtime", "CgroupnsMode"):
        _optional_string(host.get(key), key)


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("container host option is invalid")
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ValueError("container host option is invalid")
