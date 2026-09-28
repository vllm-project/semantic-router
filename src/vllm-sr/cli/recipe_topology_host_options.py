"""Fail-closed validation for container host options not replayed on clone."""

from typing import Any

_DEFAULT_MASKED_PATHS = {
    "/proc/acpi",
    "/proc/asound",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/sys/devices/virtual/powercap",
    "/sys/firmware",
}
_DEFAULT_READONLY_PATHS = {
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
}
_NUMERIC_HOST_DEFAULTS = {
    "Memory": 0,
    "MemoryReservation": 0,
    "MemorySwap": 0,
    "NanoCpus": 0,
    "CpuShares": 0,
    "CpuPeriod": 0,
    "CpuQuota": 0,
    "OomScoreAdj": 0,
    "BlkioWeight": 0,
    "CpuRealtimePeriod": 0,
    "CpuRealtimeRuntime": 0,
    "CpuCount": 0,
    "CpuPercent": 0,
    "IOMaximumIOps": 0,
    "IOMaximumBandwidth": 0,
}


def _reject_unreplayed_host_options(host: dict[str, Any]) -> None:
    _reject_unreplayed_resource_options(host)
    _reject_unreplayed_runtime_options(host)
    _reject_unreplayed_misc_options(host)
    _reject_unknown_host_options(host)


def _reject_unreplayed_resource_options(host: dict[str, Any]) -> None:
    if host.get("PublishAllPorts") not in (None, False):
        raise ValueError("non-default PublishAllPorts is not safely cloneable")
    if host.get("DeviceRequests") not in (None, []):
        raise ValueError("non-default DeviceRequests is not safely cloneable")
    for key, default in _NUMERIC_HOST_DEFAULTS.items():
        value = host.get(key)
        if value not in (None, default):
            raise ValueError(f"non-default {key} is not safely cloneable")
    for key in ("CpusetCpus", "CpusetMems"):
        if host.get(key) not in (None, ""):
            raise ValueError(f"non-default {key} is not safely cloneable")
    if host.get("OomKillDisable") not in (None, False):
        raise ValueError("non-default OomKillDisable is not safely cloneable")
    # Init, MemorySwappiness, and PidsLimit are optional Docker API fields.
    # Explicit false/zero/-1 values can override daemon defaults, so accepting
    # those values without replaying them would silently change the clone.
    if host.get("Init") is not None:
        raise ValueError("non-default Init is not safely cloneable")
    for key in ("MemorySwappiness", "PidsLimit"):
        if host.get(key) is not None:
            raise ValueError(f"non-default {key} is not safely cloneable")
    _reject_unreplayed_blkio_options(host)


def _reject_unreplayed_blkio_options(host: dict[str, Any]) -> None:
    for key in (
        "BlkioWeightDevice",
        "BlkioDeviceReadBps",
        "BlkioDeviceWriteBps",
        "BlkioDeviceReadIOps",
        "BlkioDeviceWriteIOps",
    ):
        if host.get(key) not in (None, []):
            raise ValueError(f"non-default {key} is not safely cloneable")


def _reject_unreplayed_runtime_options(host: dict[str, Any]) -> None:
    for key in (
        "DnsOptions",
        "Links",
        "DeviceCgroupRules",
    ):
        if host.get(key) not in (None, []):
            raise ValueError(f"non-default {key} is not safely cloneable")
    for key in ("Sysctls", "Tmpfs"):
        if host.get(key) not in (None, {}):
            raise ValueError(f"non-default {key} is not safely cloneable")
    if host.get("Runtime") not in (None, "", "runc"):
        raise ValueError("non-default Runtime is not safely cloneable")
    if host.get("CgroupnsMode") not in (None, "", "private"):
        raise ValueError("non-default CgroupnsMode is not safely cloneable")
    log_config = host.get("LogConfig")
    if log_config not in (
        None,
        {},
        {"Type": "json-file", "Config": {}},
    ):
        raise ValueError("non-default LogConfig is not safely cloneable")


def _reject_unreplayed_misc_options(host: dict[str, Any]) -> None:
    for key in (
        "CgroupParent",
        "UsernsMode",
        "UTSMode",
        "Isolation",
        "VolumeDriver",
        "ContainerIDFile",
    ):
        if host.get(key) not in (None, ""):
            raise ValueError(f"non-default {key} is not safely cloneable")
    for key in ("StorageOpt",):
        if host.get(key) not in (None, {}):
            raise ValueError(f"non-default {key} is not safely cloneable")
    if host.get("AutoRemove") not in (None, False):
        raise ValueError("non-default AutoRemove is not safely cloneable")
    console_size = host.get("ConsoleSize")
    if console_size not in (None, [], [0, 0]):
        raise ValueError("non-default ConsoleSize is not safely cloneable")
    for key, allowed in (
        ("MaskedPaths", _DEFAULT_MASKED_PATHS),
        ("ReadonlyPaths", _DEFAULT_READONLY_PATHS),
    ):
        value = host.get(key)
        if value in (None, []):
            continue
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
            or set(value) != allowed
        ):
            raise ValueError(f"non-default {key} is not safely cloneable")


def _reject_unknown_host_options(host: dict[str, Any]) -> None:
    covered = {
        "RestartPolicy",
        "PortBindings",
        "PublishAllPorts",
        "NetworkMode",
        "Binds",
        "VolumesFrom",
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
        "Privileged",
        "ReadonlyRootfs",
        "IpcMode",
        "PidMode",
        "ShmSize",
        *_NUMERIC_HOST_DEFAULTS,
        "CpusetCpus",
        "CpusetMems",
        "OomKillDisable",
        "Init",
        "BlkioWeightDevice",
        "BlkioDeviceReadBps",
        "BlkioDeviceWriteBps",
        "BlkioDeviceReadIOps",
        "BlkioDeviceWriteIOps",
        "MemorySwappiness",
        "PidsLimit",
        "DnsOptions",
        "Links",
        "DeviceCgroupRules",
        "Sysctls",
        "Tmpfs",
        "Runtime",
        "CgroupnsMode",
        "LogConfig",
        "CgroupParent",
        "UsernsMode",
        "UTSMode",
        "Isolation",
        "VolumeDriver",
        "ContainerIDFile",
        "StorageOpt",
        "AutoRemove",
        "ConsoleSize",
        "MaskedPaths",
        "ReadonlyPaths",
    }
    for key, value in host.items():
        if key not in covered and value not in (None, False, 0, "", [], {}):
            raise ValueError(f"non-default {key} is not safely cloneable")
