"""Shared snapshots and transitions for topology reconciliation tests."""

from copy import deepcopy

import pytest


def _remove_transition() -> dict[str, object]:
    return {
        "service": "redis",
        "name": "vllm-sr-redis",
        "backup_name": "vllm-sr-redis-recipe-backup-0123456789ab",
        "action": "remove",
        "was_running": True,
    }


def _repair_transition(service: str = "redis") -> dict[str, object]:
    return {
        "service": service,
        "name": f"vllm-sr-{service}",
        "backup_name": f"vllm-sr-{service}-recipe-backup-0123456789ab",
        "action": "repair",
        "was_running": False,
    }


_REDIS_NAMED_MOUNT = {
    "Type": "volume",
    "Name": "redis-state",
    "Source": "/var/lib/docker/volumes/redis-state/_data",
    "Destination": "/data",
    "Driver": "local",
    "Mode": "z",
    "RW": True,
    "Propagation": "",
}
_POSTGRES_ANONYMOUS_MOUNT = {
    "Type": "volume",
    "Name": "f" * 64,
    "Source": "/var/lib/docker/volumes/" + "f" * 64 + "/_data",
    "Destination": "/var/lib/postgresql/data",
    "Driver": "local",
    "Mode": "",
    "RW": True,
    "Propagation": "",
}
_MILVUS_BIND_MOUNT = {
    "Type": "bind",
    "Name": "",
    "Source": "/srv/vllm-sr/milvus-data",
    "Destination": "/var/lib/milvus",
    "Driver": "",
    "Mode": "z",
    "RW": True,
    "Propagation": "rprivate",
}
_STORAGE_CASES = (
    pytest.param(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        id="redis-named-volume",
    ),
    pytest.param(
        "postgres",
        _POSTGRES_ANONYMOUS_MOUNT,
        "5432",
        id="postgres-anonymous-volume",
    ),
    pytest.param(
        "milvus",
        _MILVUS_BIND_MOUNT,
        "19530",
        id="milvus-bind-mount",
    ),
)


def _storage_snapshot(
    service: str,
    mount: dict[str, object],
    port: str,
    *,
    container_id: str,
    container_name: str,
) -> dict[str, object]:
    return {
        "Id": container_id,
        "Image": "sha256:" + "b" * 64,
        "Name": "/" + container_name,
        "Config": {
            "Image": f"registry.invalid/{service}@sha256:" + "a" * 64,
            "Env": ["PATH=/usr/bin", "STORAGE_SECRET=must-not-leak"],
            "Hostname": service,
            "User": "",
            "WorkingDir": "/var/lib/storage",
            "StopSignal": "SIGTERM",
            "Labels": {"io.vllm-sr.managed": "true"},
            "Entrypoint": ["/storage-entrypoint"],
            "Cmd": ["serve"],
            "Healthcheck": None,
        },
        "HostConfig": {
            "RestartPolicy": {
                "Name": "unless-stopped",
                "MaximumRetryCount": 0,
            },
            "PortBindings": {
                f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}]
            },
            "PublishAllPorts": False,
            "NetworkMode": "vllm-sr-network",
            "Ulimits": [{"Name": "nofile", "Soft": 1024, "Hard": 2048}],
            "Devices": [],
            "DeviceRequests": [],
            "GroupAdd": [],
            "CapAdd": [],
            "CapDrop": [],
            "SecurityOpt": ["seccomp=unconfined"] if service == "milvus" else [],
            "Dns": [],
            "DnsSearch": [],
            "ExtraHosts": [],
            "Privileged": False,
            "ReadonlyRootfs": False,
            "IpcMode": "private",
            "PidMode": "",
            "ShmSize": 64 * 1024 * 1024,
        },
        "NetworkSettings": {
            "Ports": {f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}]},
            "Networks": {
                "vllm-sr-network": {
                    "Aliases": [container_name, container_id[:12], service],
                    "IPAMConfig": None,
                    "DriverOpts": None,
                }
            },
        },
        "Mounts": [deepcopy(mount)],
    }
