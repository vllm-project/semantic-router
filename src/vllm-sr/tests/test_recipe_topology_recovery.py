"""Rollback, commit, and fail-closed clone coverage for Recipe topology."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from cli import recipe_topology_reconcile as topology
from cli import recipe_topology_reconcile_io as topology_io
from recipe_topology_test_support import (
    _POSTGRES_ANONYMOUS_MOUNT,
    _REDIS_NAMED_MOUNT,
    _STORAGE_CASES,
    _remove_transition,
    _repair_transition,
    _storage_snapshot,
)


@pytest.mark.parametrize("network_mode", ["", "default", "bridge"])
def test_clone_does_not_reconnect_the_implicit_bridge_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, network_mode: str
):
    transition = {
        "service": "router",
        "name": "vllm-sr-router",
        "backup_name": "vllm-sr-router-recipe-backup-0123456789ab",
        "action": "replace",
        "was_running": True,
    }
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    snapshot["HostConfig"]["NetworkMode"] = network_mode
    snapshot["NetworkSettings"]["Networks"] = {
        "bridge": {
            "Aliases": [str(transition["name"]), "a" * 12],
            "IPAMConfig": None,
            "DriverOpts": None,
        }
    }
    commands: list[list[str]] = []

    def run(arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        commands.append(arguments)

    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)
    monkeypatch.setattr(topology, "_run", run)

    topology._clone_preserved_container(
        tmp_path / "topology.json",
        transition,
        listeners=None,
        credential=None,
    )

    assert not any(
        command[:3] == ["network", "connect", "bridge"] for command in commands
    )
    assert not any(command[:2] == ["network", "connect"] for command in commands)


def test_environment_transport_fails_closed_without_linux_memfd(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delattr(topology.os, "memfd_create", raising=False)

    with pytest.raises(topology.TopologyReconcileError, match="memfd support"):
        topology._open_environment_memfd(["SECRET=must-not-hit-disk"])


def test_run_raw_inherits_environment_memfd_into_runtime_child(
    monkeypatch: pytest.MonkeyPatch,
):
    descriptor = topology._open_environment_memfd(["MEMFD_SENTINEL=value"])
    monkeypatch.setattr(topology_io, "get_container_runtime", lambda: sys.executable)
    try:
        result = topology._run_raw(
            [
                "-c",
                (
                    "import pathlib,sys; "
                    "sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())"
                ),
                f"/proc/self/fd/{descriptor}",
            ],
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)

    assert result.returncode == 0
    assert result.stdout == "MEMFD_SENTINEL=value\n"


def test_clone_interruption_closes_anonymous_environment_without_recovery_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transition = {
        "service": "router",
        "name": "vllm-sr-router",
        "backup_name": "vllm-sr-router-recipe-backup-0123456789ab",
        "action": "replace",
        "was_running": True,
    }
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    inherited_descriptors: list[int] = []

    def interrupt(_arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        inherited_descriptors.extend(pass_fds)
        raise topology.TopologyReconcileError("simulated container runtime exit")

    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)
    monkeypatch.setattr(topology, "_run", interrupt)

    with pytest.raises(topology.TopologyReconcileError, match="simulated"):
        topology._clone_preserved_container(
            tmp_path / "topology.json",
            transition,
            listeners=None,
            credential=None,
        )

    assert len(inherited_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(inherited_descriptors[0])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("service,mount,port", _STORAGE_CASES)
def test_repair_rollback_restores_original_storage_mount_identity(
    monkeypatch: pytest.MonkeyPatch,
    service: str,
    mount: dict[str, object],
    port: str,
):
    del port
    transition = _repair_transition(service)
    statuses = {
        transition["name"]: "running",
        transition["backup_name"]: "exited",
    }
    data_mounts = {
        transition["name"]: {"Source": "/wrong/new-storage"},
        transition["backup_name"]: deepcopy(mount),
    }
    events: list[str] = []

    def remove(name: str) -> None:
        events.append(f"remove:{name}")
        statuses[name] = "not found"
        data_mounts.pop(name, None)

    def run(arguments: list[str]) -> None:
        events.append(":".join(arguments))
        statuses[arguments[2]] = statuses.pop(arguments[1])
        data_mounts[arguments[2]] = data_mounts.pop(arguments[1])

    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", remove)
    monkeypatch.setattr(topology, "_run", run)

    topology._restore_preserved_transition(transition, restore_running=True)

    assert statuses[transition["name"]] == "exited"
    assert data_mounts[transition["name"]] == mount
    assert all(not event.startswith("start:") for event in events)


@pytest.mark.parametrize("service,mount,port", _STORAGE_CASES)
def test_repair_commit_verifies_storage_contract_before_cleaning_backup(
    monkeypatch: pytest.MonkeyPatch,
    service: str,
    mount: dict[str, object],
    port: str,
):
    transition = _repair_transition(service)
    statuses = {
        transition["name"]: "running",
        transition["backup_name"]: "exited",
    }
    snapshots = {
        transition["backup_name"]: _storage_snapshot(
            service,
            mount,
            port,
            container_id="a" * 64,
            container_name=str(transition["name"]),
        ),
        transition["name"]: _storage_snapshot(
            service,
            mount,
            port,
            container_id="b" * 64,
            container_name=str(transition["name"]),
        ),
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    topology._commit({"containers": [transition], "storage_after": [service]})

    assert removed == [transition["backup_name"]]


def test_repair_commit_is_idempotent_after_verified_backup_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _repair_transition()
    statuses = {
        transition["name"]: "running",
        transition["backup_name"]: "not found",
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(
        topology,
        "_inspect",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("an already-cleaned backup cannot be re-inspected")
        ),
    )
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)
    monkeypatch.setattr(topology, "_validate_storage_isolation", lambda _name: None)

    topology._commit({"containers": [transition], "storage_after": ["redis"]})

    assert removed == [transition["backup_name"]]


def test_repair_commit_fails_closed_before_backup_cleanup_on_mount_change(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _repair_transition("postgres")
    expected_mount = deepcopy(_POSTGRES_ANONYMOUS_MOUNT)
    changed_mount = deepcopy(_POSTGRES_ANONYMOUS_MOUNT)
    changed_mount["Name"] = "e" * 64
    changed_mount["Source"] = "/var/lib/docker/volumes/" + "e" * 64 + "/_data"
    snapshots = {
        transition["backup_name"]: _storage_snapshot(
            "postgres",
            expected_mount,
            "5432",
            container_id="a" * 64,
            container_name=str(transition["name"]),
        ),
        transition["name"]: _storage_snapshot(
            "postgres",
            changed_mount,
            "5432",
            container_id="b" * 64,
            container_name=str(transition["name"]),
        ),
    }
    statuses = {
        transition["name"]: "running",
        transition["backup_name"]: "exited",
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    with pytest.raises(
        topology.TopologyReconcileError, match="protected runtime fields: mounts"
    ):
        topology._commit({"containers": [transition], "storage_after": ["postgres"]})

    assert removed == []


def test_repair_apply_fails_closed_before_run_without_durable_data_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transition = _repair_transition("redis")
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    snapshot["Mounts"] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(
        topology,
        "_container_status",
        lambda name: "not found" if name == transition["name"] else "exited",
    )
    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)
    monkeypatch.setattr(topology, "_run", commands.append)

    with pytest.raises(topology.TopologyReconcileError, match="data mount identity"):
        topology._repair_storage(tmp_path / "topology.json", transition)

    assert commands == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Memory", 256 * 1024 * 1024),
        ("Init", False),
        ("MemorySwappiness", 0),
        ("MemorySwappiness", -1),
        ("MemorySwappiness", 50),
        ("PidsLimit", 0),
        ("PidsLimit", -1),
        ("PidsLimit", 128),
        ("IpcMode", "container:peer"),
        ("PidMode", "container:peer"),
        ("LogConfig", {"Type": "syslog", "Config": {}}),
        ("Sysctls", {"net.core.somaxconn": "4096"}),
        ("Tmpfs", {"/run": "rw,noexec"}),
        ("MaskedPaths", ["/private/custom"]),
    ],
)
def test_repair_fails_closed_for_unreplayed_host_constraint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
):
    transition = _repair_transition("redis")
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name="vllm-sr-redis",
    )
    snapshot["HostConfig"][field] = value
    commands: list[list[str]] = []
    monkeypatch.setattr(
        topology,
        "_container_status",
        lambda name: "not found" if name == transition["name"] else "exited",
    )
    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)
    monkeypatch.setattr(topology, "_run", commands.append)

    with pytest.raises(topology.TopologyReconcileError, match=field):
        topology._repair_storage(tmp_path / "topology.json", transition)

    assert commands == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Tty", True),
        ("OpenStdin", True),
        ("StdinOnce", True),
        ("StopTimeout", 0),
        ("StopTimeout", 30),
        ("NetworkDisabled", True),
    ],
)
def test_apply_rejects_unreplayed_runtime_config_before_preserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
):
    suffix = "-recipe-backup-0123456789ab"
    transitions = [
        {
            "service": service,
            "name": f"vllm-sr-{service}",
            "backup_name": f"vllm-sr-{service}{suffix}",
            "action": "replace",
            "was_running": True,
        }
        for service in ("router", "envoy")
    ]
    snapshots = {
        transition["name"]: _storage_snapshot(
            "redis",
            _REDIS_NAMED_MOUNT,
            "6379",
            container_id=("a" if transition["service"] == "router" else "b") * 64,
            container_name=str(transition["name"]),
        )
        for transition in transitions
    }
    snapshots["vllm-sr-router"]["Config"][field] = value
    preserved: list[str] = []
    monkeypatch.setattr(
        topology,
        "_container_status",
        lambda name: "not found" if "recipe-backup" in name else "running",
    )
    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(
        topology,
        "_preserve_original",
        lambda transition: preserved.append(str(transition["service"])) or True,
    )

    with pytest.raises(topology.TopologyReconcileError, match=field):
        topology._apply(
            tmp_path / "topology.json",
            {"containers": transitions, "listeners": [], "storage_before": []},
            "",
        )

    assert preserved == []


@pytest.mark.parametrize("status", sorted(topology._OFFLINE_COMMIT_STATES))
def test_offline_commit_accepts_existing_target_and_cleans_backup(
    monkeypatch: pytest.MonkeyPatch, status: str
):
    transition = _repair_transition()
    statuses = {
        transition["name"]: status,
        transition["backup_name"]: "exited",
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)
    monkeypatch.setattr(topology, "_verify_storage_repair", lambda _transition: None)
    monkeypatch.setattr(topology, "_validate_storage_isolation", lambda _name: None)

    topology._commit_offline({"containers": [transition], "storage_after": ["redis"]})

    assert removed == [transition["backup_name"]]


def test_offline_commit_rejects_missing_target(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _repair_transition()
    monkeypatch.setattr(topology, "_container_status", lambda _name: "not found")

    with pytest.raises(topology.TopologyReconcileError, match="recoverable state"):
        topology._commit_offline(
            {"containers": [transition], "storage_after": ["redis"]}
        )


def test_storage_clone_exit_rolls_back_repair(
    monkeypatch: pytest.MonkeyPatch,
):
    transaction_suffix = "-recipe-backup-0123456789ab"
    repair = _repair_transition()
    transitions = [
        repair,
        {
            "service": "router",
            "name": "vllm-sr-router",
            "backup_name": "vllm-sr-router" + transaction_suffix,
            "action": "replace",
            "was_running": True,
        },
        {
            "service": "envoy",
            "name": "vllm-sr-envoy",
            "backup_name": "vllm-sr-envoy" + transaction_suffix,
            "action": "replace",
            "was_running": True,
        },
    ]
    statuses = {
        repair["name"]: "exited",
        repair["backup_name"]: "not found",
        "vllm-sr-router": "running",
        "vllm-sr-router" + transaction_suffix: "not found",
        "vllm-sr-envoy": "running",
        "vllm-sr-envoy" + transaction_suffix: "not found",
    }
    events: list[str] = []

    def run(arguments: list[str]) -> None:
        events.append(":".join(arguments))
        if arguments[0] == "stop":
            statuses[arguments[1]] = "exited"
        elif arguments[0] == "rename":
            statuses[arguments[2]] = statuses.pop(arguments[1])

    def remove(name: str) -> None:
        events.append(f"remove:{name}")
        statuses[name] = "not found"

    def repair_storage(_state_path, _transition):
        statuses[repair["name"]] = "exited"
        raise topology.TopologyReconcileError("managed redis repair exited")

    monkeypatch.setattr(
        topology, "_container_status", lambda name: statuses.get(name, "not found")
    )
    monkeypatch.setattr(topology, "_run", run)
    monkeypatch.setattr(topology, "_remove_if_present", remove)
    monkeypatch.setattr(topology, "_repair_storage", repair_storage)

    monkeypatch.setattr(topology, "_validate_storage_isolation", lambda _name: None)
    monkeypatch.setattr(
        topology, "_validate_cloneable_transition", lambda _transition: None
    )
    monkeypatch.setattr(
        topology, "_validate_repair_transition", lambda _transition: None
    )
    state = {"containers": transitions, "listeners": [], "storage_before": ["redis"]}
    with pytest.raises(topology.TopologyReconcileError, match="exited"):
        topology._apply(Path("/tmp/topology.json"), state, "")
    topology._rollback(state)

    assert statuses[repair["name"]] == "exited"
    assert statuses.get(repair["backup_name"], "not found") == "not found"


def test_apply_failure_rollback_restores_removed_storage_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    transaction_suffix = "-recipe-backup-0123456789ab"
    transitions = [
        {
            "service": "envoy",
            "name": "vllm-sr-envoy",
            "backup_name": "vllm-sr-envoy" + transaction_suffix,
            "action": "replace",
            "was_running": True,
        },
        _remove_transition(),
        {
            "service": "router",
            "name": "vllm-sr-router",
            "backup_name": "vllm-sr-router" + transaction_suffix,
            "action": "replace",
            "was_running": True,
        },
    ]
    statuses = {
        "vllm-sr-envoy": "not found",
        "vllm-sr-envoy" + transaction_suffix: "exited",
        "vllm-sr-redis": "not found",
        "vllm-sr-redis" + transaction_suffix: "exited",
        "vllm-sr-router": "running",
        "vllm-sr-router" + transaction_suffix: "exited",
    }
    events: list[str] = []

    def remove(name: str) -> None:
        events.append(f"remove:{name}")
        statuses[name] = "not found"

    def run(arguments: list[str]) -> None:
        events.append(":".join(arguments))
        if arguments[0] == "rename":
            statuses[arguments[2]] = statuses.pop(arguments[1])
        elif arguments[0] == "start":
            statuses[arguments[1]] = "running"

    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", remove)
    monkeypatch.setattr(topology, "_run", run)

    topology._rollback({"containers": transitions})

    redis_start = events.index("start:vllm-sr-redis")
    router_start = events.index("start:vllm-sr-router")
    envoy_start = events.index("start:vllm-sr-envoy")
    assert redis_start < router_start < envoy_start
    assert statuses["vllm-sr-redis"] == "running"
    assert statuses["vllm-sr-router"] == "running"
    assert statuses["vllm-sr-envoy"] == "running"
