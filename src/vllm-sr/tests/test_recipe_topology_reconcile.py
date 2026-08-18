import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from cli import recipe_topology_reconcile as topology
from recipe_topology_test_support import (
    _REDIS_NAMED_MOUNT,
    _STORAGE_CASES,
    _remove_transition,
    _repair_transition,
    _storage_snapshot,
)


def test_preserved_router_environment_reconciles_managed_credential():
    existing = [
        "NORMAL=value",
        f"{topology.MANAGEMENT_CREDENTIAL_ENV}=stale-token",
    ]

    assert topology._normalized_environment(existing, "") == ["NORMAL=value"]
    assert topology._normalized_environment(existing, "new-token") == [
        "NORMAL=value",
        f"{topology.MANAGEMENT_CREDENTIAL_ENV}=new-token",
    ]
    assert topology._normalized_environment(existing, None) == existing


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("0.0.0.0", "0.0.0.0"),
        ("::", "::"),
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
    ],
)
def test_listener_host_address_preserves_explicit_publish_scope(address, expected):
    assert topology._listener_host_address({"address": address}) == expected


def test_listener_host_address_rejects_ambiguous_hostname():
    with pytest.raises(topology.TopologyReconcileError, match="listener address"):
        topology._listener_host_address({"address": "localhost"})


def test_management_listener_from_config_resolves_managed_host_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "global:\n"
        "  services:\n"
        "    management_api:\n"
        "      bind_address: 0.0.0.0\n"
        "      port: 9090\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "7")

    assert topology._management_listener_from_config(config_path) == (9090, 9097)


@pytest.mark.parametrize("bind_address", ["127.0.0.1", "::1", "localhost"])
def test_management_listener_from_config_rejects_unreachable_split_bind(
    bind_address: str, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "global:\n"
        "  services:\n"
        "    management_api:\n"
        f"      bind_address: {bind_address}\n"
        "      port: 8080\n",
        encoding="utf-8",
    )

    with pytest.raises(
        topology.TopologyReconcileError, match=r"bind_address 0\.0\.0\.0"
    ):
        topology._management_listener_from_config(config_path)


def test_router_management_bindings_are_rewritten_to_one_loopback_publication():
    host = {
        "NetworkMode": "vllm-sr-network",
        "PublishAllPorts": False,
        "PortBindings": {
            "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}],
            "50051/tcp": [{"HostIp": "127.0.0.1", "HostPort": "50051"}],
        },
    }

    assert topology._rewrite_router_management_bindings(host, (8080, 8080)) == {
        "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}],
        "50051/tcp": [{"HostIp": "127.0.0.1", "HostPort": "50051"}],
    }


def test_router_management_isolation_rejects_actual_wildcard_publication():
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "8080",
        container_id="a" * 64,
        container_name="vllm-sr-router-container",
    )
    snapshot["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"

    with pytest.raises(topology.TopologyReconcileError, match="not loopback"):
        topology._validate_router_management_isolation(snapshot, (8080, 8080))


def test_preserved_ipv6_port_binding_uses_bracketed_docker_syntax():
    command: list[str] = []
    topology._append_port_bindings(
        command,
        {"8899/tcp": [{"HostIp": "::1", "HostPort": "8899"}]},
    )
    assert command == ["-p", "[::1]:8899:8899/tcp"]


def test_container_status_distinguishes_absence_from_runtime_failure(monkeypatch):
    monkeypatch.setattr(
        topology,
        "_run_raw",
        lambda _arguments: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(topology.TopologyReconcileError, match="could not be inspected"):
        topology._container_status("vllm-sr-redis")

    monkeypatch.setattr(
        topology,
        "_run_raw",
        lambda _arguments: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="No such container: missing"
        ),
    )
    assert topology._container_status("missing") == "not found"


def test_host_hidden_state_dir_resolves_exact_split_dashboard_bind(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "vllm-sr-dashboard")
    monkeypatch.setattr(
        topology,
        "_inspect",
        lambda _name: {
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/srv/vllm-sr/state",
                    "Destination": "/app/.vllm-sr",
                    "RW": True,
                }
            ]
        },
    )

    assert topology._host_hidden_state_dir() == "/srv/vllm-sr/state"


def test_host_hidden_state_dir_rejects_container_local_or_ambiguous_mounts(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "vllm-sr-dashboard")
    for mounts in (
        [{"Type": "bind", "Source": "/app", "Destination": "/app", "RW": True}],
        [
            {
                "Type": "volume",
                "Source": "/srv/state",
                "Destination": "/app/.vllm-sr",
                "RW": True,
            }
        ],
        [
            {
                "Type": "bind",
                "Source": "/srv/one",
                "Destination": "/app/.vllm-sr",
                "RW": True,
            },
            {
                "Type": "bind",
                "Source": "/srv/two",
                "Destination": "/app/.vllm-sr",
                "RW": True,
            },
        ],
    ):
        monkeypatch.setattr(
            topology, "_inspect", lambda _name, value=mounts: {"Mounts": value}
        )
        with pytest.raises(topology.TopologyReconcileError, match="mount"):
            topology._host_hidden_state_dir()


def test_remove_commit_accepts_absent_active_and_cleans_backup(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _remove_transition()
    statuses = {
        transition["name"]: "not found",
        transition["backup_name"]: "exited",
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    topology._commit({"containers": [transition]})

    assert removed == [transition["backup_name"]]


def test_remove_commit_is_idempotent_after_backup_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _remove_transition()
    monkeypatch.setattr(topology, "_container_status", lambda _name: "not found")
    removed: list[str] = []
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    topology._commit({"containers": [transition]})

    assert removed == [transition["backup_name"]]


def test_remove_commit_rejects_still_active_container(
    monkeypatch: pytest.MonkeyPatch,
):
    transition = _remove_transition()
    monkeypatch.setattr(
        topology,
        "_container_status",
        lambda name: "running" if name == transition["name"] else "exited",
    )

    with pytest.raises(topology.TopologyReconcileError, match="unexpectedly remains"):
        topology._commit({"containers": [transition]})


def test_repair_apply_preserves_stopped_sidecar_before_snapshot_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    transaction_suffix = "-recipe-backup-0123456789ab"
    transitions = [
        _repair_transition(),
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
    events: list[str] = []

    def preserve(transition):
        events.append(f"preserve:{transition['service']}")
        return transition["service"] == "redis"

    monkeypatch.setattr(topology, "_preserve_original", preserve)
    monkeypatch.setattr(
        topology,
        "_repair_storage",
        lambda _path, transition: events.append(f"repair:{transition['service']}"),
    )
    monkeypatch.setattr(topology, "_container_status", lambda _name: "running")
    monkeypatch.setattr(
        topology,
        "_validate_storage_isolation",
        lambda name: events.append(f"isolate:{name}"),
    )
    monkeypatch.setattr(
        topology, "_validate_cloneable_transition", lambda _transition: None
    )
    monkeypatch.setattr(
        topology, "_validate_repair_transition", lambda _transition: None
    )
    monkeypatch.setattr(
        topology,
        "_clone_preserved_container",
        lambda *_args, **_kwargs: events.append("unexpected-clone"),
    )

    topology._apply(
        tmp_path / "topology.json",
        {"containers": transitions, "listeners": [], "storage_before": ["redis"]},
        "",
    )

    assert events.index("preserve:redis") < events.index("repair:redis")
    assert events.index("isolate:vllm-sr-redis") < events.index("preserve:redis")
    assert "unexpected-clone" not in events


@pytest.mark.parametrize("service,mount,port", _STORAGE_CASES)
def test_repair_apply_clones_exact_storage_data_and_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service: str,
    mount: dict[str, object],
    port: str,
):
    transition = _repair_transition(service)
    original = _storage_snapshot(
        service,
        mount,
        port,
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    snapshots = {str(transition["backup_name"]): original}
    statuses = {
        str(transition["name"]): "not found",
        str(transition["backup_name"]): "exited",
    }
    commands: list[list[str]] = []
    inherited_environment: list[bytes] = []
    inherited_descriptors: list[int] = []

    def run(arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        commands.append(arguments)
        if arguments[0] != "run":
            raise AssertionError(f"unexpected command: {arguments}")
        assert len(pass_fds) == 1
        environment_fd = pass_fds[0]
        inherited_descriptors.append(environment_fd)
        assert arguments[arguments.index("--env-file") + 1] == (
            f"/proc/self/fd/{environment_fd}"
        )
        inherited_environment.append(os.pread(environment_fd, 64 * 1024, 0))
        replacement = deepcopy(original)
        replacement["Id"] = "b" * 64
        replacement["Name"] = "/" + str(transition["name"])
        replacement["Config"]["Image"] = str(original["Image"])
        replacement["NetworkSettings"]["Networks"]["vllm-sr-network"]["Aliases"] = [
            str(transition["name"]),
            "b" * 12,
            service,
        ]
        snapshots[str(transition["name"])] = replacement
        statuses[str(transition["name"])] = "running"

    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_run", run)
    monkeypatch.setattr(
        topology,
        "_start_storage",
        lambda _service: (_ for _ in ()).throw(
            AssertionError("repair must not provision fresh storage")
        ),
    )

    topology._repair_storage(tmp_path / "topology.json", transition)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--volumes-from") + 1] == transition["backup_name"]
    assert command[command.index("--restart") + 1] == "unless-stopped"
    assert command[command.index("--network") + 1] == "vllm-sr-network"
    assert command[command.index("--hostname") + 1] == service
    assert command[command.index("--workdir") + 1] == "/var/lib/storage"
    assert command[command.index("--stop-signal") + 1] == "SIGTERM"
    assert f"127.0.0.1:{port}:{port}/tcp" in command
    assert original["Image"] in command
    assert original["Config"]["Image"] not in command
    assert all("must-not-leak" not in argument for argument in command)
    assert inherited_environment == [b"PATH=/usr/bin\nSTORAGE_SECRET=must-not-leak\n"]
    with pytest.raises(OSError):
        os.fstat(inherited_descriptors[0])
    assert not list(tmp_path.glob(".recipe-env-*"))


@pytest.mark.parametrize("service", ["router", "envoy"])
def test_runtime_replacement_pins_immutable_image_id_when_tag_drifts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, service: str
):
    suffix = "-recipe-backup-0123456789ab"
    transition = {
        "service": service,
        "name": f"vllm-sr-{service}",
        "backup_name": f"vllm-sr-{service}{suffix}",
        "action": "replace",
        "was_running": True,
    }
    original = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    original["Config"]["Image"] = "registry.invalid/runtime:mutable"
    snapshots = {str(transition["backup_name"]): original}
    commands: list[list[str]] = []

    def run(arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        assert len(pass_fds) == 1
        commands.append(arguments)
        replacement = deepcopy(original)
        replacement["Id"] = "c" * 64
        replacement["Name"] = "/" + str(transition["name"])
        replacement["Config"]["Image"] = str(original["Image"])
        snapshots[str(transition["name"])] = replacement

    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_run", run)

    topology._clone_preserved_container(
        tmp_path / "topology.json",
        transition,
        listeners=None,
        credential=None,
    )

    assert len(commands) == 1
    assert original["Image"] in commands[0]
    assert original["Config"]["Image"] not in commands[0]
    assert not list(tmp_path.iterdir())


def test_router_replacement_rewrites_and_verifies_management_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transition = {
        "service": "router",
        "name": "vllm-sr-router-container",
        "backup_name": "vllm-sr-router-container-recipe-backup-0123456789ab",
        "action": "replace",
        "was_running": True,
    }
    original = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "8080",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    original["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
    original["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
    snapshots = {str(transition["backup_name"]): original}
    commands: list[list[str]] = []

    def run(arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        assert len(pass_fds) == 1
        commands.append(arguments)
        replacement = deepcopy(original)
        replacement["Id"] = "c" * 64
        replacement["Name"] = "/" + str(transition["name"])
        replacement["Config"]["Image"] = str(original["Image"])
        exact = [{"HostIp": "127.0.0.1", "HostPort": "8080"}]
        replacement["HostConfig"]["PortBindings"]["8080/tcp"] = deepcopy(exact)
        replacement["NetworkSettings"]["Ports"]["8080/tcp"] = deepcopy(exact)
        snapshots[str(transition["name"])] = replacement

    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_run", run)

    topology._clone_preserved_container(
        tmp_path / "topology.json",
        transition,
        listeners=None,
        credential=None,
        management=(8080, 8080),
    )

    assert len(commands) == 1
    assert "127.0.0.1:8080:8080/tcp" in commands[0]
    assert "0.0.0.0:8080:8080/tcp" not in commands[0]


def test_clone_rejects_missing_immutable_image_id_before_staging_environment(
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
    snapshot["Image"] = "registry.invalid/runtime:mutable"
    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)
    monkeypatch.setattr(
        topology,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unpinned image must not start")
        ),
    )

    with pytest.raises(
        topology.TopologyReconcileError, match="immutable image identity"
    ):
        topology._clone_preserved_container(
            tmp_path / "topology.json",
            transition,
            listeners=None,
            credential=None,
        )

    assert not list(tmp_path.iterdir())


def test_start_milvus_uses_dashboard_host_state_bind_and_rechecks_isolation(
    monkeypatch: pytest.MonkeyPatch,
):
    statuses = iter(("not found", "running"))
    captured: dict[str, object] = {}
    isolation: list[str] = []
    monkeypatch.setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "vllm-sr-dashboard")
    monkeypatch.setattr(topology, "_container_status", lambda _name: next(statuses))
    monkeypatch.setattr(
        topology, "_host_hidden_state_dir", lambda: "/srv/vllm-sr/state"
    )
    monkeypatch.setattr(
        topology,
        "container_start_milvus",
        lambda network, layout, **kwargs: (
            captured.update({"network": network, "layout": layout, **kwargs})
            or (0, "", "")
        ),
    )
    monkeypatch.setattr(topology, "_validate_storage_isolation", isolation.append)

    topology._start_storage("milvus")

    assert captured["state_root_dir"] == "/app"
    assert captured["host_hidden_state_dir"] == "/srv/vllm-sr/state"
    assert captured["reuse_existing"] is False
    assert isolation == [topology._storage_name("milvus")]


def test_start_storage_rejects_container_appearing_after_preview(monkeypatch):
    monkeypatch.setattr(topology, "_container_status", lambda _name: "running")
    monkeypatch.setattr(
        topology,
        "container_start_redis",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unexpected container must never be reused")
        ),
    )

    with pytest.raises(
        topology.TopologyReconcileError, match="unexpected managed redis"
    ):
        topology._start_storage("redis")


def test_commit_rejects_storage_that_became_wide_bound(monkeypatch):
    transition = {
        "service": "redis",
        "name": "vllm-sr-redis",
        "backup_name": "",
        "action": "add",
        "was_running": False,
    }
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name="vllm-sr-redis",
    )
    snapshot["HostConfig"]["PortBindings"]["6379/tcp"][0]["HostIp"] = "0.0.0.0"
    monkeypatch.setattr(topology, "_container_status", lambda _name: "running")
    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)

    with pytest.raises(topology.TopologyReconcileError, match="not isolated"):
        topology._commit({"containers": [transition], "storage_after": ["redis"]})


def test_repair_rejects_container_network_namespace(monkeypatch, tmp_path: Path):
    transition = _repair_transition("redis")
    snapshot = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name="vllm-sr-redis",
    )
    snapshot["HostConfig"]["NetworkMode"] = "container:peer"
    monkeypatch.setattr(
        topology,
        "_container_status",
        lambda name: "not found" if name == transition["name"] else "exited",
    )
    monkeypatch.setattr(topology, "_inspect", lambda _name: snapshot)

    with pytest.raises(topology.TopologyReconcileError, match="not isolated"):
        topology._repair_storage(tmp_path / "topology.json", transition)


def test_clone_rejects_replacement_with_different_immutable_image_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transition = {
        "service": "router",
        "name": "vllm-sr-router",
        "backup_name": "vllm-sr-router-recipe-backup-0123456789ab",
        "action": "replace",
        "was_running": True,
    }
    original = _storage_snapshot(
        "redis",
        _REDIS_NAMED_MOUNT,
        "6379",
        container_id="a" * 64,
        container_name=str(transition["name"]),
    )
    snapshots = {str(transition["backup_name"]): original}
    inherited_descriptors: list[int] = []

    def run(_arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        inherited_descriptors.extend(pass_fds)
        replacement = deepcopy(original)
        replacement["Image"] = "sha256:" + "c" * 64
        snapshots[str(transition["name"])] = replacement

    monkeypatch.setattr(topology, "_inspect", snapshots.__getitem__)
    monkeypatch.setattr(topology, "_run", run)

    with pytest.raises(topology.TopologyReconcileError, match="changed its immutable"):
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
