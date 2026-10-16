"""Recipe stack recreation on a standalone stack, where the Router serves the
listeners and no Envoy container exists."""

import pytest
from cli import recipe_topology_contract as contract
from cli import recipe_topology_reconcile as topology

SUFFIX = "-recipe-backup-0123456789ab"
LISTENERS = [
    {"name": "http-9999", "address": "0.0.0.0", "port": 9999, "host_port": 9999}
]


def _router() -> dict[str, object]:
    return {
        "service": "router",
        "name": "vllm-sr-router",
        "backup_name": "vllm-sr-router" + SUFFIX,
        "action": "replace",
        "was_running": True,
    }


def test_a_standalone_journal_needs_only_the_router():
    contract._validate_storage_transitions([_router()], set(), set())
    with pytest.raises(contract.TopologyReconcileError, match="incomplete"):
        contract._validate_storage_transitions([], set(), set())


def test_the_router_clone_publishes_the_target_listeners():
    host = {
        "NetworkMode": "vllm-sr-network",
        "PublishAllPorts": False,
        "PortBindings": {
            "50051/tcp": [{"HostIp": "127.0.0.1", "HostPort": "50051"}],
            "9190/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9190"}],
            "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}],
            "8899/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8899"}],
        },
    }

    assert topology._clone_port_bindings(host, LISTENERS, (8080, 8080)) == {
        "50051/tcp": [{"HostIp": "127.0.0.1", "HostPort": "50051"}],
        "9190/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9190"}],
        "9999/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9999"}],
        "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}],
    }


def test_a_listener_cannot_take_the_management_host_port():
    host = {"NetworkMode": "vllm-sr-network", "PublishAllPorts": False}
    listeners = [{**LISTENERS[0], "host_port": 8080}]

    with pytest.raises(topology.TopologyReconcileError, match="shared"):
        topology._clone_port_bindings(host, listeners, (8080, 8080))


def test_apply_gives_the_router_the_listeners(monkeypatch, tmp_path):
    clones = []
    monkeypatch.setattr(topology, "_preserve_original", lambda _transition: True)
    monkeypatch.setattr(topology, "_container_status", lambda _name: "not found")
    monkeypatch.setattr(
        topology,
        "_clone_preserved_container",
        lambda _path, transition, **options: clones.append(
            (transition["service"], options)
        ),
    )

    topology._replace_runtime_services(
        tmp_path / "topology.json",
        {"listeners": LISTENERS},
        [_router()],
        "token",
        (8080, 8080),
    )

    assert clones == [
        (
            "router",
            {"listeners": LISTENERS, "credential": "token", "management": (8080, 8080)},
        )
    ]


def test_rollback_restores_the_router_alone(monkeypatch):
    statuses = {"vllm-sr-router": "running", "vllm-sr-router" + SUFFIX: "exited"}
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

    monkeypatch.setattr(
        topology, "_container_status", lambda name: statuses.get(name, "not found")
    )
    monkeypatch.setattr(topology, "_remove_if_present", remove)
    monkeypatch.setattr(topology, "_run", run)

    topology._rollback({"containers": [_router()]})

    assert statuses == {"vllm-sr-router": "running"}
    assert events[-1] == "start:vllm-sr-router"
