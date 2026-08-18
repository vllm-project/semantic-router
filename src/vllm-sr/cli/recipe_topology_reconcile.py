"""Transactional container topology reconciliation for managed Recipe activation.

The Dashboard writes an immutable, credential-free topology journal before
calling this helper. Existing containers are renamed, never deleted, until the
activation journal commits; rollback can therefore restore their exact image,
mount, device, environment, and network state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from cli.container_runtime import get_container_runtime
from cli.container_services import (
    container_start_milvus,
    container_start_postgres,
    container_start_redis,
)
from cli.recipe_topology_clone import (
    _append_port_bindings,
    _base_clone_command,
    _clone_port_bindings,
    _clone_snapshot_contract,
    _normalized_environment,
    _open_environment_memfd,
    _preserved_network_aliases,
    _validate_clone_replacement,
    _validate_router_management_isolation,
)
from cli.recipe_topology_clone import (
    _listener_host_address as _listener_host_address_impl,
)
from cli.recipe_topology_clone import (
    _rewrite_router_management_bindings as _rewrite_router_management_bindings_impl,
)
from cli.recipe_topology_contract import (
    _DEFAULT_MANAGEMENT_BIND_ADDRESS,
    _DEFAULT_MANAGEMENT_PORT,
    _OFFLINE_COMMIT_STATES,
    _RUNTIME_SERVICES,
    _STORAGE,
    TopologyReconcileError,
    _valid_port,
    load_topology_state,
)
from cli.recipe_topology_contract import (
    MANAGEMENT_CREDENTIAL_ENV as _MANAGEMENT_CREDENTIAL_ENV,
)
from cli.recipe_topology_contract import TOPOLOGY_SCHEMA as _TOPOLOGY_SCHEMA
from cli.recipe_topology_storage import (
    assert_storage_contract_preserved,
    storage_clone_contract,
    validate_cloneable_container_snapshot,
    validate_storage_port_isolation,
)
from cli.runtime_stack import resolve_runtime_stack

MANAGEMENT_CREDENTIAL_ENV = _MANAGEMENT_CREDENTIAL_ENV
TOPOLOGY_SCHEMA = _TOPOLOGY_SCHEMA
_listener_host_address = _listener_host_address_impl
_rewrite_router_management_bindings = _rewrite_router_management_bindings_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("apply", "rollback", "commit"))
    parser.add_argument("--state", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        state_path = Path(args.state).absolute()
        state = _load_state(state_path)
        config_path = Path(args.config).absolute()
        _require_regular_file(config_path, 16 * 1024 * 1024)
        credential = ""
        if args.action == "apply" and state.get("credential_env"):
            credential = _read_credential()
        if args.action == "apply":
            management = _management_listener_from_config(config_path)
            _apply(state_path, state, credential, management=management)
        elif args.action == "rollback":
            _rollback(state)
        else:
            _commit(state)
    except (OSError, ValueError, TopologyReconcileError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps({"status": "ok"}))
    return 0


def _load_state(path: Path) -> dict[str, Any]:
    return load_topology_state(
        path,
        read_bounded_file=_read_bounded_file,
        expected_container_names=_expected_container_names(),
    )


def _management_listener_from_config(path: Path) -> tuple[int, int]:
    encoded = _read_bounded_file(path, 16 * 1024 * 1024)
    try:
        document = yaml.safe_load(encoded)
    except yaml.YAMLError as error:
        raise TopologyReconcileError(
            "runtime management API configuration is invalid"
        ) from error
    if not isinstance(document, dict):
        raise TopologyReconcileError("runtime management API configuration is invalid")
    global_config = document.get("global", {})
    if not isinstance(global_config, dict):
        raise TopologyReconcileError("runtime management API configuration is invalid")
    services = global_config.get("services", {})
    if not isinstance(services, dict):
        raise TopologyReconcileError("runtime management API configuration is invalid")
    management = services.get("management_api", {})
    if not isinstance(management, dict):
        raise TopologyReconcileError("runtime management API configuration is invalid")
    bind_address = management.get("bind_address", _DEFAULT_MANAGEMENT_BIND_ADDRESS)
    if not isinstance(bind_address, str):
        raise TopologyReconcileError("runtime management API bind address is invalid")
    bind_address = bind_address.strip() or _DEFAULT_MANAGEMENT_BIND_ADDRESS
    if bind_address != "0.0.0.0":
        raise TopologyReconcileError(
            "managed split runtime requires management_api.bind_address 0.0.0.0"
        )
    port = management.get("port", _DEFAULT_MANAGEMENT_PORT)
    if not isinstance(port, int) or isinstance(port, bool):
        raise TopologyReconcileError("runtime management API port is invalid")
    if port == 0:
        port = _DEFAULT_MANAGEMENT_PORT
    if not _valid_port(port) or port in {50051, 9190}:
        raise TopologyReconcileError("runtime management API port is invalid")
    host_port = port + resolve_runtime_stack().port_offset
    if not _valid_port(host_port):
        raise TopologyReconcileError("runtime management API host port is invalid")
    return port, host_port


def _apply(
    state_path: Path,
    state: dict[str, Any],
    credential: str,
    *,
    management: tuple[int, int] | None = None,
) -> None:
    transitions = state["containers"]
    _validate_apply_preconditions(state, transitions, management)
    _preserve_storage_for_apply(transitions)
    _apply_storage_transitions(state_path, transitions)
    _replace_runtime_services(state_path, state, transitions, credential, management)


def _validate_apply_preconditions(
    state: dict[str, Any],
    transitions: list[dict[str, Any]],
    management: tuple[int, int] | None,
) -> None:
    for service in state["storage_before"]:
        _validate_storage_isolation(_storage_name(service))
    for transition in transitions:
        if transition["action"] == "repair":
            _validate_repair_transition(transition)
        elif transition["action"] == "replace":
            if transition["service"] == "router" and management is not None:
                _validate_cloneable_transition(transition, management=management)
            else:
                _validate_cloneable_transition(transition)


def _preserve_storage_for_apply(transitions: list[dict[str, Any]]) -> None:
    for transition in transitions:
        if transition["action"] in {"remove", "repair"}:
            _preserve_original(transition)


def _apply_storage_transitions(
    state_path: Path, transitions: list[dict[str, Any]]
) -> None:
    for transition in transitions:
        if transition["action"] == "add":
            _start_storage(transition["service"])
        elif transition["action"] == "repair":
            _repair_storage(state_path, transition)


def _replace_runtime_services(
    state_path: Path,
    state: dict[str, Any],
    transitions: list[dict[str, Any]],
    credential: str,
    management: tuple[int, int] | None,
) -> None:
    for service in _RUNTIME_SERVICES:
        transition = next(item for item in transitions if item["service"] == service)
        newly_preserved = _preserve_original(transition)
        if _container_status(transition["name"]) == "not found":
            _clone_preserved_container(
                state_path,
                transition,
                listeners=state["listeners"] if service == "envoy" else None,
                credential=credential if service == "router" else "",
                management=management if service == "router" else None,
            )
        elif newly_preserved:
            raise TopologyReconcileError(
                f"replacement container unexpectedly exists: {transition['name']}"
            )


def _rollback(state: dict[str, Any], *, restore_running: bool = True) -> None:
    transitions = state["containers"]

    # Tear down replacement ingress first. Starting either prior runtime before
    # its storage dependencies have been restored can make it exit immediately.
    for service in ("router", "envoy"):
        transition = next(item for item in transitions if item["service"] == service)
        name, backup = transition["name"], transition["backup_name"]
        if _container_status(backup) == "not found":
            if _container_status(name) == "not found":
                raise TopologyReconcileError(
                    f"both original and backup containers are missing: {name}"
                )
            continue
        _remove_if_present(name)

    # Restore the complete prior storage topology before restarting the Router.
    for transition in transitions:
        if transition["service"] not in _STORAGE:
            continue
        if transition["action"] == "add":
            _remove_if_present(transition["name"])
            continue
        _restore_preserved_transition(transition, restore_running=restore_running)

    # The Router is restored only after storage, and Envoy is restored last so
    # no request can enter a partially reconstructed stack.
    for service in ("router", "envoy"):
        transition = next(item for item in transitions if item["service"] == service)
        _restore_preserved_transition(transition, restore_running=restore_running)


def _restore_preserved_transition(
    transition: dict[str, Any], *, restore_running: bool
) -> None:
    name, backup = transition["name"], transition["backup_name"]
    if _container_status(backup) != "not found":
        _remove_if_present(name)
        _run(["rename", backup, name])
    elif _container_status(name) == "not found":
        raise TopologyReconcileError(
            f"both original and backup containers are missing: {name}"
        )
    if (
        restore_running
        and transition["was_running"]
        and _container_status(name) != "running"
    ):
        _run(["start", name])


def _commit(state: dict[str, Any]) -> None:
    _commit_with_expected_state(state, require_running=True)


def _commit_offline(state: dict[str, Any]) -> None:
    """Finish cleanup while runtime services are down after a verified commit."""
    _commit_with_expected_state(state, require_running=False)


def _commit_with_expected_state(
    state: dict[str, Any], *, require_running: bool
) -> None:
    def is_expected_target(status: str) -> bool:
        if require_running:
            return status == "running"
        return status in _OFFLINE_COMMIT_STATES

    for service in state.get("storage_after", []):
        name = _storage_name(service)
        status = _container_status(name)
        if not is_expected_target(status):
            mode = "running" if require_running else "present in a recoverable state"
            raise TopologyReconcileError(
                f"required managed storage is not {mode} during commit: {name}"
            )
        _validate_storage_isolation(name)
    for transition in state["containers"]:
        active_status = _container_status(transition["name"])
        if transition["action"] == "remove":
            if active_status != "not found":
                raise TopologyReconcileError(
                    f"removed container unexpectedly remains active: {transition['name']}"
                )
            continue
        if not is_expected_target(active_status):
            mode = "running" if require_running else "present in a recoverable state"
            raise TopologyReconcileError(
                f"activated container is not {mode} during commit: {transition['name']}"
            )
        if (
            transition["action"] == "repair"
            and _container_status(transition["backup_name"]) != "not found"
        ):
            _verify_storage_repair(transition)
    for transition in state["containers"]:
        backup = transition["backup_name"]
        if backup:
            _remove_if_present(backup)


def _preserve_original(transition: dict[str, Any]) -> bool:
    name, backup = transition["name"], transition["backup_name"]
    backup_status = _container_status(backup)
    if backup_status != "not found":
        return False
    status = _container_status(name)
    if status == "not found":
        raise TopologyReconcileError(f"managed container is missing: {name}")
    if status in {"running", "paused", "restarting"}:
        _run(["stop", name])
    _run(["rename", name, backup])
    return True


def _clone_preserved_container(
    state_path: Path,
    transition: dict[str, Any],
    *,
    listeners: list[dict[str, Any]] | None,
    credential: str | None,
    management: tuple[int, int] | None = None,
    snapshot: dict[str, Any] | None = None,
    preserve_container_identity: bool = False,
) -> None:
    del state_path  # Environment transport is an anonymous, inherited memfd.
    snapshot = snapshot or _inspect(transition["backup_name"])
    config, host, networks, image_id = _clone_snapshot_contract(snapshot, transition)
    command, primary_endpoint_network = _base_clone_command(
        transition, snapshot, config, host, networks
    )
    bindings = _clone_port_bindings(host, listeners, management)
    _append_port_bindings(command, bindings)
    _run_clone_command(command, config, credential, image_id)
    replacement = _inspect(transition["name"])
    _validate_clone_replacement(replacement, image_id, management)
    _connect_secondary_networks(
        transition, snapshot, networks, primary_endpoint_network
    )


def _run_clone_command(
    command: list[str],
    config: dict[str, Any],
    credential: str | None,
    image_id: str,
) -> None:
    env = _normalized_environment(config.get("Env") or [], credential)
    env_fd = _open_environment_memfd(env)
    try:
        command.extend(["--env-file", f"/proc/self/fd/{env_fd}"])
        for key, value in (config.get("Labels") or {}).items():
            if isinstance(key, str) and isinstance(value, str):
                command.extend(["--label", f"{key}={value}"])
        entrypoint = config.get("Entrypoint") or []
        command_args = list(config.get("Cmd") or [])
        if entrypoint:
            command.extend(["--entrypoint", str(entrypoint[0])])
            command_args = [str(item) for item in entrypoint[1:]] + command_args
        command.append(image_id)
        command.extend(str(item) for item in command_args)
        _run(command, pass_fds=(env_fd,))
    finally:
        os.close(env_fd)


def _connect_secondary_networks(
    transition: dict[str, Any],
    snapshot: dict[str, Any],
    networks: dict[str, Any],
    primary_endpoint_network: str,
) -> None:
    for network_name, endpoint in networks.items():
        if network_name == primary_endpoint_network:
            continue
        connect = ["network", "connect"]
        for alias in _preserved_network_aliases(endpoint, transition, snapshot):
            connect.extend(["--alias", alias])
        connect.extend([network_name, transition["name"]])
        _run(connect)


def _repair_storage(state_path: Path, transition: dict[str, Any]) -> None:
    """Clone a repair target from its journaled backup without changing data."""

    name, backup = transition["name"], transition["backup_name"]
    if _container_status(name) != "not found":
        raise TopologyReconcileError(
            f"managed {transition['service']} replacement already exists"
        )
    snapshot = _inspect(backup)
    _storage_contract(snapshot, transition)
    _clone_preserved_container(
        state_path,
        transition,
        listeners=None,
        credential=None,
        snapshot=snapshot,
        preserve_container_identity=True,
    )
    if _container_status(name) != "running":
        raise TopologyReconcileError(
            f"managed {transition['service']} repair exited before topology verification"
        )
    _verify_storage_repair(transition)


def _verify_storage_repair(transition: dict[str, Any]) -> None:
    expected = _storage_contract(_inspect(transition["backup_name"]), transition)
    actual = _storage_contract(_inspect(transition["name"]), transition)
    try:
        assert_storage_contract_preserved(expected, actual)
    except ValueError as error:
        raise TopologyReconcileError(
            f"managed {transition['service']} repair did not preserve its runtime contract: {error}"
        ) from error


def _storage_contract(snapshot: dict[str, Any], transition: dict[str, Any]):
    try:
        _validate_storage_snapshot_isolation(snapshot)
        return storage_clone_contract(
            snapshot,
            transition["service"],
            transient_names={transition["name"], transition["backup_name"]},
        )
    except ValueError as error:
        raise TopologyReconcileError(
            f"managed {transition['service']} snapshot is not safely cloneable: {error}"
        ) from error


def _validate_cloneable_transition(
    transition: dict[str, Any], *, management: tuple[int, int] | None = None
) -> None:
    name = transition["name"]
    backup = transition["backup_name"]
    source = backup if _container_status(backup) != "not found" else name
    if _container_status(source) == "not found":
        raise TopologyReconcileError(f"managed container is missing: {name}")
    try:
        snapshot = _inspect(source)
        validate_cloneable_container_snapshot(snapshot)
        if management is not None:
            _validate_router_management_isolation(snapshot, management)
    except ValueError as error:
        raise TopologyReconcileError(
            f"managed {transition['service']} container is not safely cloneable: {error}"
        ) from error


def _validate_repair_transition(transition: dict[str, Any]) -> None:
    name = transition["name"]
    backup = transition["backup_name"]
    source = backup if _container_status(backup) != "not found" else name
    if _container_status(source) == "not found":
        raise TopologyReconcileError(f"managed container is missing: {name}")
    _storage_contract(_inspect(source), transition)


def _start_storage(service: str) -> None:
    name = _storage_name(service)
    if _container_status(name) != "not found":
        raise TopologyReconcileError(
            f"unexpected managed {service} container appeared during activation"
        )
    layout = resolve_runtime_stack()
    if service == "redis":
        result = container_start_redis(
            layout.network_name, layout, reuse_existing=False
        )
    elif service == "postgres":
        result = container_start_postgres(
            layout.network_name, layout, reuse_existing=False
        )
    elif service == "milvus":
        result = container_start_milvus(
            layout.network_name,
            layout,
            state_root_dir="/app",
            host_hidden_state_dir=_host_hidden_state_dir(),
            reuse_existing=False,
        )
    else:
        raise TopologyReconcileError("unsupported managed storage backend")
    if result[0] != 0:
        raise TopologyReconcileError(f"failed to provision managed {service}")
    if _container_status(name) != "running":
        raise TopologyReconcileError(
            f"managed {service} exited before topology verification"
        )
    _validate_storage_isolation(name)


def _storage_name(service: str) -> str:
    layout = resolve_runtime_stack()
    return {
        "redis": layout.redis_container_name,
        "postgres": layout.postgres_container_name,
        "milvus": layout.milvus_container_name,
    }[service]


def _expected_container_names() -> dict[str, str]:
    layout = resolve_runtime_stack()
    return {
        "router": os.getenv(
            "VLLM_SR_ROUTER_CONTAINER_NAME", layout.router_container_name
        ).strip(),
        "envoy": os.getenv(
            "VLLM_SR_ENVOY_CONTAINER_NAME", layout.envoy_container_name
        ).strip(),
        "redis": layout.redis_container_name,
        "postgres": layout.postgres_container_name,
        "milvus": layout.milvus_container_name,
    }


def _host_hidden_state_dir() -> str:
    dashboard = os.getenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "").strip()
    if not dashboard:
        raise TopologyReconcileError(
            "managed Dashboard container identity is unavailable"
        )
    snapshot = _inspect(dashboard)
    mounts = snapshot.get("Mounts")
    if not isinstance(mounts, list):
        raise TopologyReconcileError("managed Dashboard state mount is invalid")
    matches = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == "/app/.vllm-sr"
    ]
    if len(matches) != 1:
        raise TopologyReconcileError("managed Dashboard state mount is unavailable")
    mount = matches[0]
    source = mount.get("Source")
    if (
        mount.get("Type") != "bind"
        or mount.get("RW") is not True
        or not isinstance(source, str)
        or not os.path.isabs(source)
        or os.path.normpath(source) != source
        or source == "/"
        or "\x00" in source
        or "\n" in source
    ):
        raise TopologyReconcileError("managed Dashboard state mount is unsafe")
    return source


def _read_credential() -> str:
    value = sys.stdin.readline(257).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise TopologyReconcileError("managed Router credential is invalid")
    return value


def _container_status(name: str) -> str:
    result = _run_raw(["inspect", "--format", "{{.State.Status}}", name])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).lower()
        if any(
            marker in detail
            for marker in (
                "no such container",
                "no such object",
                "no container with name or id",
            )
        ):
            return "not found"
        raise TopologyReconcileError(
            f"managed container status could not be inspected: {name}"
        )
    status = result.stdout.strip().lower()
    if status not in {
        "created",
        "restarting",
        "running",
        "removing",
        "paused",
        "exited",
        "dead",
    }:
        raise TopologyReconcileError("container status inspection is invalid")
    return status


def _validate_storage_isolation(name: str) -> None:
    _validate_storage_snapshot_isolation(_inspect(name))


def _validate_storage_snapshot_isolation(snapshot: dict[str, Any]) -> None:
    host = snapshot.get("HostConfig")
    network = snapshot.get("NetworkSettings")
    if not isinstance(host, dict) or not isinstance(network, dict):
        raise TopologyReconcileError("managed storage port inspection is invalid")
    if (
        not {"NetworkMode", "PublishAllPorts", "PortBindings"}.issubset(host)
        or "Ports" not in network
    ):
        raise TopologyReconcileError("managed storage port inspection is invalid")
    try:
        validate_storage_port_isolation(
            {
                "network_mode": host["NetworkMode"],
                "publish_all_ports": host["PublishAllPorts"],
                "configured": host["PortBindings"],
                "actual": network["Ports"],
            }
        )
    except ValueError as error:
        raise TopologyReconcileError(
            "managed storage is not isolated to host loopback"
        ) from error


def _inspect(name: str) -> dict[str, Any]:
    result = _run_raw(["inspect", name])
    if result.returncode != 0:
        raise TopologyReconcileError(
            f"managed container could not be inspected: {name}"
        )
    try:
        documents = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise TopologyReconcileError(
            "container inspection returned invalid JSON"
        ) from error
    if (
        not isinstance(documents, list)
        or len(documents) != 1
        or not isinstance(documents[0], dict)
    ):
        raise TopologyReconcileError(
            "container inspection returned an invalid document"
        )
    return documents[0]


def _remove_if_present(name: str) -> None:
    status = _container_status(name)
    if status == "not found":
        return
    if status in {"running", "paused", "restarting"}:
        _run(["stop", name])
    _run(["rm", name])


def _run(arguments: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
    result = _run_raw(arguments, pass_fds=pass_fds)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise TopologyReconcileError(
            f"container runtime command failed: {arguments[0]} ({detail})"
        )


def _run_raw(
    arguments: list[str], *, pass_fds: tuple[int, ...] = ()
) -> subprocess.CompletedProcess[str]:
    runtime = get_container_runtime()
    try:
        return subprocess.run(
            [runtime, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            pass_fds=pass_fds,
        )
    except subprocess.TimeoutExpired as error:
        raise TopologyReconcileError("container runtime command timed out") from error


def _require_regular_file(path: Path, limit: int) -> os.stat_result:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size > limit
    ):
        raise TopologyReconcileError("topology input is not a bounded regular file")
    return info


def _read_bounded_file(path: Path, limit: int) -> bytes:
    before = _require_regular_file(path, limit)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size > limit
        ):
            raise TopologyReconcileError("topology journal changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise TopologyReconcileError("topology journal exceeds its size limit")
        return data
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
