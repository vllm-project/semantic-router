"""Validated journal contract shared by Recipe topology reconciliation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOPOLOGY_SCHEMA = "vllm-sr/recipe-activation-topology/v1"
MAX_TOPOLOGY_BYTES = 256 * 1024
MANAGEMENT_CREDENTIAL_ENV = "VLLM_SR_DASHBOARD_RECIPE_TOKEN"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "transaction_id",
    "plan_digest",
    "listeners",
    "storage_before",
    "storage_after",
    "containers",
    "credential_env",
}
_LISTENER_FIELDS = {"name", "address", "port", "host_port"}
_TRANSITION_FIELDS = {
    "service",
    "name",
    "backup_name",
    "action",
    "was_running",
}
_SERVICES = {"router", "envoy", "redis", "postgres", "milvus"}
_STORAGE = {"redis", "postgres", "milvus"}
_OFFLINE_COMMIT_STATES = {"running", "created", "exited", "paused", "dead"}
_PUBLISHABLE_LISTENER_ADDRESSES = {"0.0.0.0", "::", "127.0.0.1", "::1"}
_DEFAULT_MANAGEMENT_BIND_ADDRESS = "127.0.0.1"
_DEFAULT_MANAGEMENT_PORT = 8080
_IPV6_VERSION = 6
_MAX_PORT = 65_535
_RUNTIME_SERVICES = ("router", "envoy")


class TopologyReconcileError(RuntimeError):
    """A topology transaction could not be applied or restored safely."""


def load_topology_state(
    path: Path,
    *,
    read_bounded_file: Callable[[Path, int], bytes],
    expected_container_names: dict[str, str],
) -> dict[str, Any]:
    encoded = read_bounded_file(path, MAX_TOPOLOGY_BYTES)
    try:
        state = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TopologyReconcileError("topology journal is invalid") from error
    if not isinstance(state, dict) or set(state) != _TOP_LEVEL_FIELDS:
        raise TopologyReconcileError("topology journal has an invalid field set")
    transaction_id = _validate_state_identity(path, state)
    _validate_storage(state.get("storage_before"))
    _validate_storage(state.get("storage_after"))
    _validate_listeners(state.get("listeners"))
    transitions = _validate_transitions(
        state.get("containers"), transaction_id, expected_container_names
    )
    _validate_storage_transitions(
        transitions, set(state["storage_before"]), set(state["storage_after"])
    )
    return state


def _validate_state_identity(path: Path, state: dict[str, Any]) -> str:
    transaction_id = state.get("transaction_id")
    if (
        state.get("schema_version") != TOPOLOGY_SCHEMA
        or not isinstance(transaction_id, str)
        or not _TRANSACTION_ID.fullmatch(transaction_id)
        or not isinstance(state.get("plan_digest"), str)
        or not _DIGEST.fullmatch(state["plan_digest"])
        or path.name != "topology.json"
        or path.parent.name != transaction_id
    ):
        raise TopologyReconcileError("topology journal identity is invalid")
    credential_env = state.get("credential_env")
    if credential_env not in ("", MANAGEMENT_CREDENTIAL_ENV):
        raise TopologyReconcileError("topology credential binding is invalid")
    return transaction_id


def _validate_listeners(listeners: object) -> None:
    if not isinstance(listeners, list) or not listeners:
        raise TopologyReconcileError("topology listeners are invalid")
    listener_names: set[str] = set()
    host_bindings: list[tuple[str, int]] = []
    listener_bindings: set[tuple[str, int, int]] = set()
    for listener in listeners:
        if (
            not isinstance(listener, dict)
            or set(listener) != _LISTENER_FIELDS
            or not isinstance(listener.get("name"), str)
            or not listener["name"].strip()
            or not isinstance(listener.get("address"), str)
            or listener["address"] not in _PUBLISHABLE_LISTENER_ADDRESSES
            or not _valid_port(listener.get("port"))
            or not _valid_port(listener.get("host_port"))
        ):
            raise TopologyReconcileError("topology listener is invalid")
        host_binding = (listener["address"], listener["host_port"])
        listener_binding = (
            listener["address"],
            listener["port"],
            listener["host_port"],
        )
        if (
            listener["name"] in listener_names
            or any(
                _listener_bindings_conflict(host_binding, existing)
                for existing in host_bindings
            )
            or listener_binding in listener_bindings
        ):
            raise TopologyReconcileError("topology listeners contain a conflict")
        listener_names.add(listener["name"])
        host_bindings.append(host_binding)
        listener_bindings.add(listener_binding)


def _listener_bindings_conflict(left: tuple[str, int], right: tuple[str, int]) -> bool:
    if left[1] != right[1]:
        return False
    left_family = _IPV6_VERSION if ":" in left[0] else 4
    right_family = _IPV6_VERSION if ":" in right[0] else 4
    if left_family != right_family:
        return False
    wildcard = "::" if left_family == _IPV6_VERSION else "0.0.0.0"
    return left[0] == right[0] or left[0] == wildcard or right[0] == wildcard


def _validate_transitions(
    transitions: object,
    transaction_id: str,
    expected_container_names: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(transitions, list):
        raise TopologyReconcileError("topology transitions are invalid")
    identities: set[str] = set()
    for transition in transitions:
        _validate_transition(transition, transaction_id, expected_container_names)
        for identity in (transition["name"], transition["backup_name"]):
            if not identity:
                continue
            if identity in identities:
                raise TopologyReconcileError(
                    "topology contains duplicate container identities"
                )
            identities.add(identity)
    return transitions


def _validate_storage(value: object) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or item not in _STORAGE for item in value)
        or len(value) != len(set(value))
    ):
        raise TopologyReconcileError("managed storage set is invalid")


def _validate_transition(
    transition: object,
    transaction_id: str,
    expected_container_names: dict[str, str],
) -> None:
    if not isinstance(transition, dict) or set(transition) != _TRANSITION_FIELDS:
        raise TopologyReconcileError("topology transition has an invalid field set")
    service = transition.get("service")
    name = transition.get("name")
    backup = transition.get("backup_name")
    action = transition.get("action")
    if (
        service not in _SERVICES
        or not isinstance(name, str)
        or not _CONTAINER_NAME.fullmatch(name)
        or not isinstance(backup, str)
        or (backup and not _CONTAINER_NAME.fullmatch(backup))
        or action not in {"replace", "add", "remove", "repair"}
        or not isinstance(transition.get("was_running"), bool)
        or (action == "add") != (backup == "")
        or (backup and not backup.endswith("-recipe-backup-" + transaction_id[:12]))
    ):
        raise TopologyReconcileError("topology transition is invalid")
    if service in {"router", "envoy"} and action != "replace":
        raise TopologyReconcileError("runtime service transition must replace")
    if service in _STORAGE and action == "replace":
        raise TopologyReconcileError("storage transition cannot replace")
    expected = expected_container_names.get(service)
    if name != expected:
        raise TopologyReconcileError(
            "topology transition targets an unmanaged container"
        )


def _validate_storage_transitions(
    transitions: list[dict[str, Any]], before: set[str], after: set[str]
) -> None:
    by_service = {transition["service"]: transition for transition in transitions}
    if "router" not in by_service or "envoy" not in by_service:
        raise TopologyReconcileError("runtime topology transitions are incomplete")
    for service in _STORAGE:
        transition = by_service.get(service)
        expected = None
        if service not in before and service in after:
            expected = "add"
        elif service in before and service not in after:
            expected = "remove"
        elif service in before and service in after and transition is not None:
            expected = "repair"
        if (transition is None) != (expected is None) or (
            transition is not None and transition["action"] != expected
        ):
            raise TopologyReconcileError(
                "storage transition does not match managed set"
            )


def _valid_port(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= _MAX_PORT
    )
