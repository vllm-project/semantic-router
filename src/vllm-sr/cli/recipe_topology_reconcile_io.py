"""How the reconciler reads its inputs and drives the container runtime.

Everything that crosses the process boundary is here: the bounded, symlink-safe
reads of the journals and configs Dashboard hands over, and the one place a
container-runtime command is actually spawned. Both are attack surface -- the
reconciler runs against files and a runtime socket it does not own -- so they
are read once through a checked path and never re-opened by name, and every
failure raises rather than returning a partial answer.

The reconciler keeps its own thin `_run` on top of `_run_raw`, so a test that
substitutes the transport still intercepts every command the reconciler issues.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import yaml

from cli.container_runtime import get_container_runtime
from cli.recipe_topology_contract import (
    _DEFAULT_MANAGEMENT_BIND_ADDRESS,
    _DEFAULT_MANAGEMENT_PORT,
    TopologyReconcileError,
    _valid_port,
)
from cli.runtime_stack import resolve_runtime_stack

# A managed reconcile pulls or renames containers, so the bound is generous;
# it exists only so a wedged runtime socket cannot hang the activation forever.
RUNTIME_COMMAND_TIMEOUT_SECONDS = 120


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
            timeout=RUNTIME_COMMAND_TIMEOUT_SECONDS,
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
