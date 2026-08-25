"""Issue the assembled runtime container commands and unwind on failure.

Command assembly lives in ``container_start``; this module only runs what it
produced. The split exists because a service is no longer one command: Router
is created, attached to the data network, and started, and the rollback and
credential-handling rules that come with that are easier to state on their own.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping

from cli.container_services import (
    container_remove_container,
    container_status,
    container_stop_container,
)
from cli.container_start_environment import service_child_environment
from cli.utils import get_logger

log = get_logger(__name__)


def run_container_specs(
    container_specs,
    *,
    router_secret_values: Mapping[str, str],
    dashboard_secret_values: Mapping[str, str],
):
    """Bring up each service in order, unwinding the stack on the first failure.

    A service can need more than one command, so a container is registered for
    rollback as soon as its creating command succeeds. Registering any earlier
    would put a name on the rollback list that this run may not own: a
    `create` that fails because the name is already taken would otherwise make
    the unwind stop and remove somebody else's container.
    """

    started_containers: list[str] = []
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    for service_name, container_name, commands in container_specs:
        log.info(f"Starting {service_name} container: {container_name}")
        return_code, stdout, stderr = _run_service_commands(
            commands,
            service_name,
            router_secret_values,
            dashboard_secret_values,
            on_created=lambda name=container_name: started_containers.append(name),
        )
        if stdout:
            stdout_chunks.append(stdout)
        if stderr:
            stderr_chunks.append(stderr)
        if return_code != 0:
            _cleanup_started_containers(started_containers)
            return (return_code, "\n".join(stdout_chunks), "\n".join(stderr_chunks))

    return (0, "\n".join(stdout_chunks), "\n".join(stderr_chunks))


def _run_service_commands(
    commands,
    service_name: str,
    router_secret_values: Mapping[str, str],
    dashboard_secret_values: Mapping[str, str],
    *,
    on_created,
):
    """Run one service's commands in order and stop at the first failure.

    *on_created* fires once the first command returns successfully, which is
    the moment the container exists and becomes this run's to unwind.
    """

    # Only the creating command resolves the inheriting `-e NAME` flags, so it
    # is the only child that is handed the credential values.
    creation_env = service_child_environment(
        service_name, router_secret_values, dashboard_secret_values
    )
    followup_env = service_child_environment(
        "followup", router_secret_values, dashboard_secret_values
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    for index, cmd in enumerate(commands):
        log.debug(f"Container command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=creation_env if index == 0 else followup_env,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                stdout_chunks.append(exc.stdout)
            stderr_chunks.append(exc.stderr)
            return (exc.returncode, "\n".join(stdout_chunks), "\n".join(stderr_chunks))
        if index == 0:
            on_created()
        if result.stdout:
            stdout_chunks.append(result.stdout)
        if result.stderr:
            stderr_chunks.append(result.stderr)

    return (0, "\n".join(stdout_chunks), "\n".join(stderr_chunks))


def _cleanup_started_containers(container_names: list[str]) -> None:
    for container_name in reversed(container_names):
        status = container_status(container_name)
        if status == "not found":
            continue
        if status == "running":
            container_stop_container(container_name)
        container_remove_container(container_name)
