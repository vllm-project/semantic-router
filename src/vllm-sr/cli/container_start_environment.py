"""Process isolation and rollback helpers for container startup."""

import os
from collections.abc import Mapping

from cli.container_services import (
    container_remove_container,
    container_status,
    container_stop_container,
)


def service_child_environment(
    service_name: str,
    router_secret_values: Mapping[str, str],
    dashboard_secret_values: Mapping[str, str],
) -> dict[str, str] | None:
    """Build one child environment without leaking secrets across services."""

    blocked = set(router_secret_values) | set(dashboard_secret_values)
    if not blocked:
        return None
    environment = {
        name: value for name, value in os.environ.items() if name not in blocked
    }
    if service_name == "router":
        environment.update(router_secret_values)
    elif service_name == "dashboard":
        environment.update(dashboard_secret_values)
    return environment


def cleanup_started_containers(container_names: list[str]) -> None:
    """Remove only containers started by the current launch transaction."""

    for container_name in reversed(container_names):
        if container_status(container_name) == "running":
            container_stop_container(container_name)
        container_remove_container(container_name)
