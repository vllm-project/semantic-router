"""Explicit one-shot schema migration for durable Management storage."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping

from cli.container_images import get_runtime_images
from cli.container_runtime import get_container_runtime
from cli.control_plane_deployment import (
    control_plane_store_references,
    runtime_capabilities,
)
from cli.utils import get_logger

log = get_logger(__name__)

MIGRATION_BINARY = "/usr/local/bin/management-migrate"
MIGRATION_ATTEMPTS = 12
MIGRATION_RETRY_SECONDS = 1.0


def build_management_migration_command(
    config: Mapping[str, object],
    *,
    env_vars: Mapping[str, str],
    network_name: str,
    router_image: str,
    container_runtime: str,
) -> list[str]:
    """Build a secret-safe one-shot migration command for Management storage."""

    references = control_plane_store_references(config)
    postgres = references.postgres
    if postgres is None:
        raise ValueError("Management PostgreSQL store is not configured")
    command = [
        container_runtime,
        "run",
        "--rm",
        "--network",
        network_name,
        "--entrypoint",
        MIGRATION_BINARY,
    ]
    if postgres.kind == "env":
        if not _secret_environment_value(postgres.value, env_vars):
            raise ValueError(
                "Management PostgreSQL DSN environment reference is not populated"
            )
        # Pass the name only. The credential value remains in the parent
        # environment and never enters argv or generated configuration.
        command.extend(["-e", postgres.value])
        migration_args = ["--dsn-env", postgres.value]
    else:
        if not os.path.isfile(postgres.value):
            raise ValueError(
                f"Management PostgreSQL DSN file does not exist: {postgres.value}"
            )
        command.extend(["-v", f"{postgres.value}:{postgres.value}:ro"])
        migration_args = ["--dsn-file", postgres.value]
    command.append(router_image)
    command.extend([*migration_args, "--timeout", "20s"])
    return command


def run_management_migration(
    config: Mapping[str, object],
    *,
    env_vars: Mapping[str, str],
    network_name: str,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    pull_policy: str | None,
) -> None:
    """Migrate durable desired state before starting any Router replica."""

    if not runtime_capabilities(config).durable_management:
        return
    images = get_runtime_images(
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=None,
        pull_policy=pull_policy,
        platform=env_vars.get("VLLM_SR_PLATFORM"),
        include_dashboard=False,
    )
    command = build_management_migration_command(
        config,
        env_vars=env_vars,
        network_name=network_name,
        router_image=images["router"],
        container_runtime=get_container_runtime(),
    )
    child_env = _migration_child_environment(config, env_vars)
    log.info("Applying Management-store schema...")
    for attempt in range(1, MIGRATION_ATTEMPTS + 1):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=child_env,
        )
        if result.returncode == 0:
            log.info("Management-store schema is current")
            return
        if attempt < MIGRATION_ATTEMPTS:
            time.sleep(MIGRATION_RETRY_SECONDS)
    raise RuntimeError(
        "Management-store schema migration failed; Router was not started"
    )


def _secret_environment_value(name: str, env_vars: Mapping[str, str]) -> str:
    value = env_vars.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _migration_child_environment(
    config: Mapping[str, object], env_vars: Mapping[str, str]
) -> dict[str, str]:
    """Build the migrator's exact child environment without global mutation.

    Values explicitly collected for runtime use are removed from the inherited
    host environment first. The one PostgreSQL DSN named by the durable store is
    then added back when migration uses an environment reference. Provider,
    Redis, Dashboard, and storage-password values cannot reach this child.
    """

    environment = {
        name: value for name, value in os.environ.items() if name not in env_vars
    }
    reference = control_plane_store_references(config).postgres
    if reference is None:
        raise ValueError("Management PostgreSQL store is not configured")
    if reference.kind == "env":
        value = _secret_environment_value(reference.value, env_vars)
        if not value:
            raise ValueError(
                "Management PostgreSQL DSN environment reference is not populated"
            )
        environment[reference.value] = value
    return environment
