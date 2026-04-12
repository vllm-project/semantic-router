"""Config-driven storage backend provisioning for vllm-sr serve."""

from __future__ import annotations

from cli.docker_services import (
    docker_start_milvus,
    docker_start_postgres,
    docker_start_redis,
)
from cli.runtime_stack import RuntimeStackLayout
from cli.service_defaults import detect_canonical_storage_backends
from cli.utils import get_logger

log = get_logger(__name__)


def detect_required_backends(config: dict) -> set[str]:
    """Read store_backend values from the config and return backends that need provisioning.

    Reads from the canonical v0.3 path: global.services.<key>.store_backend and
    falls back to router-owned canonical defaults for local serve workflows.
    Returns only backends the CLI knows how to provision (redis, postgres).
    """
    return detect_canonical_storage_backends(config)


def start_storage_backends(
    required_backends: set[str],
    network_name: str,
    stack_layout: RuntimeStackLayout,
) -> set[str]:
    """Start Docker containers for the required storage backends.

    Returns the set of backends that were actually started.
    """
    if not required_backends:
        return set()

    started: set[str] = set()
    log.info(
        f"Storage backends required by config: {', '.join(sorted(required_backends))}"
    )

    if "redis" in required_backends:
        _start_backend("Redis", lambda: docker_start_redis(network_name, stack_layout))
        started.add("redis")

    if "postgres" in required_backends:
        _start_backend(
            "Postgres", lambda: docker_start_postgres(network_name, stack_layout)
        )
        started.add("postgres")

    if "milvus" in required_backends:
        _start_backend(
            "Milvus", lambda: docker_start_milvus(network_name, stack_layout)
        )
        started.add("milvus")

    return started


def provision_storage_backends(
    config: dict,
    network_name: str,
    stack_layout: RuntimeStackLayout,
) -> set[str]:
    """Detect which storage backends the config requires and start them."""
    required = detect_required_backends(config)
    return start_storage_backends(required, network_name, stack_layout)


def _start_backend(name: str, starter) -> None:
    return_code, _stdout, stderr = starter()
    if return_code != 0:
        log.error(f"Failed to start {name}: {stderr}")
        raise SystemExit(1)
    log.info(f"{name} started successfully")
