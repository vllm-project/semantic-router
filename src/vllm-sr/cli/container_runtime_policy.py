"""Docker lifecycle and health policy for CLI-managed services."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from cli.consts import (
    CONTAINER_RUNTIME_DOCKER,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_ENVOY_PORT,
)

DOCKER_RESTART_POLICY = "unless-stopped"


@dataclass(frozen=True, slots=True)
class DockerHealthcheck:
    """One Docker-native healthcheck, expressed without secret values."""

    command: str
    interval: str = "10s"
    timeout: str = "5s"
    retries: int = 12
    start_period: str = "30s"

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("Docker healthcheck command must not be empty")
        if self.retries < 1:
            raise ValueError("Docker healthcheck retries must be positive")


def append_docker_runtime_policy(
    command: list[str], runtime: str, healthcheck: DockerHealthcheck
) -> None:
    """Append Docker-only lifecycle flags without changing Podman semantics."""

    if runtime != CONTAINER_RUNTIME_DOCKER:
        return
    command.extend(
        [
            "--restart",
            DOCKER_RESTART_POLICY,
            "--health-cmd",
            healthcheck.command,
            "--health-interval",
            healthcheck.interval,
            "--health-timeout",
            healthcheck.timeout,
            "--health-retries",
            str(healthcheck.retries),
            "--health-start-period",
            healthcheck.start_period,
        ]
    )


def router_healthcheck(
    management_port: int,
    *,
    tls_enabled: bool,
) -> DockerHealthcheck:
    """Probe Router serving readiness through its private operational listener."""

    scheme = "https" if tls_enabled else "http"
    arguments = ["curl", "-fsS"]
    if tls_enabled:
        # Runtime readiness validates the configured trust chain separately.
        # This Docker probe stays on container loopback.
        arguments.append("-k")
    arguments.append(f"{scheme}://127.0.0.1:{management_port}/ready")
    return DockerHealthcheck(
        command=shlex.join(arguments),
        start_period="30m",
    )


def envoy_healthcheck() -> DockerHealthcheck:
    """Probe Envoy's admin readiness endpoint without adding curl to its image."""

    request = (
        f"exec 3<>/dev/tcp/127.0.0.1/{DEFAULT_ENVOY_PORT}; "
        "printf 'GET /ready HTTP/1.1\\r\\nHost: localhost\\r\\n"
        "Connection: close\\r\\n\\r\\n' >&3; "
        "grep -Eq '^HTTP/1\\.[01] 200' <&3"
    )
    return DockerHealthcheck(
        command=shlex.join(["timeout", "3", "bash", "-c", request]),
    )


def dashboard_healthcheck() -> DockerHealthcheck:
    """Probe the Dashboard readiness endpoint served inside its container."""

    return DockerHealthcheck(
        command=shlex.join(
            [
                "curl",
                "-fsS",
                f"http://127.0.0.1:{DEFAULT_DASHBOARD_PORT}/healthz",
            ]
        ),
        start_period="2m",
    )


def postgres_healthcheck(user: str, database: str) -> DockerHealthcheck:
    """Probe the CLI-managed PostgreSQL database without a password argument."""

    return DockerHealthcheck(
        command=shlex.join(["pg_isready", "-q", "-U", user, "-d", database])
    )


def redis_healthcheck(config_file: str) -> DockerHealthcheck:
    """Probe Redis/Valkey while keeping its password out of Docker metadata."""

    script = (
        'password="$(sed -n '
        "'s/^requirepass[[:space:]][[:space:]]*//p' "
        f'{shlex.quote(config_file)})"; '
        'test -n "$password"; '
        'REDISCLI_AUTH="$password" redis-cli --no-auth-warning ping | '
        "grep -qx PONG"
    )
    return DockerHealthcheck(command=script)
