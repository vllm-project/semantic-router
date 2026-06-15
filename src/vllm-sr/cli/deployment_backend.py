"""Deployment backend protocol for unified Docker and Kubernetes management."""

from __future__ import annotations

from typing import Any, Protocol


class DeploymentBackend(Protocol):
    """Interface that all deployment targets (Docker, Kubernetes) must implement."""

    def deploy(
        self,
        config_file: str,
        env_vars: dict[str, str] | None = None,
        *,
        image: str | None = None,
        pull_policy: str | None = None,
        enable_observability: bool = True,
        **kwargs: Any,
    ) -> None:
        """Deploy the full vLLM Semantic Router stack."""
        ...

    def teardown(self) -> None:
        """Tear down the running stack."""
        ...

    def logs(self, service: str, follow: bool = False) -> None:
        """Stream logs for a specific service."""
        ...

    def status(self, service: str = "all") -> None:
        """Show status of deployed services."""
        ...

    def get_dashboard_url(self) -> str | None:
        """Return the dashboard URL, or None if unavailable."""
        ...

    def is_running(self) -> bool:
        """Return True if the stack is currently running."""
        ...


VALID_TARGETS = ("docker", "k8s")
DEFAULT_TARGET = "docker"


def resolve_target(target: str | None) -> str:
    """Resolve and validate the deployment target string.

    Falls back to DEFAULT_TARGET when *target* is None.
    """
    if target is None:
        return DEFAULT_TARGET
    normalised = target.lower().strip()
    if normalised not in VALID_TARGETS:
        raise ValueError(
            f"Invalid deployment target '{target}'. "
            f"Must be one of: {', '.join(VALID_TARGETS)}"
        )
    return normalised
