"""Deployment backend protocol for unified Docker and Kubernetes management."""

from __future__ import annotations

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
