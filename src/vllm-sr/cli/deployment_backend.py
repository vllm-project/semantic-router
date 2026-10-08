"""Deployment targets: where `vllm-sr` runs a stack."""

from __future__ import annotations

from cli.utils import get_logger

log = get_logger(__name__)

TARGET_DOCKER = "docker"
TARGET_KUBERNETES = "kubernetes"
VALID_TARGETS = (TARGET_DOCKER, TARGET_KUBERNETES)
DEFAULT_TARGET = TARGET_DOCKER
# v0.4.0 released the Kubernetes target as `k8s`; it stays one more release.
TARGET_ALIASES = {"k8s": TARGET_KUBERNETES}


def resolve_target(target: str | None) -> str:
    """Resolve and validate the deployment target string.

    Falls back to DEFAULT_TARGET when *target* is None.
    """
    if target is None:
        return DEFAULT_TARGET
    normalised = target.lower().strip()
    if normalised in TARGET_ALIASES:
        renamed = TARGET_ALIASES[normalised]
        log.warning(
            f"--target {normalised} is renamed to --target {renamed}; "
            f"{normalised} keeps working for this release only"
        )
        return renamed
    if normalised not in VALID_TARGETS:
        raise ValueError(
            f"Invalid deployment target '{target}'. "
            f"Must be one of: {', '.join(VALID_TARGETS)}"
        )
    return normalised
