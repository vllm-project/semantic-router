"""Decision model runtime catalog contract and launch request types.

The built-in model catalog owns model identities and revisions. This module
contains no second model inventory; the adapter resolves catalog entries into
launch descriptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

SUPPORTED_DECISION_BACKENDS = ("auto", "rocm", "cuda", "cpu")
RESOLVED_DECISION_BACKENDS = frozenset({"rocm", "cuda", "cpu", "mlx"})


class DecisionCatalogError(ValueError):
    """The runtime catalog cannot resolve an exact Decision model launch."""


@dataclass(frozen=True)
class DecisionRuntimeMount:
    """One read-only host artifact tree exposed to an OCI runtime."""

    source: str
    target: str


@dataclass(frozen=True)
class DecisionRuntimeRequest:
    """User intent passed to the sole catalog/runtime resolver."""

    model: str
    revision: str | None
    backend: str
    image: str | None
    max_batch: int | None
    max_concurrency: int | None
    max_queue: int | None


@dataclass(frozen=True)
class ResolvedDecisionRuntime:
    """Immutable launch description returned by a catalog adapter."""

    canonical_model: str
    revision: str
    backend: str
    dtype: str
    image: str
    artifact_digest: str
    max_batch: int
    max_concurrency: int
    max_queue: int
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[DecisionRuntimeMount, ...] = ()
    container_port: int = 8000
    health_path: str = "/ready"
    api_path: str = "/v1/systemone"

    def __post_init__(self) -> None:
        """Detach the frozen value object from resolver-owned mutable mappings."""

        try:
            environment = dict(self.environment.items())
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError(
                "Decision runtime environment must be a mapping."
            ) from error
        object.__setattr__(self, "environment", MappingProxyType(environment))
        try:
            mounts = tuple(self.mounts)
        except TypeError as error:
            raise TypeError("Decision runtime mounts must be an iterable.") from error
        object.__setattr__(self, "mounts", mounts)


class DecisionCatalogResolver(Protocol):
    """Resolve exact model identity plus its backend launch profile."""

    def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
        """Return one immutable, fully qualified runtime launch."""


def default_catalog_resolver() -> DecisionCatalogResolver:
    """Construct the integrated catalog resolver on demand.

    The import stays local because the adapter depends on the request types in
    this module; every Decision-enabled CLI build includes that adapter.
    """

    from .catalog_adapter import get_catalog_resolver  # noqa: PLC0415

    return get_catalog_resolver()
