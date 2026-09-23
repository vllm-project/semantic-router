"""Injectable Decision model runtime catalog contract.

The built-in model catalog is the sole owner of Decision model identities and
revisions. This module intentionally contains no model inventory. A stacked
runtime implementation adapts the catalog's provider-model resolver into this
launch contract; tests use small explicit fixtures.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

SUPPORTED_DECISION_BACKENDS = ("auto", "rocm", "cuda")
RESOLVED_DECISION_BACKENDS = frozenset({"rocm", "cuda", "mlx"})
CATALOG_ADAPTER_MODULE = "cli.decision_runtime.catalog_adapter"


class DecisionCatalogError(ValueError):
    """The runtime catalog cannot resolve an exact Decision model launch."""


@dataclass(frozen=True)
class DecisionRuntimeRequest:
    """User intent passed to the sole catalog/runtime resolver."""

    model: str
    revision: str | None
    backend: str
    dtype: str | None
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


class DecisionCatalogResolver(Protocol):
    """Resolve exact model identity plus its backend launch profile."""

    def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
        """Return one immutable, fully qualified runtime launch."""


def default_catalog_resolver() -> DecisionCatalogResolver:
    """Load the stacked runtime-catalog adapter without owning a second catalog.

    The lifecycle foundation deliberately lands independently from the Decision
    catalog/runtime implementation. Once stacked, that implementation provides
    ``cli.decision_runtime.catalog_adapter.get_catalog_resolver``.
    """

    try:
        adapter = importlib.import_module(CATALOG_ADAPTER_MODULE)
    except ModuleNotFoundError as error:
        if error.name != CATALOG_ADAPTER_MODULE:
            raise
        raise DecisionCatalogError(
            "Decision runtime catalog support is not included in this build. "
            "Install a build containing the Decision model runtime catalog."
        ) from error
    factory = getattr(adapter, "get_catalog_resolver", None)
    if not callable(factory):
        raise DecisionCatalogError(
            "The installed Decision runtime catalog adapter is malformed: "
            "get_catalog_resolver is missing."
        )
    resolver = factory()
    if not callable(getattr(resolver, "resolve", None)):
        raise DecisionCatalogError(
            "The installed Decision runtime catalog adapter returned an invalid "
            "resolver."
        )
    return resolver


def runtime_catalog_available() -> bool:
    """Return whether the integrated Decision runtime adapter is installed."""

    try:
        if importlib.util.find_spec(CATALOG_ADAPTER_MODULE) is None:
            return False
        adapter = importlib.import_module(CATALOG_ADAPTER_MODULE)
    except (ImportError, AttributeError, ValueError):
        return False
    return callable(getattr(adapter, "get_catalog_resolver", None))
