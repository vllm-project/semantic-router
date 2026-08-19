"""Data contracts shared by built-in model catalog modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelCatalogError(ValueError):
    """A packaged catalog is invalid or cannot satisfy the requested operation."""


@dataclass(frozen=True)
class CatalogComponentVersions:
    """CLI and Router versions evaluated against a catalog independently."""

    cli: str
    router: str


@dataclass(frozen=True)
class CatalogCompatibility:
    compatible: bool
    reason: str


@dataclass(frozen=True)
class CatalogModel:
    id: str
    display_name: str
    description: str
    kind: str
    family: str
    generation: int
    policy_version: str
    asset: str
    entrypoint: str
    recipe: str
    protocols: tuple[str, ...]
    traits: tuple[str, ...]
    roles: tuple[dict[str, Any], ...]
    verification: dict[str, str]
    catalog_version: str
    channel: str
    compatibility: CatalogCompatibility
    enabled_by_default: bool
    default: bool

    @property
    def verified(self) -> bool:
        """Return whether provenance binds this model to its exact catalog asset.

        Verification and runtime compatibility are independent axes: an
        authentic catalog entry can be incompatible with the installed CLI or
        Router, and an unverified entry must never become verified merely by
        satisfying a version range.
        """

        return bool(
            self.verification.get("authority") and self.verification.get("asset_sha256")
        )


@dataclass(frozen=True)
class ModelCatalog:
    version: str
    channel: str
    default_model: str
    enabled_models: tuple[str, ...]
    assets: dict[str, dict[str, str]]
    models: tuple[CatalogModel, ...]


@dataclass(frozen=True)
class MaterializedCatalog:
    """One deterministic runtime-config projection of a packaged catalog."""

    catalog: ModelCatalog
    enabled_models: tuple[str, ...]
    default_model: str
    document: dict[str, Any]
