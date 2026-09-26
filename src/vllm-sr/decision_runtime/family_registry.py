"""Lightweight registration of installed Decision model families.

This module is safe to import from the CLI: a family implementation (and its
optional inference framework) is loaded only when a profile or resident model
actually needs it. Adding a family requires one registration and an owned
adapter, rather than new branches in assembly and row scheduling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .families.base import DecisionFamilyAdapter


@dataclass(frozen=True, slots=True)
class FamilyRegistration:
    family: str
    catalog_family: str
    profile_directory: str
    manifest_paths: frozenset[str]
    adapter_module: str
    image_environment: str
    rocm_target: str | None
    cpu_below_billions: float | None

    @property
    def container_python(self) -> str:
        return f"/opt/vllm-sr/venvs/{self.image_environment}/bin/python"


_FAMILIES = (
    FamilyRegistration(
        "vela",
        "decision-encoder",
        "vela",
        frozenset({"native/MANIFEST.json"}),
        "decision_runtime.families.vela",
        "vela",
        "gfx942",
        1.0,
    ),
    FamilyRegistration(
        "qwen3.5",
        "decision-qwen3.5",
        "qwen35",
        frozenset({"MODEL_MANIFEST.json", "bundle-manifest.json"}),
        "decision_runtime.families.qwen35",
        "qwen35",
        "gfx942",
        1.0,
    ),
)
_BY_FAMILY = {registration.family: registration for registration in _FAMILIES}
_BY_CATALOG = {registration.catalog_family: registration for registration in _FAMILIES}
_IMAGE_ENV_NAME = re.compile(r"[a-z][a-z0-9_-]*")
if (
    len(_BY_FAMILY) != len(_FAMILIES)
    or len(_BY_CATALOG) != len(_FAMILIES)
    or len({item.profile_directory for item in _FAMILIES}) != len(_FAMILIES)
    or len({item.image_environment for item in _FAMILIES}) != len(_FAMILIES)
    or any(not item.manifest_paths for item in _FAMILIES)
    or len({path for item in _FAMILIES for path in item.manifest_paths})
    != sum(len(item.manifest_paths) for item in _FAMILIES)
    or any(
        _IMAGE_ENV_NAME.fullmatch(item.image_environment) is None for item in _FAMILIES
    )
):
    raise RuntimeError("Decision family registrations must have unique identities")


def registered_families() -> tuple[FamilyRegistration, ...]:
    return _FAMILIES


def family_registration(family: str) -> FamilyRegistration | None:
    return _BY_FAMILY.get(family)


def catalog_family_registration(catalog_family: str) -> FamilyRegistration | None:
    return _BY_CATALOG.get(catalog_family)


def family_adapter(family: str) -> DecisionFamilyAdapter:
    registration = family_registration(family)
    if registration is None:
        raise ValueError(f"Decision model family {family!r} has no owned adapter")
    adapter: DecisionFamilyAdapter = import_module(registration.adapter_module).ADAPTER
    if adapter.family != family:
        raise ValueError(f"Decision model family {family!r} adapter identity drifted")
    return adapter
