"""Entry-point discovery for families, engines, accelerators and profiles.

Plugins come from the entry points of installed distributions. When the
runtime itself runs uninstalled from a source tree, the entry points its own
``pyproject.toml`` declares stand in for its distribution's, so the built-in
plugins are listed in one place.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

GROUPS = {
    "families": "vllm_srun.families",
    "engines": "vllm_srun.engines",
    "accelerators": "vllm_srun.accelerators",
    "profiles": "vllm_srun.profiles",
}
DISTRIBUTION = "vllm-srun"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class PluginError(RuntimeError):
    """An entry point that does not name a plugin of its group."""


def _bases() -> dict[str, type]:
    from .base import Accelerator, Engine, ModelFamily, Profile

    return {
        "families": ModelFamily,
        "engines": Engine,
        "accelerators": Accelerator,
        "profiles": Profile,
    }


@dataclass(frozen=True)
class PluginEntry:
    group: str
    name: str
    target: str
    distribution: str | None
    version: str | None

    def load(self) -> Any:
        """The plugin class; it must subclass its group's base and carry its entry point's name."""
        module_name, _, attribute = self.target.partition(":")
        loaded = getattr(importlib.import_module(module_name), attribute)
        base = _bases()[self.group]
        if not (isinstance(loaded, type) and issubclass(loaded, base)):
            raise PluginError(
                f"{GROUPS[self.group]} entry point {self.name!r} names "
                f"{self.target}, which is not a {base.__name__}"
            )
        if getattr(loaded, "name", None) != self.name:
            raise PluginError(
                f"{GROUPS[self.group]} entry point {self.name!r} names "
                f"{self.target}, whose name is {getattr(loaded, 'name', None)!r}"
            )
        return loaded

    def describe(self) -> dict[str, Any]:
        return {
            "group": GROUPS[self.group],
            "name": self.name,
            "distribution": self.distribution,
            "version": self.version,
            "capabilities": self.capabilities(),
        }

    def capabilities(self) -> dict[str, Any]:
        """The plugin class's capability descriptor; a plugin that fails to import reports why."""
        try:
            descriptor = getattr(self.load(), "descriptor", None)
            return dict(descriptor()) if callable(descriptor) else {}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class PluginConflictError(RuntimeError):
    pass


def _source_tree() -> list[tuple[str, str, str]]:
    """(group, name, target) from this package's pyproject.toml when it is not installed."""
    try:
        metadata.distribution(DISTRIBUTION)
        return []
    except metadata.PackageNotFoundError:
        pass
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 reads installed metadata only
        return []
    if not PYPROJECT.is_file():
        return []
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("project", {})
    if project.get("name") != DISTRIBUTION:
        return []
    return [
        (group, name, target)
        for group, entries in project.get("entry-points", {}).items()
        for name, target in entries.items()
    ]


@lru_cache(maxsize=1)
def discover() -> dict[str, dict[str, PluginEntry]]:
    """All plugins by kind; a name claimed with two different targets is refused."""
    found: dict[str, dict[str, PluginEntry]] = {kind: {} for kind in GROUPS}
    kinds = {group: kind for kind, group in GROUPS.items()}

    def add(plugin: PluginEntry) -> None:
        existing = found[plugin.group].get(plugin.name)
        if existing and existing.target != plugin.target:
            raise PluginConflictError(
                f"{GROUPS[plugin.group]}: {plugin.name!r} is registered by "
                f"{existing.distribution} and {plugin.distribution}"
            )
        found[plugin.group][plugin.name] = plugin

    for kind, group in GROUPS.items():
        for entry in metadata.entry_points(group=group):
            distribution = getattr(entry, "dist", None)
            add(
                PluginEntry(
                    group=kind,
                    name=entry.name,
                    target=entry.value,
                    distribution=distribution.name if distribution else None,
                    version=distribution.version if distribution else None,
                )
            )
    for group, name, target in _source_tree():
        if group in kinds:
            add(PluginEntry(kinds[group], name, target, DISTRIBUTION, None))
    return found


def plugin(kind: str, name: str) -> PluginEntry:
    entries = discover()[kind]
    if not entries:
        raise KeyError(f"no {kind} are installed; install {DISTRIBUTION}")
    if name not in entries:
        raise KeyError(
            f"no {kind[:-1]} plugin named {name!r}; available: {', '.join(sorted(entries))}"
        )
    return entries[name]


def instantiate(kind: str, name: str) -> Any:
    return plugin(kind, name).load()()


def names(kind: str) -> list[str]:
    return sorted(discover()[kind])
