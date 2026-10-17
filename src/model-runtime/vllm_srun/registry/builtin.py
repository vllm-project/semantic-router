"""First-party models the runtime serves out of the box, pinned by revision and identity.

Each family plugin names its table (``ModelFamily.builtin_table``, a module
whose ``MODELS`` lists its pinned packages; the built-in families keep theirs
in ``registry/tables``). The runtime reads the tables of the installed
families once, at first use, and again only after plugin discovery is
refreshed; families are taken in name order, each table in its own order.
A new model revision is a new entry; a built-in model is never resolved
through a moving branch. Golden requests gate readiness; their reference
answers per device class live in ``golden_answers*.json``, recorded with
``tools/golden_answers.py`` for the pinned revision. ``kernel_choices.json``
holds, per device class, the autotuned kernel configurations the released
runtime ran with (``tools/kernel_choices.py``); the runtime pins them so
every process computes the released numerics.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from ..plugins import registry
from .tables.common import ORG, BuiltinModel, with_recorded

__all__ = [
    "ORG",
    "BuiltinModel",
    "all_models",
    "by_identity",
    "golden",
    "kernel_choices",
    "lookup",
    "with_recorded",
]

log = logging.getLogger("vllm_srun")


@dataclass(frozen=True)
class _Table:
    """The built-in models of the installed families, indexed."""

    models: tuple[BuiltinModel, ...]
    by_repo: dict[str, BuiltinModel]
    by_name: dict[str, BuiltinModel]
    by_identity: dict[str, BuiltinModel]


_cached: tuple[object, _Table] | None = None


def _table() -> _Table:
    """The table of the families plugin discovery found (``registry.discover``), read once per discovery."""
    global _cached  # noqa: PLW0603 - one table per discovery result
    found = registry.discover()
    cached = _cached
    if cached is None or cached[0] is not found:
        cached = (found, _read(found["families"]))
        _cached = cached
    return cached[1]


def _read(families: dict[str, registry.PluginEntry]) -> _Table:
    models: list[BuiltinModel] = []
    for name in sorted(families):
        try:
            module = families[name].load().builtin_table
        except (
            Exception
        ) as exc:  # a family that can't load serves nothing, pinned or not
            log.warning("family %s is not loadable: %s", name, exc)
            continue
        if not module:
            continue
        try:
            pinned = tuple(importlib.import_module(module).MODELS)
        except Exception as exc:  # a table that can't load pins nothing
            log.warning("family %s's table %s is not loadable: %s", name, module, exc)
            continue
        strays = sorted({model.family for model in pinned} - {name})
        if strays:
            raise registry.PluginError(
                f"the {name} family's table {module} pins models of {', '.join(strays)}"
            )
        models.extend(pinned)
    by_repo: dict[str, BuiltinModel] = {}
    for model in models:
        key = model.repo_id.lower()
        if key in by_repo:
            raise registry.PluginConflictError(
                f"{model.repo_id} is pinned by the {by_repo[key].family} and {model.family} families"
            )
        by_repo[key] = model
    by_identity: dict[str, BuiltinModel] = {}
    for model in models:
        by_identity.setdefault(model.model_sha256, model)
    by_name: dict[str, BuiltinModel] = {}
    ambiguous: set[str] = set()
    for model in models:
        name = model.repo_id.split("/", 1)[1].lower()
        if name in by_name:
            ambiguous.add(name)
        by_name[name] = model
    for name in sorted(ambiguous):
        log.warning(
            "built-in models of several organisations are named %s; "
            "name them by repository",
            name,
        )
        del by_name[name]
    return _Table(
        models=tuple(models),
        by_repo=by_repo,
        by_name=by_name,
        by_identity=by_identity,
    )


def lookup(model: str) -> BuiltinModel | None:
    """A built-in model by repository ID or bare model name (case-insensitive)."""
    key = model.strip().lower()
    known = _table()
    return known.by_repo.get(key) or known.by_name.get(key)


def by_identity(model_sha256: str) -> BuiltinModel | None:
    return _table().by_identity.get(model_sha256)


def golden(
    model_sha256: str, surface: str, body: dict[str, Any]
) -> list[dict[str, Any]]:
    """A readiness request expecting the built-in model's recorded answers (none for another model)."""
    known = by_identity(model_sha256)
    expected = dict(known.golden_answers) if known else {}
    golden = {"surface": surface, "body": body, "expected": expected}
    if known and known.golden_tolerances:
        golden["tolerances"] = dict(known.golden_tolerances)
    return [golden]


def all_models(family: str | None = None) -> tuple[BuiltinModel, ...]:
    models = _table().models
    if family is None:
        return models
    return tuple(model for model in models if model.family == family)


def kernel_choices(
    model_sha256: str, accelerator: str, arch: str | None
) -> dict[str, Any]:
    """The recorded kernel choices of a built-in model on one device class (``accelerator:arch``), if any."""
    known = by_identity(model_sha256)
    if known is None or not arch:
        return {}
    choices: dict[str, Any] = known.kernel_choices.get(f"{accelerator}:{arch}", {})
    return choices
