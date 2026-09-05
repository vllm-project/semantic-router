"""Inventory of the Router Model artifacts a maintained configuration loads.

#3197 asks for a baseline over "the exact artifacts used by maintained
configurations". The evaluation registry in ``constants.py`` is hand maintained
and has drifted from ``config/config.yaml``, so this module reads the config
instead and reports what the router would actually serve.

Nothing here loads weights. It walks the config for every declared artifact
path, groups the load sites by task, and checks that each site agrees with the
``system:`` reference table. That is enough to tell an evaluation run which
artifact it must measure, and to show when two sites disagree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
HF_ORG = "llm-semantic-router"
MODEL_PREFIX = "models/"
MODEL_PATH_KEYS = ("model_id", "model_path")

# ``system:`` reference name -> evaluation task name.
REF_TASKS = {
    "prompt_guard": "jailbreak",
    "domain_classifier": "domain",
    "pii_classifier": "pii",
    "fact_check_classifier": "fact-check",
    "feedback_detector": "feedback",
}

# Load sites that declare no model_ref are classified by their config location.
LOCATION_TASKS = (
    ("modality_detector", "modality"),
    ("signals.classifiers", "jailbreak"),
)

MAPPING_KEYS = (
    "jailbreak_mapping_path",
    "category_mapping_path",
    "pii_mapping_path",
    "feedback_mapping_path",
    "label_mapping_path",
)

# Evaluation-registry keys that name the same task under a different word.
REGISTRY_ALIASES = {"domain": "intent"}

# Artifacts the router loads that no classifier task in this module covers.
UNCLASSIFIED_REFS = frozenset({"hallucination_detector", "hallucination_explainer"})


@dataclass(frozen=True)
class LoadSite:
    """One place in the config that names an artifact."""

    location: str
    model_path: str
    model_ref: str | None
    threshold: float | None
    mapping_path: str | None


@dataclass(frozen=True)
class ServedArtifact:
    """An artifact a maintained configuration loads, with every site that loads it."""

    task: str
    model_path: str
    sites: tuple[LoadSite, ...]

    @property
    def artifact_name(self) -> str:
        return self.model_path.removeprefix(MODEL_PREFIX)

    @property
    def hf_repo(self) -> str:
        return f"{HF_ORG}/{self.artifact_name}"

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(
            sorted(
                {site.threshold for site in self.sites if site.threshold is not None}
            )
        )

    @property
    def mapping_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted({site.mapping_path for site in self.sites if site.mapping_path})
        )


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config


def system_refs(config: dict[str, Any]) -> dict[str, str]:
    """Read the ``system:`` table that names one artifact per model reference."""
    for _, block in _walk_mappings(config):
        system = block.get("system")
        if not isinstance(system, dict):
            continue
        refs = {
            key: value
            for key, value in system.items()
            if isinstance(value, str) and value.startswith(MODEL_PREFIX)
        }
        if refs:
            return refs
    return {}


def load_sites(config: dict[str, Any]) -> list[LoadSite]:
    """Every config mapping that names a ``models/`` artifact path."""
    sites: list[LoadSite] = []
    for location, block in _walk_mappings(config):
        if block.get("enabled") is False:
            continue
        for key in MODEL_PATH_KEYS:
            value = block.get(key)
            if not isinstance(value, str) or not value.startswith(MODEL_PREFIX):
                continue
            sites.append(
                LoadSite(
                    location=f"{location}.{key}" if location else key,
                    model_path=value,
                    model_ref=_optional_str(block.get("model_ref")),
                    threshold=_as_float(block.get("threshold")),
                    mapping_path=_mapping_path(block),
                )
            )
    return sites


def served_artifacts(config: dict[str, Any]) -> dict[str, ServedArtifact]:
    """Group load sites into one artifact per evaluation task."""
    grouped: dict[str, list[LoadSite]] = {}
    for site in load_sites(config):
        task = _site_task(site)
        if task is None:
            continue
        grouped.setdefault(task, []).append(site)

    inventory: dict[str, ServedArtifact] = {}
    for task, sites in grouped.items():
        paths = sorted({site.model_path for site in sites})
        if len(paths) > 1:
            rendered = ", ".join(f"{site.location}={site.model_path}" for site in sites)
            raise ValueError(
                f"task {task!r} is loaded from conflicting artifacts: {rendered}"
            )
        inventory[task] = ServedArtifact(
            task=task, model_path=paths[0], sites=tuple(sites)
        )
    return inventory


def ref_mismatches(config: dict[str, Any]) -> list[str]:
    """Load sites whose ``model_id`` disagrees with the ``system:`` table."""
    refs = system_refs(config)
    findings: list[str] = []
    for site in load_sites(config):
        if site.model_ref is None:
            continue
        declared = refs.get(site.model_ref)
        if declared is None:
            findings.append(
                f"{site.location} references {site.model_ref!r}, which the system "
                "table does not define"
            )
        elif declared != site.model_path:
            findings.append(
                f"{site.location} loads {site.model_path} but the system table maps "
                f"{site.model_ref!r} to {declared}"
            )
    return findings


def registry_drift(
    inventory: dict[str, ServedArtifact], registry: dict[str, dict[str, Any]]
) -> list[str]:
    """Report evaluation-registry entries that do not name the served artifact.

    A drifted entry is not cosmetic. The registry decides which checkpoint the
    harness downloads, so a baseline built from it can describe a model the
    router never loads.
    """
    findings: list[str] = []
    for task, artifact in sorted(inventory.items()):
        registry_key = REGISTRY_ALIASES.get(task, task)
        entry = registry.get(registry_key)
        if entry is None:
            findings.append(
                f"{task}: config serves {artifact.artifact_name} but the evaluation "
                "registry has no entry for it"
            )
            continue
        registered = str(entry.get("id", ""))
        if registered != artifact.hf_repo:
            findings.append(
                f"{task}: config serves {artifact.hf_repo} but the evaluation "
                f"registry measures {registered}"
            )
    reverse = {alias: task for task, alias in REGISTRY_ALIASES.items()}
    for registry_key in sorted(registry):
        if reverse.get(registry_key, registry_key) not in inventory:
            findings.append(
                f"{registry_key}: measured by the evaluation registry but no "
                "maintained configuration loads it"
            )
    return findings


def uncovered_artifacts(config: dict[str, Any]) -> list[str]:
    """Artifacts a maintained configuration loads that no task in this module covers."""
    uncovered: dict[str, set[str]] = {}
    for site in load_sites(config):
        if _site_task(site) is not None:
            continue
        uncovered.setdefault(site.model_path, set()).add(site.location)
    return [
        f"{path} is loaded at {', '.join(sorted(locations))} but no evaluation task "
        "covers it"
        for path, locations in sorted(uncovered.items())
    ]


def _site_task(site: LoadSite) -> str | None:
    if site.model_ref is not None:
        return REF_TASKS.get(site.model_ref)
    for marker, task in LOCATION_TASKS:
        if marker in site.location:
            return task
    return None


def _walk_mappings(
    node: Any, location: str = ""
) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        yield location, node
        for key, value in node.items():
            yield from _walk_mappings(value, f"{location}.{key}".lstrip("."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_mappings(value, f"{location}[{index}]")


def _mapping_path(block: dict[str, Any]) -> str | None:
    for key in MAPPING_KEYS:
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
