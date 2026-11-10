"""Inventory of the Router Model artifacts a maintained configuration loads.

#3197 asks for a baseline over "the exact artifacts used by maintained
configurations". The evaluation registry in ``constants.py`` is hand maintained
and has drifted from ``config/config.yaml``, so this module reads the config
instead and reports what the router would actually serve.

Nothing here loads weights. Every rule of a model-answered signal that a
recipe declares (``recipes[].signals.<collection>``) asks one deployment: the
one it names, else the one its recipe's ``model_bindings.<type>`` names, else
the one ``global.model_catalog.bindings.<type>`` names, else the decision
model's (``global.model_catalog.system.decision_model``, the pre-declared
``primary`` unless it names another). This module reports the artifact that
deployment serves, grouped by evaluation task; a remote deployment serves
none. A rule runs at the bound of its predicate, or, when it sets none, at the
threshold calibrated for its model, which the Router reads and this module
does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
HF_ORG = "vllm-sr"
MODEL_PREFIX = "models/"

# The model_runtime deployments the Router always declares, mirroring
# DefaultModelDeployments; a deployment the configuration declares under the
# same name replaces one.
PREDECLARED_ARTIFACTS = {
    "primary": "vllm-sr/vela3-fast",
    "vela-embedding": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
}
DEFAULT_DECISION_DEPLOYMENT = "primary"

# A recipe's signal collections whose rules ask a model: the signal type its
# bindings name (a systemone rule takes none), and the evaluation task that
# measures it, if any.
MODEL_SIGNALS = {
    "domains": ("domain", "domain"),
    "fact_check": ("fact_check", "fact-check"),
    "user_feedbacks": ("user_feedback", "feedback"),
    "jailbreak": ("jailbreak", "jailbreak"),
    "safety": ("safety", None),
    "hallucination": ("hallucination", None),
    "refusal": ("refusal", None),
    "turn_relation": ("turn_relation", None),
    "pii": ("pii", "pii"),
    "complexity": ("complexity", None),
    "modality": ("modality", "modality"),
    "systemone": ("systemone", None),
}

# A task whose rules are binary gates rather than argmax classifiers, and the
# option a gate matches: the guard's positive one.
GATE_LABELS = {"jailbreak": ("jailbreak",)}

# Evaluation-registry keys that name the same task under a different word.
REGISTRY_ALIASES = {"domain": "intent"}


@dataclass(frozen=True)
class LoadSite:
    """One rule in the config that asks an artifact."""

    location: str
    model_path: str
    threshold: float | None
    positive_labels: tuple[str, ...]


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
    def positive_labels(self) -> tuple[str, ...]:
        """Labels whose probability mass the router compares to the threshold.

        A site that declares them is a binary gate rather than an argmax
        classifier, so its recall and false-positive rate are what the report
        has to state.
        """
        return tuple(
            dict.fromkeys(
                label for site in self.sites for label in site.positive_labels
            )
        )


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config


def load_sites(config: dict[str, Any]) -> list[tuple[str | None, LoadSite]]:
    """Every rule that asks an artifact, with the evaluation task that measures it."""
    catalog = _mapping(_mapping(config.get("global")).get("model_catalog"))
    decision = _deployment_name(_mapping(catalog.get("system")).get("decision_model"))
    sites: list[tuple[str | None, LoadSite]] = []
    for index, raw_recipe in enumerate(_sequence(config.get("recipes"))):
        recipe = _mapping(raw_recipe)
        name = _optional_str(recipe.get("name")) or str(index)
        signals = _mapping(recipe.get("signals"))
        bindings = (
            _mapping(recipe.get("model_bindings")),
            _mapping(catalog.get("bindings")),
        )
        for collection, (signal_type, task) in MODEL_SIGNALS.items():
            for raw_rule in _sequence(signals.get(collection)):
                rule = _mapping(raw_rule)
                deployment = _optional_str(rule.get("deployment"))
                if deployment is None and signal_type != "systemone":
                    for scope in bindings:
                        if deployment := _deployment_name(scope.get(signal_type)):
                            break
                model_path = _deployment_artifact(
                    catalog, deployment or decision or DEFAULT_DECISION_DEPLOYMENT
                )
                if model_path is None:
                    continue
                site = LoadSite(
                    location=f"recipes[{name}].signals.{collection}"
                    f"[{_optional_str(rule.get('name')) or ''}]",
                    model_path=model_path,
                    threshold=_predicate_bound(rule.get("predicate")),
                    positive_labels=GATE_LABELS.get(task or "", ()),
                )
                sites.append((task, site))
    return sites


def served_artifacts(config: dict[str, Any]) -> dict[str, ServedArtifact]:
    """Group load sites into one artifact per evaluation task."""
    grouped: dict[str, list[LoadSite]] = {}
    for task, site in load_sites(config):
        if task is not None:
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
    for task, site in load_sites(config):
        if task is None:
            uncovered.setdefault(site.model_path, set()).add(site.location)
    return [
        f"{path} is loaded at {', '.join(sorted(locations))} but no evaluation task "
        "covers it"
        for path, locations in sorted(uncovered.items())
    ]


def _deployment_artifact(catalog: dict[str, Any], name: str) -> str | None:
    """The model a deployment serves, or None for one the Router runs nowhere."""
    deployment = _mapping(catalog.get("deployments")).get(name)
    if deployment is None:
        artifact = PREDECLARED_ARTIFACTS.get(name)
    elif deployment.get("provider") != "model_runtime" or deployment.get("endpoint"):
        return None
    else:
        artifact = deployment.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        return None
    artifact = artifact.strip()
    if artifact.startswith(f"{HF_ORG}/"):
        return MODEL_PREFIX + artifact.removeprefix(f"{HF_ORG}/")
    return artifact


def _deployment_name(value: Any) -> str | None:
    """A binding written as a deployment name or as ``{deployment: name}``."""
    if isinstance(value, dict):
        value = value.get("deployment")
    return _optional_str(value)


def _predicate_bound(predicate: Any) -> float | None:
    """The lower bound a rule's predicate holds its probability to."""
    predicate = _mapping(predicate)
    for key in ("gte", "gt"):
        if (bound := _as_float(predicate.get(key))) is not None:
            return bound
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
