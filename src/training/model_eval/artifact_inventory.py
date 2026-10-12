"""Inventory of the Router Model artifacts a maintained configuration loads.

#3197 asks for a baseline over "the exact artifacts used by maintained
configurations". The evaluation registry in ``constants.py`` is hand maintained
and has drifted from ``config/config.yaml``, so this module reads the config
instead and reports what the router would actually serve.

Nothing here loads weights. It walks the config for every declared artifact
path, groups the load sites by task, and checks that each site agrees with the
``system:`` reference table. That is enough to tell an evaluation run which
artifact it must measure, and to show when two sites disagree.

Since #4721 a module load site usually names no artifact path at all: it
declares only a ``model_ref``, and the router resolves the reference in two
steps this module mirrors (``resolveSystemModelRef`` and
``applyDecisionModel`` in the Router's config package). An explicit
``model_id``/``model_path`` on the module wins; otherwise the ``system:``
table entry for the reference decides, and every system line the
configuration leaves unset is filled from the table of the decision model
``system.decision_model`` names (the default is Vela-2.0-0.3B). The
modality classifier names no model either: an empty ``model_path`` runs the
decision model's modality model. A module that sets no threshold likewise
runs at the threshold of the model it runs, so a resolved site reports the
model's published module thresholds from
``src/model-runtime/docs/records/vela2-decision-model-sizes.json`` — the
same published record the e2e guard check reads — or, for any other model,
the Vela 1.0 specialists' thresholds the Router falls back to.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
DECISION_MODEL_RECORD = (
    REPO_ROOT
    / "src"
    / "model-runtime"
    / "docs"
    / "records"
    / "vela2-decision-model-sizes.json"
)
HF_ORG = "vllm-sr"
MODEL_PREFIX = "models/"
SCHEMA_PATH = (
    REPO_ROOT
    / "src"
    / "semantic-router"
    / "pkg"
    / "configschema"
    / "router-config-v0.3.schema.json"
)
DEFAULT_DECISION_DEPLOYMENT = "primary"
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
LOCATION_TASKS = (("modality_detector", "modality"),)

MAPPING_KEYS = (
    "jailbreak_mapping_path",
    "category_mapping_path",
    "pii_mapping_path",
    "feedback_mapping_path",
    "label_mapping_path",
)

# Evaluation-registry keys that name the same task under a different word.
REGISTRY_ALIASES = {"domain": "intent"}

# The ``system:`` table's model lines, in the Router's field order
# (CanonicalSystemModels). ``decision_model`` is the remaining key.
SYSTEM_KEYS = (
    "safety",
    "hazard",
    "prompt_guard",
    "domain_classifier",
    "pii_classifier",
    "fact_check_classifier",
    "hallucination_detector",
    "feedback_detector",
)

VELA1_HAZARD_MODEL = "models/Vela-1.0-Encoder-307M-Hazard"

# The Vela 1.0 specialists, mirroring Vela1SystemModels in the Router's
# canonical defaults: the system table of the ``Vela-1.0`` decision model.
# Module thresholds of a model that is not a Vela 2.0 size, mirroring
# vela1ModuleThresholds in the Router's canonical operating points. Vela 2.0
# thresholds are not copied here; they are read from the published record.
VELA1_MODULE_THRESHOLDS = {
    "jailbreak": 0.5,
    "domain": 0.5,
    "pii": 0.9,
    "fact_check": 0.95,
    "feedback": 0.7,
}

# ``system:`` reference name -> the record's module-threshold key. Only these
# five modules take a model threshold when they set none.
THRESHOLD_KEYS = {
    "prompt_guard": "jailbreak",
    "domain_classifier": "domain",
    "pii_classifier": "pii",
    "fact_check_classifier": "fact_check",
    "feedback_detector": "feedback",
}


@dataclass(frozen=True)
class DecisionModel:
    """What a decision model binds, mirroring DecisionModelSpec in Go."""

    name: str
    # The one model that answers every built-in signal: the deployment's
    # served artifact, translated to its registered local path when the
    # built-in registry knows it.
    model: str
    # The modality classifier's model when it names none.
    modality: str
    # The system table this decision model fills unset lines from.
    system: dict[str, str]


def _vela2_decision_model(name: str, model: str) -> DecisionModel:
    # A Vela 2.0 size answers every built-in signal itself; Hazard has no
    # trained question and stays on the Vela 1.0 Hazard model.
    system = dict.fromkeys(SYSTEM_KEYS, model)
    system["hazard"] = VELA1_HAZARD_MODEL
    return DecisionModel(name=name, model=model, modality=model, system=system)


# Decision models in size order, mirroring decisionModels in decision_model.go.
DECISION_MODELS = (
    _vela2_decision_model("Vela-2.0-0.3B", "models/Vela-2.0-0.3B"),
    _vela2_decision_model("Vela-2.0-0.8B", "models/Vela-2.0-0.8B"),
    _vela2_decision_model("Vela-2.0-4B", "models/Vela-2.0-4B"),
    _vela2_decision_model("Vela-2.0-9B", "models/Vela-2.0-9B"),
)

# The built-in registry's repo-id to local-path translation, mirroring
# DefaultModelRegistry in the Router's config package: decisionModelSpec
# rewrites a served artifact to its registered local path and leaves an
# unregistered artifact as declared.
_REGISTRY_LOCAL_PATHS = {
    "KRLabsOrg/lettucedect-base-modernbert-en-v1": "models/mom-halugate-detector",
    "KRLabsOrg/lettucedect-v2-mmbert-base": "models/lettucedect-v2-mmbert-base",
    "Qwen/Qwen3-Embedding-0.6B": "models/mom-embedding-pro",
    "google/embeddinggemma-300m": "models/mom-embedding-flash",
    "sentence-transformers/all-MiniLM-L12-v2": "models/mom-embedding-light",
    "vllm-sr/Vela-1.0-Encoder-307M": "models/Vela-1.0-Encoder-307M",
    "vllm-sr/Vela-1.0-Encoder-307M-Domain": "models/Vela-1.0-Encoder-307M-Domain",
    "vllm-sr/Vela-1.0-Encoder-307M-Embedding": "models/Vela-1.0-Encoder-307M-Embedding",
    "vllm-sr/Vela-1.0-Encoder-307M-FactCheck": "models/Vela-1.0-Encoder-307M-FactCheck",
    "vllm-sr/Vela-1.0-Encoder-307M-Feedback": "models/Vela-1.0-Encoder-307M-Feedback",
    "vllm-sr/Vela-1.0-Encoder-307M-Guard": "models/Vela-1.0-Encoder-307M-Guard",
    "vllm-sr/Vela-1.0-Encoder-307M-Halu": "models/Vela-1.0-Encoder-307M-Halu",
    "vllm-sr/Vela-1.0-Encoder-307M-Hazard": "models/Vela-1.0-Encoder-307M-Hazard",
    "vllm-sr/Vela-1.0-Encoder-307M-Modality": "models/Vela-1.0-Encoder-307M-Modality",
    "vllm-sr/Vela-1.0-Encoder-307M-PII": "models/Vela-1.0-Encoder-307M-PII",
    "vllm-sr/Vela-1.0-Encoder-307M-Reranker": "models/Vela-1.0-Encoder-307M-Reranker",
    "vllm-sr/Vela-1.0-Encoder-307M-Safety": "models/Vela-1.0-Encoder-307M-Safety",
    "vllm-sr/Vela-1.0-Encoder-307M-Shield": "models/Vela-1.0-Encoder-307M-Shield",
    "vllm-sr/Vela-1.0-Omni-Mini": "models/vela-1.0-omni-mini",
    "vllm-sr/Vela-1.0-Omni-Nano": "models/vela-1.0-omni-nano",
    "vllm-sr/feedback-detector": "models/mom-feedback-detector",
    "vllm-sr/halugate-sentinel": "models/mom-halugate-sentinel",
    "vllm-sr/jailbreak_classifier_modernbert-base_model": "models/mom-jailbreak-classifier",
    "vllm-sr/lora_intent_classifier_bert-base-uncased_model": "models/mom-domain-classifier",
    "vllm-sr/lora_pii_detector_bert-base-uncased_model": "models/mom-pii-classifier",
    "vllm-sr/mmbert-embed-32k-2d-matryoshka": "models/mmbert-embed-32k-2d-matryoshka",
    "vllm-sr/mmbert-pii-detector-merged": "models/mom-mmbert-pii-detector",
    "vllm-sr/mmbert32k-factcheck-classifier-lora": "models/mmbert32k-factcheck-classifier-lora",
    "vllm-sr/mmbert32k-factcheck-classifier-merged": "models/mmbert32k-factcheck-classifier-merged",
    "vllm-sr/mmbert32k-feedback-detector-lora": "models/mmbert32k-feedback-detector-lora",
    "vllm-sr/mmbert32k-feedback-detector-merged": "models/mmbert32k-feedback-detector-merged",
    "vllm-sr/mmbert32k-intent-classifier-lora": "models/mmbert32k-intent-classifier-lora",
    "vllm-sr/mmbert32k-intent-classifier-merged": "models/mmbert32k-intent-classifier-merged",
    "vllm-sr/mmbert32k-jailbreak-detector-lora": "models/mmbert32k-jailbreak-detector-lora",
    "vllm-sr/mmbert32k-jailbreak-detector-merged": "models/mmbert32k-jailbreak-detector-merged",
    "vllm-sr/mmbert32k-modality-router-merged": "models/mmbert32k-modality-router-merged",
    "vllm-sr/mmbert32k-pii-detector-lora": "models/mmbert32k-pii-detector-lora",
    "vllm-sr/mmbert32k-pii-detector-merged": "models/mmbert32k-pii-detector-merged",
    "vllm-sr/multi-modal-embed-small": "models/mom-embedding-multimodal",
}

# The built-in registry's repo-id form of each decision model, the shape
# ServedModel returns for a managed deployment.
DECISION_MODELS_BY_REPO = {
    f"{HF_ORG}/{spec.model.removeprefix(MODEL_PREFIX)}": spec
    for spec in DECISION_MODELS
}


@dataclass(frozen=True)
class LoadSite:
    """One place in the config that names an artifact."""

    location: str
    model_path: str
    model_ref: str | None
    threshold: float | None
    mapping_path: str | None
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
        """The Hub repository: a registered local path's org-qualified name,
        or the repo-id an unregistered artifact's deployment declared."""
        if self.model_path.startswith(MODEL_PREFIX):
            return f"{HF_ORG}/{self.artifact_name}"
        return self.model_path

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


def _system_block(config: dict[str, Any]) -> dict[str, Any]:
    """The raw ``system:`` table of ``global.model_catalog``, or an empty one."""
    for _, block in _walk_mappings(config):
        system = block.get("system")
        if not isinstance(system, dict):
            continue
        if "decision_model" in system or any(key in system for key in SYSTEM_KEYS):
            return system
    return {}


@lru_cache(maxsize=1)
def _builtin_deployments() -> dict[str, dict[str, Any]]:
    """The schema's built-in deployments: the Router's own defaults."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return deepcopy(
        schema["$defs"]["CanonicalModelCatalog"]["properties"]["deployments"]["default"]
    )


def _effective_deployments(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The deployments a decision binding resolves against: the schema's
    built-ins merged by key with the config's declarations, the same
    whole-entry override cli.model_runtime_defaults and the Router's canonical
    load perform."""
    catalog = ((config or {}).get("global") or {}).get("model_catalog") or {}
    if "deployments" in catalog and catalog["deployments"] is None:
        return {}
    return {
        **deepcopy(_builtin_deployments()),
        **deepcopy(catalog.get("deployments") or {}),
    }


def _served_model(key: str, deployment: dict[str, Any]) -> str:
    """deployment.ServedModel: the artifact, or, for an attached runtime without
    one, the model the runtime serves under served_name or the deployment's
    name."""
    artifact = deployment.get("artifact") or ""
    if artifact or not deployment.get("endpoint"):
        return artifact
    return deployment.get("served_name") or key


def _decision_model(
    system: dict[str, Any], deployments: dict[str, dict[str, Any]]
) -> DecisionModel:
    """Resolve ``system.decision_model`` the way the Router's
    ``applyDecisionModel`` does: the binding names one declared model_runtime
    deployment, whose served artifact fills every unset system line while
    Hazard stays on the Vela 1.0 Hazard model. A binding that is not the
    canonical ``{deployment: <key>}`` mapping, an unknown deployment, or a
    non-model_runtime deployment is the Router's load error, never a silent
    fallback."""
    binding = system.get("decision_model")
    if binding is None:
        key = DEFAULT_DECISION_DEPLOYMENT
    elif isinstance(binding, dict) and set(binding) == {"deployment"}:
        key = binding["deployment"]
        if not isinstance(key, str) or not key or key.strip() != key:
            raise ValueError(
                "decision_model.deployment must be a non-empty, trimmed deployment key"
            )
    else:
        raise ValueError(
            "decision_model must be {deployment: <declared deployment key>}"
        )
    deployment = deployments.get(key)
    if deployment is None:
        raise ValueError(f"decision_model.deployment: unknown deployment {key!r}")
    if deployment.get("provider") != "model_runtime":
        raise ValueError(
            f"decision_model.deployment {key!r} must use provider model_runtime"
        )
    model = _served_model(key, deployment)
    model = _REGISTRY_LOCAL_PATHS.get(model, model)
    spec = DECISION_MODELS_BY_REPO.get(model)
    if spec is not None:
        return spec
    system_table = dict.fromkeys(SYSTEM_KEYS, model)
    system_table["hazard"] = VELA1_HAZARD_MODEL
    return DecisionModel(name=key, model=model, modality=model, system=system_table)


def system_refs(config: dict[str, Any]) -> dict[str, str]:
    """The resolved ``system:`` table: one artifact per model reference.

    Lines the configuration sets win; every line it leaves unset takes the
    decision model's binding, exactly as the Router's ``applyDecisionModel``
    fills them at load.
    """
    system = _system_block(config)
    refs = dict(_decision_model(system, _effective_deployments(config)).system)
    for key, value in system.items():
        if key in refs and isinstance(value, str) and value.startswith(MODEL_PREFIX):
            refs[key] = value
    return refs


def load_sites(config: dict[str, Any]) -> list[LoadSite]:
    """Every config mapping that loads an artifact, after reference resolution.

    A mapping loads an artifact when it names a ``models/`` path in
    ``model_id``/``model_path``, or when it names only a ``model_ref`` the
    resolved system table binds to a path. The modality classifier's empty
    ``model_path`` loads the decision model's modality model.
    """
    system = _system_block(config)
    decision = _decision_model(system, _effective_deployments(config))
    refs = system_refs(config)
    sites: list[LoadSite] = []
    for location, block in _walk_mappings(config):
        if block.get("enabled") is False:
            continue
        model_ref = _optional_str(block.get("model_ref"))
        explicit = False
        for key in MODEL_PATH_KEYS:
            value = block.get(key)
            if not isinstance(value, str) or not value.startswith(MODEL_PREFIX):
                continue
            explicit = True
            sites.append(
                LoadSite(
                    location=f"{location}.{key}" if location else key,
                    model_path=value,
                    model_ref=model_ref,
                    threshold=_effective_threshold(block, value, model_ref),
                    mapping_path=_mapping_path(block),
                    positive_labels=_positive_labels(block),
                )
            )
        if explicit:
            continue
        if model_ref is not None:
            resolved = refs.get(model_ref)
            if resolved is None:
                # ref_mismatches reports the undefined reference.
                continue
            sites.append(
                LoadSite(
                    location=f"{location}.model_ref" if location else "model_ref",
                    model_path=resolved,
                    model_ref=model_ref,
                    threshold=_effective_threshold(block, resolved, model_ref),
                    mapping_path=_mapping_path(block),
                    positive_labels=_positive_labels(block),
                )
            )
        elif "modality_detector" in location and block.get("model_path") == "":
            sites.append(
                LoadSite(
                    location=f"{location}.model_path" if location else "model_path",
                    model_path=decision.modality,
                    model_ref=None,
                    threshold=_as_float(block.get("threshold")),
                    mapping_path=_mapping_path(block),
                    positive_labels=_positive_labels(block),
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
    for location, block in _walk_mappings(config):
        if block.get("enabled") is False:
            continue
        model_ref = _optional_str(block.get("model_ref"))
        if model_ref is None:
            continue
        declared = refs.get(model_ref)
        if declared is None:
            findings.append(
                f"{location}.model_ref references {model_ref!r}, which the system "
                "table does not define"
            )
            continue
        for key in MODEL_PATH_KEYS:
            value = block.get(key)
            if (
                isinstance(value, str)
                and value.startswith(MODEL_PREFIX)
                and value != declared
            ):
                site_location = f"{location}.{key}" if location else key
                findings.append(
                    f"{site_location} loads {value} but the system table maps "
                    f"{model_ref!r} to {declared}"
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


def _effective_threshold(
    block: dict[str, Any], model_path: str, model_ref: str | None
) -> float | None:
    """The threshold a load site runs at: its own, or its model's.

    A module that sets no threshold takes the one calibrated for the model it
    runs (the Router's ``normalizeModuleOperatingPoints``), so a resolved
    site reports the model's threshold rather than none.
    """
    declared = _as_float(block.get("threshold"))
    if declared is not None:
        return declared
    key = THRESHOLD_KEYS.get(model_ref or "")
    if key is None:
        return None
    return _module_thresholds(model_path).get(key)


def _module_thresholds(model_path: str) -> dict[str, float]:
    """The published module thresholds of one model."""
    name = model_path.removeprefix(MODEL_PREFIX)
    if name.startswith("Vela-2.0-"):
        size = name.removeprefix("Vela-2.0-")
        thresholds = _published_module_thresholds().get(size)
        if thresholds is None:
            raise ValueError(
                f"the published decision-model record has no module thresholds "
                f"for {name}"
            )
        return thresholds
    return VELA1_MODULE_THRESHOLDS


@lru_cache(maxsize=1)
def _published_module_thresholds() -> dict[str, dict[str, float]]:
    record = json.loads(DECISION_MODEL_RECORD.read_text(encoding="utf-8"))
    return {
        size: {task: float(value) for task, value in tasks.items()}
        for size, tasks in record["module_thresholds"].items()
    }


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


def _positive_labels(block: dict[str, Any]) -> tuple[str, ...]:
    declared = block.get("positive_labels")
    if not isinstance(declared, list):
        return ()
    return tuple(str(label) for label in declared if isinstance(label, str) and label)


def _optional_str(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
