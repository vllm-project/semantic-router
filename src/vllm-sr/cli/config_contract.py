"""Shared CLI config-contract inventories and helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from cli.config_schema import routing_surface_catalog, schema_document

CANONICAL_VERSION = str(routing_surface_catalog()["config_version"])

CLASSIFIER_TYPE_LOCAL = "local"
CLASSIFIER_TYPE_LLM = "llm"
CLASSIFIER_TYPE_SEQUENCE = "sequence_classifier"
ClassifierSignalType = Literal[
    "local",
    "llm",
    "sequence_classifier",
]

UNKNOWN_POLICY_VALUES = ("no_match", "match", "fail_request")
UnknownPolicy = Literal["no_match", "match", "fail_request"]

CONDITION_TYPE_DOMAIN = "domain"
CONDITION_TYPE_PROJECTION = "projection"

CANONICAL_TOP_LEVEL_KEYS = frozenset(schema_document()["properties"])

LEGACY_PROVIDER_DEFAULT_KEYS = (
    "default_model",
    "reasoning_families",
    "default_reasoning_effort",
)

LEGACY_PROVIDER_KEYS = frozenset(
    {
        *LEGACY_PROVIDER_DEFAULT_KEYS,
        "model_config",
        "vllm_endpoints",
        "provider_profiles",
    }
)

LEGACY_PROVIDER_MODEL_SURFACE_KEYS = frozenset(
    {
        "endpoints",
        "access_key",
        "param_size",
        "context_window_size",
        "description",
        "capabilities",
        "loras",
        "quality_score",
        "modality",
        "tags",
    }
)


@dataclass(frozen=True)
class SignalFamilySpec:
    """Canonical inventory for one routing signal family."""

    canonical_key: str
    signal_attr: str
    condition_type: str
    display_name: str
    legacy_key: str | None = None
    reference_suffixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionFamilySpec:
    """Canonical inventory for one derived-routing collection."""

    canonical_key: str
    projection_attr: str
    display_name: str


_LEGACY_SIGNAL_KEYS = {
    "keywords": "keyword_rules",
    "embeddings": "embedding_rules",
    "domains": "categories",
    "fact_check": "fact_check_rules",
    "user_feedbacks": "user_feedback_rules",
    "reasks": "reask_rules",
    "preferences": "preference_rules",
    "language": "language_rules",
    "context": "context_rules",
    "structure": "structure_rules",
    "complexity": "complexity_rules",
    "modality": "modality_rules",
    "role_bindings": "role_bindings",
    "jailbreak": "jailbreak",
    "hallucination": "hallucination",
    "pii": "pii",
    "kb": "kb",
    "conversation": "conversation",
    "events": "events",
    "metadata": "metadata",
    "classifiers": "classifiers",
    "input_modality": "input_modality",
}

SIGNAL_FAMILY_SPECS = tuple(
    SignalFamilySpec(
        canonical_key=surface["collection"],
        signal_attr=surface["collection"],
        condition_type=surface["type"],
        display_name=surface["display_name"],
        legacy_key=_LEGACY_SIGNAL_KEYS.get(surface["collection"]),
        reference_suffixes=tuple(surface.get("reference_suffixes", ())),
    )
    for surface in routing_surface_catalog()["signals"]
)

PROJECTION_FAMILY_SPECS = tuple(
    ProjectionFamilySpec(
        canonical_key=surface["collection"],
        projection_attr=surface["collection"],
        display_name=surface["display_name"],
    )
    for surface in routing_surface_catalog()["projections"]
)

LEGACY_SIGNAL_KEY_TO_CANONICAL = {
    spec.legacy_key: spec.canonical_key
    for spec in SIGNAL_FAMILY_SPECS
    if spec.legacy_key is not None
}

LEGACY_ROUTING_KEYS = frozenset(
    {"signals", "decisions", *LEGACY_SIGNAL_KEY_TO_CANONICAL}
)

_SIGNAL_FAMILY_BY_CONDITION_TYPE = {
    spec.condition_type: spec for spec in SIGNAL_FAMILY_SPECS
}


def iter_routing_profiles(config: Any) -> Iterable[tuple[str, Any]]:
    """Yield the default and named routing profiles through one contract."""
    yield "default", config.routing
    for recipe in config.recipes:
        yield recipe.name, recipe.routing


def iter_condition_leaves(conditions: Any) -> Iterable[Any]:
    """Yield leaf conditions from a nested decision expression."""
    for condition in conditions or ():
        children = getattr(condition, "conditions", None)
        if children:
            yield from iter_condition_leaves(children)
        else:
            yield condition


def build_signal_reference_index(signals: Any) -> dict[str, set[str]]:
    """Index valid decision references by canonical condition type."""
    names: dict[str, set[str]] = {}
    if not signals:
        return names

    for spec in SIGNAL_FAMILY_SPECS:
        family_names = names.setdefault(spec.condition_type, set())
        for signal in getattr(signals, spec.signal_attr, None) or []:
            name = getattr(signal, "name", None)
            if not name:
                continue
            if spec.reference_suffixes:
                for suffix in spec.reference_suffixes:
                    family_names.add(f"{name}:{suffix}")
                continue
            family_names.add(name)

    return names


def build_projection_reference_index(projections: Any) -> set[str]:
    """Build the valid decision reference names for declared projection outputs."""
    names: set[str] = set()
    if not projections:
        return names

    for mapping in getattr(projections, "mappings", None) or []:
        for output in getattr(mapping, "outputs", None) or []:
            name = getattr(output, "name", None)
            if name:
                names.add(name)

    return names


def is_signal_condition_type(condition_type: str | None) -> bool:
    """Return whether a decision condition type references a routing signal."""
    if not condition_type:
        return False
    return condition_type.strip().lower() in _SIGNAL_FAMILY_BY_CONDITION_TYPE


def signal_reference_exists(
    signal_names: dict[str, set[str]],
    condition_type: str | None,
    raw_name: str | None,
) -> bool:
    """Return whether a decision condition references a known signal."""
    if not raw_name or not is_signal_condition_type(condition_type):
        return False

    normalized_type = condition_type.strip().lower()
    spec = _SIGNAL_FAMILY_BY_CONDITION_TYPE[normalized_type]
    family_names = signal_names.get(spec.condition_type, set())
    return raw_name in family_names
