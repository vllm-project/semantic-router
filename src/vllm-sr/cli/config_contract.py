"""Shared CLI config-contract inventories and helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CANONICAL_VERSION = "v0.4"

DEFAULT_BACKEND_DISPATCH = {
    "bind_address": "0.0.0.0",
    "port": 8180,
    "audience": "vllm-sr.backend-dispatch",
    "capability_ttl": "30s",
    "max_request_body_bytes": 64 << 20,
}

DEFAULT_BACKEND_EGRESS_POLICY_FILE = "/app/config/backend-egress-policy.yaml"


CONDITION_TYPE_DOMAIN = "domain"
CONDITION_TYPE_PROJECTION = "projection"

CANONICAL_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "billing_currency",
        "listeners",
        "models",
        "entrypoints",
        "recipes",
        "global",
    }
)


@dataclass(frozen=True)
class SignalFamilySpec:
    """Canonical inventory for one routing signal family."""

    canonical_key: str
    signal_attr: str
    condition_type: str
    reference_suffixes: tuple[str, ...] = ()


SIGNAL_FAMILY_SPECS = (
    SignalFamilySpec("keywords", "keywords", "keyword"),
    SignalFamilySpec("embeddings", "embeddings", "embedding"),
    SignalFamilySpec("domains", "domains", CONDITION_TYPE_DOMAIN),
    SignalFamilySpec("fact_check", "fact_check", "fact_check"),
    SignalFamilySpec(
        "user_feedbacks",
        "user_feedbacks",
        "user_feedback",
    ),
    SignalFamilySpec("reasks", "reasks", "reask"),
    SignalFamilySpec("preferences", "preferences", "preference"),
    SignalFamilySpec("language", "language", "language"),
    SignalFamilySpec("context", "context", "context"),
    SignalFamilySpec("structure", "structure", "structure"),
    SignalFamilySpec(
        "complexity",
        "complexity",
        "complexity",
        reference_suffixes=("easy", "medium", "hard"),
    ),
    SignalFamilySpec("modality", "modality", "modality"),
    SignalFamilySpec("role_bindings", "role_bindings", "authz"),
    SignalFamilySpec("jailbreak", "jailbreak", "jailbreak"),
    SignalFamilySpec("pii", "pii", "pii"),
    SignalFamilySpec("kb", "kb", "kb"),
    SignalFamilySpec("conversation", "conversation", "conversation"),
    SignalFamilySpec("events", "events", "event"),
    SignalFamilySpec("metadata", "metadata", "metadata"),
    SignalFamilySpec("classifiers", "classifiers", "classifier"),
)

_SIGNAL_FAMILY_BY_CONDITION_TYPE = {
    spec.condition_type: spec for spec in SIGNAL_FAMILY_SPECS
}


def iter_routing_profiles(config: Any) -> Iterable[tuple[str, Any]]:
    """Yield every explicit v0.4 Recipe document."""
    for recipe in config.recipes:
        yield recipe.name, recipe.document


def iter_condition_leaves(conditions: Any) -> Iterable[Any]:
    """Yield leaf conditions from a nested decision expression."""
    for condition in conditions or ():
        children = getattr(condition, "conditions", None)
        if children:
            yield from iter_condition_leaves(children)
        else:
            yield condition


def iter_named_signal_entries(signals: Any) -> Iterable[tuple[str, str]]:
    """Yield canonical signal family keys and declared signal names."""
    if not signals:
        return
    for spec in SIGNAL_FAMILY_SPECS:
        for signal in getattr(signals, spec.signal_attr, None) or []:
            if spec.condition_type == "authz":
                role = getattr(signal, "role", None)
                if role:
                    yield spec.canonical_key, role
                continue
            name = getattr(signal, "name", None)
            if name:
                yield spec.canonical_key, name


def build_signal_reference_index(signals: Any) -> dict[str, set[str]]:
    """Index valid decision references by canonical condition type."""
    names: dict[str, set[str]] = {}
    if not signals:
        return names

    for spec in SIGNAL_FAMILY_SPECS:
        family_names = names.setdefault(spec.condition_type, set())
        for signal in getattr(signals, spec.signal_attr, None) or []:
            if spec.condition_type == "authz":
                role = getattr(signal, "role", None)
                if role:
                    family_names.add(role)
                continue
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
