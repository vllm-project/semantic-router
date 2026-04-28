"""Shared CLI config-contract inventories and helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CANONICAL_VERSION = "v0.3"

CANONICAL_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "listeners",
        "providers",
        "routing",
        "global",
        "setup",
    }
)

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
    legacy_key: str | None = None
    reference_suffixes: tuple[str, ...] = ()


SIGNAL_FAMILY_SPECS = (
    SignalFamilySpec("keywords", "keywords", "keyword", "keyword_rules"),
    SignalFamilySpec("embeddings", "embeddings", "embedding", "embedding_rules"),
    SignalFamilySpec("domains", "domains", "domain", "categories"),
    SignalFamilySpec("fact_check", "fact_check", "fact_check", "fact_check_rules"),
    SignalFamilySpec(
        "user_feedbacks",
        "user_feedbacks",
        "user_feedback",
        "user_feedback_rules",
    ),
    SignalFamilySpec("reasks", "reasks", "reask", "reask_rules"),
    SignalFamilySpec("preferences", "preferences", "preference", "preference_rules"),
    SignalFamilySpec("language", "language", "language", "language_rules"),
    SignalFamilySpec("context", "context", "context", "context_rules"),
    SignalFamilySpec("structure", "structure", "structure", "structure_rules"),
    SignalFamilySpec(
        "complexity",
        "complexity",
        "complexity",
        "complexity_rules",
        ("easy", "medium", "hard"),
    ),
    SignalFamilySpec("modality", "modality", "modality", "modality_rules"),
    SignalFamilySpec("role_bindings", "role_bindings", "authz", "role_bindings"),
    SignalFamilySpec("jailbreak", "jailbreak", "jailbreak", "jailbreak"),
    SignalFamilySpec("pii", "pii", "pii", "pii"),
    SignalFamilySpec("kb", "kb", "kb", "kb"),
    SignalFamilySpec("conversation", "conversation", "conversation", "conversation"),
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


def iter_named_signal_entries(signals: Any) -> Iterable[tuple[str, str]]:
    """Yield canonical signal family keys and declared signal names."""
    if not signals:
        return
    for spec in SIGNAL_FAMILY_SPECS:
        for signal in getattr(signals, spec.signal_attr, None) or []:
            name = getattr(signal, "name", None)
            if name:
                yield spec.canonical_key, name


def build_signal_reference_index(signals: Any) -> set[str]:
    """Build the valid decision reference names for declared signals."""
    names: set[str] = set()
    if not signals:
        return names

    for spec in SIGNAL_FAMILY_SPECS:
        for signal in getattr(signals, spec.signal_attr, None) or []:
            name = getattr(signal, "name", None)
            if not name:
                continue
            if spec.reference_suffixes:
                for suffix in spec.reference_suffixes:
                    names.add(f"{name}:{suffix}")
                continue
            names.add(name)

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
    signal_names: set[str], condition_type: str | None, raw_name: str | None
) -> bool:
    """Return whether a decision condition references a known signal."""
    if not raw_name or not is_signal_condition_type(condition_type):
        return False

    normalized_type = condition_type.strip().lower()
    spec = _SIGNAL_FAMILY_BY_CONDITION_TYPE[normalized_type]
    if spec.reference_suffixes:
        return raw_name in signal_names
    return raw_name.split(":", 1)[0] in signal_names
