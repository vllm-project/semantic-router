"""Shared CLI config-contract inventories and helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from cli.config_schema import routing_surface_catalog

CANONICAL_VERSION = str(routing_surface_catalog()["config_version"])

UNKNOWN_POLICY_VALUES = ("no_match", "match", "fail_request")
UnknownPolicy = Literal["no_match", "match", "fail_request"]

# Fusion panel-level quorum-failure policy. Values MUST match the Go contract in
# pkg/config/fusion_config.go (FusionQuorumFailurePolicies).
QUORUM_FAILURE_POLICY_VALUES = ("fail", "fallback")
QuorumFailurePolicy = Literal["fail", "fallback"]

CONDITION_TYPE_DOMAIN = "domain"
CONDITION_TYPE_PROJECTION = "projection"


@dataclass(frozen=True)
class SignalFamilySpec:
    """Canonical inventory for one routing signal family."""

    canonical_key: str
    signal_attr: str
    condition_type: str
    display_name: str


@dataclass(frozen=True)
class ProjectionFamilySpec:
    """Canonical inventory for one derived-routing collection."""

    canonical_key: str
    projection_attr: str
    display_name: str


SIGNAL_FAMILY_SPECS = tuple(
    SignalFamilySpec(
        canonical_key=surface["collection"],
        signal_attr=surface["collection"],
        condition_type=surface["type"],
        display_name=surface["display_name"],
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

_SIGNAL_FAMILY_BY_CONDITION_TYPE = {
    spec.condition_type: spec for spec in SIGNAL_FAMILY_SPECS
}


def iter_routing_profiles(config: Any) -> Iterable[tuple[str, Any]]:
    """Yield each recipe by name; a document without recipes routes through an
    empty recipe named default."""
    if not config.recipes:
        yield "default", config.default_recipe
    for recipe in config.recipes:
        yield recipe.name, recipe


def iter_condition_leaves(conditions: Any) -> Iterable[Any]:
    """Yield leaf conditions from a nested decision expression."""
    for condition in conditions or ():
        children = getattr(condition, "conditions", None)
        if children:
            yield from iter_condition_leaves(children)
        else:
            yield condition


# The Router emits a matched role binding's role, never the binding's name, so
# authz conditions reference the role.
_SIGNAL_REFERENCE_FIELDS = {"role_bindings": "role"}


def build_signal_reference_index(signals: Any) -> dict[str, set[str]]:
    """Index valid decision references by canonical condition type."""
    names: dict[str, set[str]] = {}
    if not signals:
        return names

    for spec in SIGNAL_FAMILY_SPECS:
        family_names = names.setdefault(spec.condition_type, set())
        field = _SIGNAL_REFERENCE_FIELDS.get(spec.signal_attr, "name")
        for signal in getattr(signals, spec.signal_attr, None) or []:
            name = getattr(signal, field, None)
            if name:
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
