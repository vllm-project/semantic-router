"""Resolve named deployment defaults from the Router-owned schema."""

from copy import deepcopy

from cli.config_contract import iter_routing_profiles
from cli.config_schema import schema_document
from cli.models import ModelBinding, UserConfig


def effective_model_deployments(config: UserConfig) -> dict[str, dict]:
    """Apply whole-entry overrides without changing the authoring document."""
    defaults = schema_document()["$defs"]["CanonicalModelCatalog"]["properties"][
        "deployments"
    ]["default"]
    catalog = (config.global_ or {}).get("model_catalog") or {}
    if "deployments" in catalog and catalog["deployments"] is None:
        return {}
    # Go merges this map by key, replacing each provided deployment value;
    # it does not merge fields inside an overridden deployment.
    return {**deepcopy(defaults), **deepcopy(catalog.get("deployments") or {})}


def global_model_bindings(config: UserConfig) -> dict[str, ModelBinding]:
    """Parse the shared catalog using the same task-binding type as recipes."""
    catalog = (config.global_ or {}).get("model_catalog") or {}
    return {
        name: ModelBinding.model_validate(value)
        for name, value in (catalog.get("bindings") or {}).items()
    }


def effective_model_bindings(config: UserConfig, profile) -> dict[str, ModelBinding]:
    """Resolve execution defaults without exporting inherited recipe policy."""
    rule_names = {
        f"classifier.{rule.name}" for rule in profile.signals.classifiers or []
    }
    for rule in profile.signals.safety:
        rule_names.add(f"safety.{rule.name}")
        if rule.hazard:
            rule_names.add(f"safety.{rule.name}.hazard")
    inherited = {
        name: binding
        for name, binding in global_model_bindings(config).items()
        if not name.startswith(("classifier.", "safety.")) or name in rule_names
    }
    return {**inherited, **profile.model_bindings}


def iter_effective_routing_profiles(config: UserConfig):
    """Yield private validation views; the authored profile remains unchanged."""
    for name, profile in iter_routing_profiles(config):
        yield (
            name,
            profile.model_copy(
                update={"model_bindings": effective_model_bindings(config, profile)}
            ),
        )
