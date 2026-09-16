"""Resolve named deployment defaults from the Router-owned schema."""

from copy import deepcopy

from cli.config_schema import schema_document
from cli.models import UserConfig


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
