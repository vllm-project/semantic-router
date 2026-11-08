"""Offline discovery, export, and explicit binding of packaged Recipes."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from cli.catalog_provider_projection import validate_provider_model_configuration
from cli.config_recipe_scopes import add_recipe
from cli.config_yaml import safe_load_router_config
from cli.decision_model import DEFAULT_DECISION_MODEL, configured_decision_model
from cli.model_bundle import MODEL_BUNDLE_FILES
from cli.model_catalog import (
    DEFAULT_CHANNEL,
    _load_asset_yaml,
    _model_assets_root,
    load_model_catalog,
)
from cli.parser import load_config_file, parse_user_config, parse_user_config_data
from cli.validator import validate_user_config


def _bundle(name: str, version: str):
    catalog = load_model_catalog(version)
    asset = catalog.assets.get(name)
    if asset is None:
        raise ValueError(f"Unknown built-in bundle '{name}'; run recipe builtin list")
    document = _load_asset_yaml(version, asset)
    return asset, document, _model_assets_root().joinpath(version, asset["bundle"])


def _requires_backend(decision: dict[str, Any]) -> bool:
    return not any(
        plugin.get("type") == "fast_response" for plugin in decision.get("plugins", [])
    )


def _decision_requirements(decision: dict[str, Any]) -> dict[str, Any]:
    requires_backend = _requires_backend(decision)
    algorithm = decision.get("algorithm", {})
    return {
        "name": decision["name"],
        "description": decision.get("description", ""),
        "algorithm": algorithm.get("type", "static") if requires_backend else None,
        "minimum_candidates": (
            algorithm.get("minimum_candidates", 1) if requires_backend else 0
        ),
    }


def list_builtin_recipes(version: str = DEFAULT_CHANNEL) -> dict[str, Any]:
    catalog = load_model_catalog(version)
    bundles = []
    for name in catalog.assets:
        asset, document, _ = _bundle(name, version)
        bundles.append(
            {
                "name": name,
                "sha256": asset["sha256"],
                "files": list(MODEL_BUNDLE_FILES),
                "recipes": [
                    {
                        "name": recipe["name"],
                        "description": recipe.get("description", ""),
                        "decisions": [
                            _decision_requirements(decision)
                            for decision in recipe.get("decisions", [])
                        ],
                    }
                    for recipe in document.get("recipes", [])
                ],
            }
        )
    return {"catalog_version": version, "bundles": bundles}


def export_builtin_bundle(
    name: str, output: Path, version: str = DEFAULT_CHANNEL
) -> dict[str, Any]:
    asset, _, resource = _bundle(name, version)
    # Export the verified five-file package verbatim, including sibling recipe
    # scopes. Selecting and binding one recipe is a separate init operation.
    contents = {
        filename: resource.joinpath(filename).read_bytes()
        for filename in MODEL_BUNDLE_FILES
    }
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    for filename, content in contents.items():
        (output / filename).write_bytes(content)
    return {
        "bundle": name,
        "catalog_version": version,
        "sha256": asset["sha256"],
        "output_dir": str(output.resolve()),
        "files": list(MODEL_BUNDLE_FILES),
    }


def _bundle_decision_model(
    candidate: dict[str, Any], bundle_document: dict[str, Any]
) -> dict[str, Any]:
    """Return the bundle's decision model for a configuration that binds none.

    A binding, or a deployment under the key that an omitted binding selects,
    is the operator's choice and stays whole: merging the two resources would
    produce a deployment neither side declared.
    """
    bundle_catalog = (bundle_document.get("global") or {}).get("model_catalog") or {}
    if "decision_model" not in (bundle_catalog.get("system") or {}):
        return {}
    key = configured_decision_model(bundle_document)
    catalog = (candidate.get("global") or {}).get("model_catalog") or {}
    deployments = catalog.get("deployments") or {}
    if (
        "decision_model" in (catalog.get("system") or {})
        or DEFAULT_DECISION_MODEL in deployments
        or key in deployments
    ):
        return {}
    selected: dict[str, Any] = {}
    deployment = (bundle_catalog.get("deployments") or {}).get(key)
    if deployment is not None:
        selected["deployments"] = {key: deployment}
    selected["system"] = {"decision_model": bundle_catalog["system"]["decision_model"]}
    return selected


def _copy_missing_plugins(candidate: dict[str, Any], plugins: Any) -> None:
    """The bundle's default plugins, of the types candidate's global.plugins
    does not declare."""
    if not isinstance(plugins, list) or not plugins:
        return
    target = candidate.setdefault("global", {}).setdefault("plugins", [])
    declared = {plugin.get("type") for plugin in target if isinstance(plugin, dict)}
    for plugin in plugins:
        if isinstance(plugin, dict) and plugin.get("type") not in declared:
            target.append(copy.deepcopy(plugin))


def apply_builtin_runtime_defaults(
    candidate: dict[str, Any], bundle_document: dict[str, Any]
) -> None:
    """Fill omitted learning, replay, default-plugin and decision-model settings;
    authored ones win.

    These are configuration-wide defaults, materialized during initialization.
    Runtime parsing does not infer policy from built-in recipe names.
    """
    bundle_global = bundle_document.get("global", {})
    defaults: dict[str, Any] = {}
    for section, setting in (("router", "learning"), ("services", "router_replay")):
        value = bundle_global.get(section, {}).get(setting)
        if value is not None:
            defaults.setdefault(section, {})[setting] = value
    decision_model = _bundle_decision_model(candidate, bundle_document)
    if decision_model:
        defaults["model_catalog"] = decision_model
    _copy_missing_plugins(candidate, bundle_global.get("plugins"))
    if not defaults:
        return

    def fill_missing(target: dict[str, Any], defaults: dict[str, Any]) -> None:
        for key, value in defaults.items():
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                fill_missing(target[key], value)

    fill_missing(candidate, {"global": defaults})


def load_provider_config(config_path: Path) -> dict[str, Any]:
    """Read a canonical provider config that a built-in recipe will join."""
    parse_user_config(str(config_path), log_summary=False)
    return load_config_file(str(config_path))


def compose_builtin_recipe(
    name: str,
    *,
    bundle: str,
    config: dict[str, Any],
    config_path: Path,
    bindings: Any,
    model_names: Sequence[str],
    version: str = DEFAULT_CHANNEL,
    excluded_decisions: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a validated copy of config that publishes one installed recipe.

    config is the provider config read from config_path. Every selected
    decision is bound to operator-supplied provider model refs, and the new
    entrypoint serves model_names, the first as its primary name. The original
    package, provider document, and authored model metadata are unchanged.
    Algorithm contracts and minimum pools remain authoritative.
    """
    _, document, _ = _bundle(bundle, version)
    recipes = [
        recipe for recipe in document.get("recipes", []) if recipe["name"] == name
    ]
    if not recipes:
        raise ValueError(
            f"Bundle '{bundle}' has no recipe '{name}'; run recipe builtin list"
        )
    selected = copy.deepcopy(recipes[0])
    candidate = copy.deepcopy(config)
    models = {
        model["name"] for model in candidate.get("providers", {}).get("models", [])
    }
    if not isinstance(bindings, dict):
        raise ValueError("Bindings must map each decision name to a list of modelRefs")
    decisions = selected["decisions"]
    excluded = set(excluded_decisions)
    unknown_exclusions = excluded - {decision["name"] for decision in decisions}
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded decisions: {sorted(unknown_exclusions)}")
    decisions = [decision for decision in decisions if decision["name"] not in excluded]
    if not decisions:
        raise ValueError("At least one recipe decision must remain")
    selected["decisions"] = decisions
    declared = {decision["name"] for decision in decisions}
    expected = {
        decision["name"] for decision in decisions if _requires_backend(decision)
    }
    if expected - set(bindings) or set(bindings) - declared:
        raise ValueError(
            f"Bindings must cover every decision that calls a backend; missing={sorted(expected - set(bindings))}, unknown={sorted(set(bindings) - declared, key=str)}"
        )
    for decision in decisions:
        if not _requires_backend(decision):
            if bindings.get(decision["name"], []) != []:
                raise ValueError(
                    f"Decision '{decision['name']}' responds immediately and needs no modelRefs"
                )
            continue
        refs = bindings[decision["name"]]
        if (
            not isinstance(refs, list)
            or not refs
            or any(
                not isinstance(ref, dict)
                or not isinstance(ref.get("model"), str)
                or not ref["model"].strip()
                or (
                    ref.get("lora_name") is not None
                    and not isinstance(ref["lora_name"], str)
                )
                for ref in refs
            )
        ):
            raise ValueError(
                f"Bindings for '{decision['name']}' must be a non-empty list of modelRefs"
            )
        unknown = {ref["model"] for ref in refs} - models
        if unknown:
            raise ValueError(
                f"Decision '{decision['name']}' references unconfigured provider models: {sorted(unknown)}"
            )
        minimum = decision.get("algorithm", {}).get("minimum_candidates", 1)
        candidates = {ref.get("lora_name") or ref["model"] for ref in refs}
        if len(candidates) < minimum:
            raise ValueError(
                f"Decision '{decision['name']}' requires at least {minimum} distinct model candidates; got {len(candidates)}"
            )
        decision["modelRefs"] = copy.deepcopy(refs)

    if any(recipe["name"] == name for recipe in candidate.get("recipes", [])):
        raise ValueError(f"Config already contains recipe '{name}'")
    add_recipe(candidate, selected)
    candidate.setdefault("entrypoints", []).append(
        {"model_names": list(model_names), "recipe": name}
    )
    apply_builtin_runtime_defaults(candidate, document)
    # Setup envelopes disable the Router; a configured candidate is an explicit
    # deployment, so callers must start from a provider config, not setup mode.
    if (candidate.get("setup") or {}).get("mode"):
        raise ValueError("Start from an explicit provider config, not setup mode")
    _validate_composition(candidate, config_path)
    return candidate


def _render_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _validate_composition(candidate: dict[str, Any], config_path: Path) -> None:
    # Validate the rendered document, typed exactly as the file will load.
    parsed = parse_user_config_data(
        safe_load_router_config(_render_config(candidate)),
        str(config_path),
        log_summary=False,
    )
    errors = validate_user_config(parsed, log_summary=False)
    if errors:
        raise ValueError(
            "Configuration validation failed: "
            + "; ".join(str(error) for error in errors)
        )
    validate_provider_model_configuration(
        parsed,
        allow_backendless_physical=(
            "listeners" in parsed.model_fields_set and not parsed.listeners
        ),
    )


def initialize_builtin_recipe(
    name: str,
    *,
    bundle: str,
    config_path: Path,
    bindings_path: Path,
    model_name: str,
    output: Path,
    version: str = DEFAULT_CHANNEL,
    excluded_decisions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compose one installed recipe into a provider config as a new file.

    See compose_builtin_recipe for the binding contract. The source config is
    unchanged and an existing output file is never replaced.
    """
    if output.exists():
        raise ValueError(f"Output file already exists: {output}")
    config = load_provider_config(config_path)
    candidate = compose_builtin_recipe(
        name,
        bundle=bundle,
        config=config,
        config_path=config_path,
        bindings=yaml.safe_load(bindings_path.read_text(encoding="utf-8")),
        model_names=[model_name],
        version=version,
        excluded_decisions=excluded_decisions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(_render_config(candidate))
    return {
        "bundle": bundle,
        "recipe": name,
        "model_name": model_name,
        "catalog_version": version,
        "output": str(output.resolve()),
        "valid": True,
        "decisions": len(candidate["recipes"][-1]["decisions"]),
        "excluded_decisions": sorted(set(excluded_decisions)),
    }
