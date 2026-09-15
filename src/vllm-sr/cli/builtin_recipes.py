"""Offline discovery, export, and explicit binding of packaged Recipes."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cli.catalog_provider_projection import validate_provider_model_configuration
from cli.commands.runtime_kb import validate_kb_source_output_location
from cli.model_bundle import MODEL_BUNDLE_FILES
from cli.model_catalog import (
    DEFAULT_CHANNEL,
    _load_asset_yaml,
    _model_assets_root,
    load_model_catalog,
)
from cli.parser import load_config_file, parse_user_config
from cli.validator import validate_user_config


def _bundle(name: str, version: str):
    catalog = load_model_catalog(version)
    asset = catalog.assets.get(name)
    if asset is None:
        raise ValueError(f"Unknown built-in bundle '{name}'; run recipe builtin list")
    document = _load_asset_yaml(version, asset)
    return asset, document, _model_assets_root().joinpath(version, asset["bundle"])


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
                            {
                                "name": decision["name"],
                                "algorithm": decision.get("algorithm", {}).get(
                                    "type", "static"
                                ),
                                "minimum_candidates": decision.get("algorithm", {}).get(
                                    "minimum_candidates", 1
                                ),
                            }
                            for decision in recipe.get("routing", {}).get(
                                "decisions", []
                            )
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
    """Bind every selected decision to operator-supplied provider model refs.

    The original package, provider document, and authored model metadata are
    unchanged. Algorithm contracts and minimum pools remain authoritative.
    """
    if output.exists():
        raise ValueError(f"Output file already exists: {output}")
    parse_user_config(str(config_path), log_summary=False)
    _, document, _ = _bundle(bundle, version)
    recipes = [
        recipe for recipe in document.get("recipes", []) if recipe["name"] == name
    ]
    if not recipes:
        raise ValueError(
            f"Bundle '{bundle}' has no recipe '{name}'; run recipe builtin list"
        )
    selected = copy.deepcopy(recipes[0])
    candidate = load_config_file(str(config_path))
    validate_kb_source_output_location(candidate, config_path, output)
    models = {
        model["name"] for model in candidate.get("providers", {}).get("models", [])
    }
    bindings = yaml.safe_load(bindings_path.read_text(encoding="utf-8"))
    if not isinstance(bindings, dict):
        raise ValueError("Bindings must map each decision name to a list of modelRefs")
    decisions = selected["routing"]["decisions"]
    excluded = set(excluded_decisions)
    unknown_exclusions = excluded - {decision["name"] for decision in decisions}
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded decisions: {sorted(unknown_exclusions)}")
    decisions = [decision for decision in decisions if decision["name"] not in excluded]
    if not decisions:
        raise ValueError("At least one recipe decision must remain")
    selected["routing"]["decisions"] = decisions
    expected = {decision["name"] for decision in decisions}
    if set(bindings) != expected:
        raise ValueError(
            f"Bindings must cover every decision; missing={sorted(expected - set(bindings))}, unknown={sorted(set(bindings) - expected, key=str)}"
        )
    for decision in decisions:
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
    candidate.setdefault("recipes", []).append(selected)
    candidate.setdefault("entrypoints", []).append(
        {"model_names": [model_name], "recipe": name}
    )
    # Setup envelopes disable the Router; a configured candidate is an explicit
    # deployment, so callers must start from a provider config, not setup mode.
    if (candidate.get("setup") or {}).get("mode"):
        raise ValueError("Start from an explicit provider config, not setup mode")
    content = yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True)
    with tempfile.TemporaryDirectory(prefix="vllm-sr-builtin-") as directory:
        temporary = Path(directory) / "config.yaml"
        temporary.write_text(content, encoding="utf-8")
        parsed = parse_user_config(str(temporary), log_summary=False)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return {
        "bundle": bundle,
        "recipe": name,
        "model_name": model_name,
        "catalog_version": version,
        "output": str(output.resolve()),
        "valid": True,
        "decisions": len(decisions),
        "excluded_decisions": sorted(excluded),
    }
