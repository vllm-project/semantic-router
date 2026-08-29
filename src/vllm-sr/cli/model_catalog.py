"""Versioned built-in virtual-model catalog loading and materialization."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from cli import __version__
from cli.model_bundle import model_bundle_digest
from cli.model_catalog_types import (
    CatalogCompatibility,
    CatalogComponentVersions,
    CatalogModel,
    ModelCatalog,
    ModelCatalogError,
)
from cli.model_catalog_validation import (
    _ASSET_KEYS,
    _CATALOG_RESOURCE_VERSION,
    _DEFAULT_KEYS,
    _MODEL_KEYS,
    _ROOT_KEYS,
    _VERIFICATION_KEYS,
    CATALOG_SCHEMA,
    DEFAULT_CHANNEL,
    SUPPORTED_CATALOG_FEATURES,
    SUPPORTED_CONFIG_SCHEMAS,
    SUPPORTED_MODEL_KINDS,
    SUPPORTED_PROTOCOLS,
    _compatibility,
    _merge_compatibility,
    _parse_roles,
    _reject_unknown_keys,
    _required_catalog_identity,
    _required_enum,
    _required_mapping,
    _required_positive_int,
    _required_public_model_id,
    _required_semver,
    _required_slug,
    _required_string,
    _unique_enum_strings,
    _unique_model_ids,
    _unique_slugs,
    _validate_catalog_binding,
    _validate_catalog_model_role_pool,
    _validate_compatibility_mapping,
    _validate_manifest_security,
)

__all__ = [
    "CATALOG_SCHEMA",
    "DEFAULT_CHANNEL",
    "SUPPORTED_CATALOG_FEATURES",
    "SUPPORTED_CONFIG_SCHEMAS",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_PROTOCOLS",
    "CatalogCompatibility",
    "CatalogComponentVersions",
    "CatalogModel",
    "ModelCatalog",
    "ModelCatalogError",
    "available_catalog_versions",
    "catalog_model_to_dict",
    "catalog_to_json",
    "find_catalog_model",
    "load_all_model_catalogs",
    "load_model_catalog",
]


@dataclass(frozen=True)
class _CatalogHeader:
    version: str
    channel: str
    default_model: str
    enabled_models: tuple[str, ...]
    compatibility: dict[str, Any]
    assets: dict[str, dict[str, str]]


def available_catalog_versions() -> tuple[str, ...]:
    root = resources.files("cli.model_assets")
    versions = {
        item.name
        for item in root.iterdir()
        if item.is_dir()
        and _CATALOG_RESOURCE_VERSION.fullmatch(item.name)
        and item.joinpath("catalog.yaml").is_file()
    }
    releases = sorted(
        (version for version in versions if version != DEFAULT_CHANNEL),
        key=lambda version: tuple(int(part) for part in version[1:].split(".")),
        reverse=True,
    )
    if DEFAULT_CHANNEL in versions:
        return (DEFAULT_CHANNEL, *releases)
    return tuple(releases)


def load_model_catalog(
    version: str = DEFAULT_CHANNEL,
    *,
    component_versions: CatalogComponentVersions | None = None,
) -> ModelCatalog:
    """Load a catalog for independently supplied CLI and Router versions.

    The installed CLI and the Router image it launches are co-versioned, so the
    default evaluates both constraints against this distribution version.
    Embedders validating a separately deployed Router can pass its actual
    version without conflating it with the CLI version.
    """

    if component_versions is None:
        component_versions = CatalogComponentVersions(
            cli=__version__, router=__version__
        )
    version, document = _load_catalog_document(version)
    header = _parse_catalog_header(document, version)
    models = _parse_catalog_models(
        document, version, header, component_versions=component_versions
    )
    known_models = {model.id for model in models}
    if header.default_model not in known_models or any(
        model_id not in known_models for model_id in header.enabled_models
    ):
        raise ModelCatalogError("catalog defaults reference unknown models")
    return ModelCatalog(
        version=header.version,
        channel=header.channel,
        default_model=header.default_model,
        enabled_models=header.enabled_models,
        assets=header.assets,
        models=models,
    )


def _load_catalog_document(version: str) -> tuple[str, dict[str, Any]]:
    version = version.strip()
    if not _CATALOG_RESOURCE_VERSION.fullmatch(version):
        raise ModelCatalogError("catalog version is invalid")
    catalog_resource = resources.files("cli.model_assets").joinpath(
        version, "catalog.yaml"
    )
    if not catalog_resource.is_file():
        raise ModelCatalogError(f"built-in catalog version is not installed: {version}")
    try:
        document = yaml.safe_load(catalog_resource.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ModelCatalogError("built-in model catalog YAML is invalid") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != CATALOG_SCHEMA
    ):
        raise ModelCatalogError("built-in model catalog schema is invalid")
    _validate_manifest_security(document)
    _reject_unknown_keys(document, _ROOT_KEYS, "root")
    return version, document


def _parse_catalog_header(
    document: dict[str, Any], resource_version: str
) -> _CatalogHeader:
    catalog_version = _required_catalog_identity(
        document, "catalog_version", special="latest"
    )
    channel = _required_enum(document, "channel", frozenset({"latest", "release"}))
    release = _required_catalog_identity(document, "release", special="unreleased")
    _validate_catalog_binding(resource_version, catalog_version, channel, release)
    defaults = _required_mapping(document, "defaults")
    _reject_unknown_keys(defaults, _DEFAULT_KEYS, "defaults")
    default_model = _required_public_model_id(defaults, "model")
    enabled_models = _unique_model_ids(defaults.get("enabled"), "defaults.enabled")
    if default_model not in enabled_models:
        raise ModelCatalogError("catalog default model must be enabled by default")
    catalog_compatibility = _required_mapping(document, "compatibility")
    _validate_compatibility_mapping(
        catalog_compatibility, "compatibility", require_complete=True
    )
    return _CatalogHeader(
        version=catalog_version,
        channel=channel,
        default_model=default_model,
        enabled_models=enabled_models,
        compatibility=catalog_compatibility,
        assets=_parse_assets(document.get("assets"), resource_version),
    )


def _parse_catalog_models(
    document: dict[str, Any],
    resource_version: str,
    header: _CatalogHeader,
    *,
    component_versions: CatalogComponentVersions,
) -> tuple[CatalogModel, ...]:
    asset_documents: dict[str, dict[str, Any]] = {}
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelCatalogError("built-in model catalog has no models")
    models: list[CatalogModel] = []
    seen: set[str] = set()
    seen_entrypoints: set[str] = set()
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ModelCatalogError("catalog model entry is invalid")
        _reject_unknown_keys(raw, _MODEL_KEYS, "model")
        model_id = _required_public_model_id(raw, "id")
        if model_id in seen:
            raise ModelCatalogError(f"duplicate catalog model: {model_id}")
        seen.add(model_id)
        asset = _required_slug(raw, "asset")
        if asset not in header.assets:
            raise ModelCatalogError(
                f"catalog model {model_id} references unknown asset"
            )
        verification = _parse_model_verification(raw, model_id, header.assets[asset])
        entrypoint = _required_public_model_id(raw, "entrypoint")
        if entrypoint in seen_entrypoints:
            raise ModelCatalogError(f"duplicate catalog entrypoint: {entrypoint}")
        seen_entrypoints.add(entrypoint)
        model = _parse_catalog_model(
            raw,
            model_id,
            asset=asset,
            entrypoint=entrypoint,
            verification=verification,
            resource_version=resource_version,
            header=header,
            asset_documents=asset_documents,
            component_versions=component_versions,
        )
        models.append(model)
    return tuple(models)


def _parse_catalog_model(
    raw: dict[str, Any],
    model_id: str,
    *,
    asset: str,
    entrypoint: str,
    verification: dict[str, str],
    resource_version: str,
    header: _CatalogHeader,
    asset_documents: dict[str, dict[str, Any]],
    component_versions: CatalogComponentVersions,
) -> CatalogModel:
    model_compatibility = raw.get("compatibility")
    if model_compatibility is not None:
        _validate_compatibility_mapping(
            model_compatibility, f"model {model_id} compatibility"
        )
    recipe = _required_slug(raw, "recipe")
    roles = _parse_roles(raw.get("roles"), model_id)
    if asset not in asset_documents:
        asset_documents[asset] = _load_asset_yaml(
            resource_version, header.assets[asset]
        )
    _validate_catalog_model_role_pool(model_id, recipe, roles, asset_documents[asset])
    return CatalogModel(
        id=model_id,
        display_name=_required_string(raw, "display_name"),
        description=_required_string(raw, "description"),
        kind=_required_enum(raw, "kind", SUPPORTED_MODEL_KINDS),
        family=_required_slug(raw, "family"),
        generation=_required_positive_int(raw, "generation"),
        policy_version=_required_semver(raw, "policy_version"),
        asset=asset,
        entrypoint=entrypoint,
        recipe=recipe,
        protocols=_unique_enum_strings(
            raw.get("protocols"), f"{model_id}.protocols", SUPPORTED_PROTOCOLS
        ),
        traits=_unique_slugs(raw.get("traits"), f"{model_id}.traits"),
        roles=roles,
        verification=verification,
        catalog_version=header.version,
        channel=header.channel,
        compatibility=_compatibility(
            _merge_compatibility(header.compatibility, model_compatibility),
            component_versions,
        ),
        enabled_by_default=model_id in header.enabled_models,
        default=model_id == header.default_model,
    )


def _parse_model_verification(
    raw: dict[str, Any], model_id: str, asset: dict[str, str]
) -> dict[str, str]:
    verification = _required_mapping(raw, "verification")
    _reject_unknown_keys(
        verification, _VERIFICATION_KEYS, f"model {model_id} verification"
    )
    authority = _required_slug(verification, "authority")
    digest = _required_string(verification, "asset_sha256")
    if digest != asset["sha256"]:
        raise ModelCatalogError(f"catalog model {model_id} verification digest drifted")
    return {"authority": authority, "asset_sha256": digest}


def load_all_model_catalogs(
    *, component_versions: CatalogComponentVersions | None = None
) -> tuple[ModelCatalog, ...]:
    return tuple(
        load_model_catalog(version, component_versions=component_versions)
        for version in available_catalog_versions()
    )


def find_catalog_model(
    model_id: str,
    *,
    catalog_version: str = DEFAULT_CHANNEL,
    component_versions: CatalogComponentVersions | None = None,
) -> tuple[ModelCatalog, CatalogModel]:
    catalog = load_model_catalog(catalog_version, component_versions=component_versions)
    for model in catalog.models:
        if model.id == model_id:
            return catalog, model
    raise ModelCatalogError(f"built-in model is not installed: {model_id}")


def catalog_to_json(catalog: ModelCatalog, models: Iterable[CatalogModel]) -> str:
    payload = {
        "catalog_version": catalog.version,
        "channel": catalog.channel,
        "models": [catalog_model_to_dict(model) for model in models],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def catalog_model_to_dict(model: CatalogModel) -> dict[str, Any]:
    verification = {
        "status": "verified" if model.verified else "unverified",
        "authority": model.verification.get("authority", ""),
        "asset_sha256": model.verification.get("asset_sha256", ""),
    }
    return {
        "id": model.id,
        "display_name": model.display_name,
        "description": model.description,
        "kind": model.kind,
        "family": model.family,
        "generation": model.generation,
        "policy_version": model.policy_version,
        "entrypoint": model.entrypoint,
        "recipe": model.recipe,
        "protocols": list(model.protocols),
        "traits": list(model.traits),
        "roles": list(model.roles),
        "catalog_version": model.catalog_version,
        "channel": model.channel,
        "compatible": model.compatibility.compatible,
        "compatibility_reason": model.compatibility.reason,
        "enabled_by_default": model.enabled_by_default,
        "default": model.default,
        "verification": verification,
    }


def _load_asset_yaml(version: str, asset: dict[str, str]) -> dict[str, Any]:
    bundle = resources.files("cli.model_assets").joinpath(version, asset["bundle"])
    try:
        digest = model_bundle_digest(bundle)
    except ValueError as error:
        raise ModelCatalogError(str(error)) from error
    if digest != asset["sha256"]:
        raise ModelCatalogError("built-in model asset digest verification failed")
    document = yaml.safe_load(bundle.joinpath("config.yaml").read_bytes())
    if not isinstance(document, dict):
        raise ModelCatalogError("built-in model asset is invalid")
    return document


def _parse_assets(value: Any, version: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ModelCatalogError("built-in catalog has no assets")
    assets: dict[str, dict[str, str]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ModelCatalogError("catalog asset entry is invalid")
        _reject_unknown_keys(item, _ASSET_KEYS, "asset")
        asset_id = _required_slug(item, "id")
        bundle = _required_slug(item, "bundle")
        digest = _required_string(item, "sha256")
        if asset_id in assets or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ModelCatalogError("catalog asset identity is invalid")
        if Path(bundle).name != bundle:
            raise ModelCatalogError("catalog asset path is invalid")
        resource = resources.files("cli.model_assets").joinpath(version, bundle)
        try:
            actual = model_bundle_digest(resource)
        except ValueError as error:
            raise ModelCatalogError(str(error)) from error
        if actual != digest:
            raise ModelCatalogError(f"catalog asset digest drifted: {bundle}")
        assets[asset_id] = {"bundle": bundle, "sha256": digest}
    return assets
