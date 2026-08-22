"""Materialize installed virtual-model selections for ``vllm-sr serve``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from cli.commands.runtime_paths import (
    private_runtime_state_nested_directory,
    write_runtime_recipe_asset_bytes,
)
from cli.deployment_backend import resolve_target
from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest_from_files
from cli.model_catalog import DEFAULT_CHANNEL, materialize_catalog_models
from cli.model_catalog_types import MaterializedCatalog, ModelCatalogError


@dataclass(frozen=True)
class ServeModelSource:
    """One immutable catalog projection owned by the local runtime workspace."""

    config_path: Path
    state_root: Path
    catalog_version: str
    enabled_models: tuple[str, ...]


def resolve_serve_model_request(
    model_ids: tuple[str, ...],
    *,
    config: str | None,
    catalog_version: str | None,
    algorithm: str | None,
    target: str | None,
) -> tuple[str, ServeModelSource | None]:
    """Resolve mutually exclusive config and catalog CLI source modes."""

    if model_ids and config is not None:
        raise ValueError(
            "MODEL operands and --config are mutually exclusive. Use catalog virtual "
            "models directly, or connect user-owned models through one config."
        )
    if model_ids and algorithm is not None:
        raise ValueError(
            "--algorithm applies only to user-owned configs. Catalog MODEL operands "
            "use their verified recipe algorithms; fork the model and serve the "
            "edited config to override them."
        )
    if not model_ids:
        if catalog_version is not None:
            raise ValueError("--catalog-version requires at least one MODEL operand")
        return config or "config.yaml", None
    if resolve_target(target) != "docker":
        raise ValueError(
            "serve MODEL currently supports the local Docker target. Use 'model fork' "
            "plus the chart or operator workflow for Kubernetes."
        )
    source = materialize_serve_model_source(model_ids, catalog_version=catalog_version)
    return str(source.config_path), source


def materialize_serve_model_source(
    model_ids: tuple[str, ...],
    *,
    catalog_version: str | None = None,
) -> ServeModelSource:
    """Write a deterministic managed Recipe for selected virtual models."""

    requested = tuple(model_id.strip() for model_id in model_ids if model_id.strip())
    if not requested:
        raise ValueError("at least one built-in virtual model is required")

    invalid = tuple(
        model_id for model_id in requested if not model_id.startswith("vllm-sr/")
    )
    if invalid:
        raise ValueError(
            "serve MODEL accepts installed vllm-sr virtual model IDs, not provider "
            "aliases or model checkpoints. Connect your own model with --config or "
            "the Dashboard: " + ", ".join(invalid)
        )

    materialized = materialize_catalog_models(
        requested,
        catalog_version=catalog_version or DEFAULT_CHANNEL,
        # The first operand owns deterministic presentation order. Requests still
        # name an explicit virtual entrypoint; this is not a physical-model fallback.
        default_model=requested[0],
    )
    encoded = yaml.safe_dump(materialized.document, sort_keys=False).encode("utf-8")
    files = _project_catalog_recipe_files(materialized, encoded)
    digest = model_bundle_digest_from_files(files).removeprefix("sha256:")
    state_root = _state_root()
    source_dir = private_runtime_state_nested_directory(
        state_root,
        "catalog-sources",
        f"recipe-{digest[:24]}",
    )
    for name in MODEL_BUNDLE_FILES:
        write_runtime_recipe_asset_bytes(source_dir / name, files[name])
    source_path = source_dir / "config.yaml"
    return ServeModelSource(
        config_path=source_path,
        state_root=state_root,
        catalog_version=materialized.catalog.version,
        enabled_models=materialized.enabled_models,
    )


def _project_catalog_recipe_files(
    materialized: MaterializedCatalog, encoded_config: bytes
) -> dict[str, bytes]:
    """Project one verified catalog bundle into the active five-file contract."""

    models = {model.id: model for model in materialized.catalog.models}
    asset_ids = {
        models[model_id].asset
        for model_id in materialized.enabled_models
        if model_id in models
    }
    if len(asset_ids) != 1:
        raise ModelCatalogError(
            "serve MODEL operands must belong to one managed built-in Recipe; "
            "use 'model fork' to combine models from different Recipe assets"
        )
    asset_id = next(iter(asset_ids))
    asset = materialized.catalog.assets[asset_id]
    bundle = resources.files("cli.model_assets").joinpath(
        materialized.catalog.version, asset["bundle"]
    )
    files = {name: bundle.joinpath(name).read_bytes() for name in MODEL_BUNDLE_FILES}
    files["config.yaml"] = encoded_config
    files["probes.yaml"] = _project_catalog_probes(
        files["probes.yaml"], materialized.enabled_models
    )
    return files


def _project_catalog_probes(
    encoded_probes: bytes, enabled_models: tuple[str, ...]
) -> bytes:
    """Retain only probes that target entrypoints started by this serve call."""

    try:
        document: Any = yaml.safe_load(encoded_probes)
    except yaml.YAMLError as error:
        raise ModelCatalogError("built-in Recipe probes are invalid") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("decisions"), list
    ):
        raise ModelCatalogError("built-in Recipe probes are invalid")
    enabled = set(enabled_models)
    decisions = [
        decision
        for decision in document["decisions"]
        if isinstance(decision, dict) and decision.get("model") in enabled
    ]
    if not decisions:
        raise ModelCatalogError("built-in Recipe has no probes for the selected models")
    document["decisions"] = decisions
    return yaml.safe_dump(document, sort_keys=False).encode("utf-8")


def _state_root() -> Path:
    configured = os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
    return (
        Path(configured).expanduser().absolute()
        if configured
        else Path.cwd().absolute()
    )
