"""Materialize installed virtual-model selections for ``vllm-sr serve``."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from cli.commands.runtime_paths import (
    private_runtime_state_subdirectory,
    write_runtime_config_bytes,
)
from cli.deployment_backend import resolve_target
from cli.model_catalog import DEFAULT_CHANNEL, materialize_catalog_models


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
    """Write a deterministic private source config for selected virtual models."""

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
    digest = hashlib.sha256(encoded).hexdigest()
    state_root = _state_root()
    source_dir = private_runtime_state_subdirectory(state_root, "catalog-sources")
    source_path = source_dir / (f"{materialized.catalog.version}.{digest[:16]}.yaml")
    write_runtime_config_bytes(source_path, encoded)
    return ServeModelSource(
        config_path=source_path,
        state_root=state_root,
        catalog_version=materialized.catalog.version,
        enabled_models=materialized.enabled_models,
    )


def _state_root() -> Path:
    configured = os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
    return (
        Path(configured).expanduser().absolute()
        if configured
        else Path.cwd().absolute()
    )
