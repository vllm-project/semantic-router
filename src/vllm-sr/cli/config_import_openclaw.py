"""Parse the OpenClaw model catalog accepted by the explicit config importer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENCLAW_CONFIG_ENV = "OPENCLAW_CONFIG_PATH"
SUPPORTED_OPENCLAW_API_PREFIXES = ("openai",)


class ConfigImportError(RuntimeError):
    """Raised when an import source cannot be converted safely."""


@dataclass(frozen=True)
class ImportedModel:
    """Imported OpenClaw model metadata and source payload."""

    provider_key: str
    source_model_id: str
    logical_name: str
    provider_config: dict[str, Any]
    model_config: dict[str, Any]


def discover_openclaw_config(source_path: str | None = None) -> Path:
    """Resolve the OpenClaw config path from an explicit path or discovery order."""

    if source_path:
        candidate = Path(source_path).expanduser()
        if not candidate.exists():
            raise ConfigImportError(f"OpenClaw config not found: {candidate}")
        if candidate.is_dir():
            raise ConfigImportError(f"OpenClaw config path is a directory: {candidate}")
        return candidate

    candidates: list[Path] = []
    if raw_env_path := os.getenv(OPENCLAW_CONFIG_ENV):
        candidates.append(Path(raw_env_path).expanduser())
    candidates.extend(
        [
            Path.cwd() / "openclaw.json",
            Path.home() / ".openclaw" / "openclaw.json",
        ]
    )
    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            return candidate
    raise ConfigImportError(
        "Could not find an OpenClaw config. "
        f"Checked {', '.join(checked)}. Set {OPENCLAW_CONFIG_ENV} or pass --source."
    )


def load_openclaw_source(source_path: Path) -> tuple[str, dict[str, Any]]:
    """Load OpenClaw JSON and return both raw text and parsed mapping."""

    try:
        raw = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigImportError(
            f"Failed to read OpenClaw config {source_path}: {exc}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigImportError(
            f"OpenClaw config {source_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigImportError(
            f"OpenClaw config {source_path} must contain a JSON object at the top level."
        )
    return raw, data


def collect_openclaw_models(source_data: dict[str, Any]) -> list[ImportedModel]:
    """Collect supported OpenClaw provider/model bindings."""

    raw_entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    unsupported: list[str] = []
    duplicates: list[str] = []
    for provider_key, provider_config in _provider_block(source_data).items():
        entries, unsupported_provider, provider_duplicates = _collect_provider_entries(
            provider_key, provider_config
        )
        if unsupported_provider:
            unsupported.append(unsupported_provider)
            continue
        raw_entries.extend(entries)
        duplicates.extend(provider_duplicates)
    if unsupported:
        raise ConfigImportError(
            "Unsupported OpenClaw provider API families: "
            f"{', '.join(unsupported)}. "
            "Supported providers must use an OpenAI-compatible api value."
        )
    if duplicates:
        raise ConfigImportError(
            "OpenClaw config contains duplicate model ids within a provider: "
            f"{', '.join(duplicates)}."
        )
    if not raw_entries:
        raise ConfigImportError(
            "OpenClaw config contains no importable provider models under models.providers."
        )
    model_id_counts: dict[str, int] = {}
    for _, _, model in raw_entries:
        model_id = str(model["id"]).strip()
        model_id_counts[model_id] = model_id_counts.get(model_id, 0) + 1
    return [
        ImportedModel(
            provider_key=provider_key,
            source_model_id=(source_model_id := str(model["id"]).strip()),
            logical_name=(
                source_model_id
                if model_id_counts[source_model_id] == 1
                else f"{provider_key}/{source_model_id}"
            ),
            provider_config=provider_config,
            model_config=model,
        )
        for provider_key, provider_config, model in raw_entries
    ]


def _provider_block(source_data: dict[str, Any]) -> dict[str, Any]:
    models_block = source_data.get("models")
    providers = (
        models_block.get("providers") if isinstance(models_block, dict) else None
    )
    if not isinstance(providers, dict) or not providers:
        raise ConfigImportError(
            "OpenClaw config is missing models.providers. "
            "Supported imports require models.providers.* with OpenAI-compatible endpoints."
        )
    return providers


def _collect_provider_entries(
    provider_key: str,
    provider_config: Any,
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], str | None, list[str]]:
    if not isinstance(provider_config, dict):
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' must be a JSON object."
        )
    provider_api = str(provider_config.get("api", "") or "").strip().lower()
    if provider_api and not provider_api.startswith(SUPPORTED_OPENCLAW_API_PREFIXES):
        return [], f"{provider_key} ({provider_api})", []
    if not str(provider_config.get("baseUrl", "") or "").strip():
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' is missing baseUrl."
        )
    headers = provider_config.get("headers")
    if isinstance(headers, dict) and any(
        value is not None and str(key).strip() for key, value in headers.items()
    ):
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' uses custom headers. "
            "Install a Provider Integration for that protocol instead of "
            "embedding transport details in Model YAML."
        )
    models = provider_config.get("models")
    if models in (None, []):
        return [], None, []
    if not isinstance(models, list):
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}'.models must be an array."
        )
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    duplicates: list[str] = []
    seen_ids: set[str] = set()
    for model in models:
        model_id = _validate_provider_model(provider_key, model)
        if model_id in seen_ids:
            duplicates.append(f"{provider_key}/{model_id}")
        else:
            seen_ids.add(model_id)
            entries.append((provider_key, provider_config, model))
    return entries, None, duplicates


def _validate_provider_model(provider_key: str, model: Any) -> str:
    if not isinstance(model, dict):
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' contains a non-object model entry."
        )
    model_id = str(model.get("id", "") or "").strip()
    if not model_id:
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' contains a model without id."
        )
    return model_id
