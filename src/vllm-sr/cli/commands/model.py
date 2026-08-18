"""Model command implementations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import yaml

from cli.commands.model_rendering import (
    render_builtin_models,
    render_configured_models,
    render_model_details,
)
from cli.model_catalog import (
    DEFAULT_CHANNEL,
    ModelCatalogError,
    catalog_model_to_dict,
    catalog_to_json,
    find_catalog_model,
    fork_catalog_models,
    load_all_model_catalogs,
    load_model_catalog,
    materialize_catalog_models,
)
from cli.parser import ConfigParseError, parse_user_config
from cli.terminal import echo
from cli.terminal import error as terminal_error
from cli.terminal import hint as terminal_hint
from cli.url_display import redact_url as _redact_url
from cli.validator import validate_user_config


def model_list_command(
    config_path: str | None = None,
    *,
    catalog_version: str = DEFAULT_CHANNEL,
    include_all_versions: bool = False,
    include_incompatible: bool = False,
    output: str = "table",
) -> None:
    """Print built-ins, optionally merged with or scoped to runtime config.

    Args:
        config_path: Explicit path to user config.yaml. When omitted, list the
            installed catalog and merge ``./config.yaml`` when it exists.
    """
    if config_path is None:
        try:
            catalogs = (
                load_all_model_catalogs()
                if include_all_versions
                else (load_model_catalog(catalog_version),)
            )
        except ModelCatalogError as error:
            _exit_with_error(str(error))
        models = [
            model
            for catalog in catalogs
            for model in catalog.models
            if include_incompatible or model.compatibility.compatible
        ]
        default_config = Path("config.yaml")
        configured = None
        if default_config.is_file():
            configured = _configured_models_payload(
                _parse_config_or_exit(default_config),
                default_config,
            )
        payload = {
            "catalogs": [
                {
                    "catalog_version": catalog.version,
                    "channel": catalog.channel,
                    "default_model": catalog.default_model,
                    "enabled_models": list(catalog.enabled_models),
                }
                for catalog in catalogs
            ],
            "models": [catalog_model_to_dict(model) for model in models],
            "configured": configured,
        }
        if output == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        render_builtin_models(models)
        if configured is not None:
            echo()
            render_configured_models(configured)
        return

    explicit_path = Path(config_path)
    if not explicit_path.exists():
        _exit_with_error(
            f"Config file not found: {config_path}",
            hint=(
                "Run 'vllm-sr serve' to bootstrap setup mode and create a config file."
            ),
        )

    user_config = _parse_config_or_exit(explicit_path)
    configured = _configured_models_payload(user_config, explicit_path)
    if output == "json":
        print(json.dumps(configured, indent=2, sort_keys=True))
        return

    render_configured_models(configured)


def model_show_command(
    model_id: str,
    *,
    catalog_version: str = DEFAULT_CHANNEL,
    output: str = "table",
) -> None:
    try:
        catalog, model = find_catalog_model(model_id, catalog_version=catalog_version)
    except ModelCatalogError as error:
        _exit_with_error(str(error))
    if output == "json":
        print(catalog_to_json(catalog, [model]))
        return
    details = catalog_model_to_dict(model)
    render_model_details(details)


def model_fork_command(
    model_ids: tuple[str, ...],
    destination: str,
    *,
    catalog_version: str = DEFAULT_CHANNEL,
    default_model: str | None = None,
) -> None:
    try:
        result = fork_catalog_models(
            model_ids,
            Path(destination),
            catalog_version=catalog_version,
            default_model=default_model,
        )
    except (ModelCatalogError, OSError) as error:
        _exit_with_error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


def model_validate_command(
    config_path: str,
    *,
    catalog_version: str = DEFAULT_CHANNEL,
) -> None:
    path = Path(config_path)
    if not path.is_file():
        _exit_with_error(f"Config file not found: {config_path}")
    try:
        user_config = parse_user_config(str(path), log_summary=False)
    except ConfigParseError as error:
        _exit_with_error(f"Failed to parse configuration: {error}")
    errors = validate_user_config(user_config, log_summary=False)
    if errors:
        for error in errors:
            terminal_error(str(error))
        sys.exit(1)

    try:
        catalog = load_model_catalog(catalog_version)
    except ModelCatalogError as error:
        _exit_with_error(str(error))
    models_by_entrypoint = {model.entrypoint: model for model in catalog.models}
    selected: list[str] = []
    for entrypoint in user_config.entrypoints:
        for name in entrypoint.model_names:
            model = models_by_entrypoint.get(name)
            if model is not None and model.id not in selected:
                selected.append(model.id)

    raw_document = yaml.safe_load(path.read_text(encoding="utf-8"))
    matches_catalog = False
    if selected and isinstance(raw_document, dict):
        try:
            expected = materialize_catalog_models(
                selected,
                catalog_version=catalog_version,
                default_model=selected[0],
            )
            matches_catalog = raw_document == expected.document
        except ModelCatalogError:
            matches_catalog = False
    status = "verified" if matches_catalog else "custom/unverified"
    print(
        json.dumps(
            {
                "valid": True,
                "catalog_version": catalog.version,
                "catalog_models": selected,
                "default_model": selected[0] if selected else None,
                "verification": status,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_config_or_exit(path: Path):
    try:
        return parse_user_config(str(path), log_summary=False)
    except ConfigParseError as error:
        _exit_with_error(f"Failed to parse configuration: {error}")


def _exit_with_error(message: str, *, hint: str | None = None) -> NoReturn:
    """Exit one model command with clean, Click-native stderr output."""
    terminal_error(message)
    if hint:
        terminal_hint(hint)
    raise SystemExit(1)


def _configured_models_payload(user_config, path: Path) -> dict:
    source = f"config:{path.resolve()}"
    providers = getattr(user_config, "providers", None)
    default_model = (
        providers.defaults.default_model
        if providers is not None and providers.defaults is not None
        else None
    )
    provider_models = []
    for model in list(getattr(providers, "models", []) or []):
        provider_models.append(
            {
                "id": model.name,
                "kind": "provider",
                "source": source,
                "default": model.name == default_model,
                "provider_model_id": model.provider_model_id,
                "reasoning_family": model.reasoning_family,
                "api_format": model.api_format,
                "backends": [
                    {
                        "name": ref.name,
                        "provider": ref.provider,
                        "base_url": _redact_url(ref.base_url or ref.endpoint or ""),
                        "protocol": ref.protocol,
                        "weight": ref.weight,
                    }
                    for ref in list(model.backend_refs or [])
                ],
            }
        )

    routing = getattr(user_config, "routing", None)
    model_cards = []
    for card in list(getattr(routing, "model_cards", []) or []):
        model_cards.append(
            {
                "id": card.name,
                "kind": "model_card",
                "source": source,
                "modality": card.modality,
                "capabilities": list(card.capabilities or []),
                "tags": list(card.tags or []),
                "context_window_size": card.context_window_size,
                "param_size": card.param_size,
                "loras": [adapter.name for adapter in list(card.loras or [])],
            }
        )
    return {
        "source": source,
        "path": str(path.resolve()),
        "default_model": default_model,
        "provider_models": provider_models,
        "model_cards": model_cards,
    }
