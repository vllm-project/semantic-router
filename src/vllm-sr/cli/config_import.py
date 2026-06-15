"""Helpers for importing external config sources into canonical VSR config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from cli.config_migration import migrate_config_data
from cli.consts import DEFAULT_LISTENER_PORT
from cli.models import UserConfig
from cli.parser import ConfigParseError, load_config_file
from cli.utils import get_logger
from cli.validator import validate_user_config

log = get_logger(__name__)

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


@dataclass(frozen=True)
class ImportResult:
    """Result of importing OpenClaw config into canonical VSR config."""

    source_path: Path
    source_backup_path: Path
    target_path: Path
    target_backup_path: Path | None
    rewritten_base_url: str
    imported_models: list[ImportedModel]


def import_config_command(
    from_type: str,
    source_path: str | None = None,
    target_path: str = "config.yaml",
    force: bool = False,
) -> ImportResult:
    """Import an external config source into canonical VSR config."""

    normalized_from = (from_type or "").strip().lower()
    if normalized_from != "openclaw":
        raise ConfigImportError(
            f"Unsupported import source '{from_type}'. Supported values: openclaw."
        )

    resolved_source = discover_openclaw_config(source_path)
    source_raw, source_data = load_openclaw_source(resolved_source)
    imported_models = collect_openclaw_models(source_data)
    resolved_target = Path(target_path).expanduser()
    target = load_or_bootstrap_target_config(resolved_target)

    merge_openclaw_models_into_target(target, imported_models)
    rewritten_base_url = build_listener_base_url(target)
    rewrite_openclaw_source(source_data, imported_models, rewritten_base_url)
    validate_import_result(target)

    resolved_target.parent.mkdir(parents=True, exist_ok=True)

    target_backup_path = (
        backup_path_for(resolved_target) if resolved_target.exists() else None
    )
    source_backup_path = backup_path_for(resolved_source)
    ensure_backup_paths_available(
        target_backup_path=target_backup_path,
        source_backup_path=source_backup_path,
        force=force,
    )

    if target_backup_path is not None:
        target_backup_path.write_text(
            resolved_target.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    resolved_target.write_text(
        yaml.safe_dump(target, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    source_backup_path.write_text(source_raw, encoding="utf-8")
    resolved_source.write_text(
        json.dumps(source_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log.info("Imported OpenClaw configuration written successfully")
    log.info(f"  Source: {resolved_source}")
    log.info(f"  Source backup: {source_backup_path}")
    log.info(f"  Target: {resolved_target}")
    if target_backup_path is not None:
        log.info(f"  Target backup: {target_backup_path}")
    log.info(f"  Rewritten OpenClaw base URL: {rewritten_base_url}")
    log.info(
        "  Imported models: %s",
        ", ".join(model.logical_name for model in imported_models),
    )

    return ImportResult(
        source_path=resolved_source,
        source_backup_path=source_backup_path,
        target_path=resolved_target,
        target_backup_path=target_backup_path,
        rewritten_base_url=rewritten_base_url,
        imported_models=imported_models,
    )


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

    providers = openclaw_provider_block(source_data)
    raw_entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    unsupported: list[str] = []
    duplicate_provider_model_ids: list[str] = []

    for provider_key, provider_config in providers.items():
        provider_entries, provider_unsupported, provider_duplicates = (
            collect_provider_entries(provider_key, provider_config)
        )
        if provider_unsupported:
            unsupported.append(provider_unsupported)
            continue
        raw_entries.extend(provider_entries)
        duplicate_provider_model_ids.extend(provider_duplicates)

    if unsupported:
        raise ConfigImportError(
            "Unsupported OpenClaw provider API families: "
            f"{', '.join(unsupported)}. "
            "Supported providers must use an OpenAI-compatible api value."
        )

    if duplicate_provider_model_ids:
        raise ConfigImportError(
            "OpenClaw config contains duplicate model ids within a provider: "
            f"{', '.join(duplicate_provider_model_ids)}."
        )

    if not raw_entries:
        raise ConfigImportError(
            "OpenClaw config contains no importable provider models under models.providers."
        )

    model_id_counts: dict[str, int] = {}
    for _, _, model in raw_entries:
        model_id = str(model["id"]).strip()
        model_id_counts[model_id] = model_id_counts.get(model_id, 0) + 1

    imported_models: list[ImportedModel] = []
    for provider_key, provider_config, model in raw_entries:
        source_model_id = str(model["id"]).strip()
        logical_name = (
            source_model_id
            if model_id_counts[source_model_id] == 1
            else f"{provider_key}/{source_model_id}"
        )
        imported_models.append(
            ImportedModel(
                provider_key=provider_key,
                source_model_id=source_model_id,
                logical_name=logical_name,
                provider_config=provider_config,
                model_config=model,
            )
        )
    return imported_models


def load_or_bootstrap_target_config(target_path: Path) -> dict[str, Any]:
    """Load an existing target config or bootstrap a minimal canonical config."""

    if target_path.exists():
        if target_path.is_dir():
            raise ConfigImportError(f"Target config path is a directory: {target_path}")
        try:
            data = load_config_file(str(target_path))
        except ConfigParseError as exc:
            raise ConfigImportError(
                f"Failed to read target config {target_path}: {exc}"
            ) from exc
        target = migrate_config_data(data)
    else:
        target = build_minimal_target_config()

    if not isinstance(target, dict):
        raise ConfigImportError(
            f"Target config {target_path} did not resolve to a mapping."
        )

    normalize_target_shape(target)
    return target


def build_minimal_target_config() -> dict[str, Any]:
    """Build the minimal canonical config used when the target does not exist."""

    return {
        "version": "v0.3",
        "listeners": [default_listener()],
        "providers": {
            "defaults": {},
            "models": [],
        },
        "routing": {
            "modelCards": [],
            "signals": {},
            "decisions": [],
        },
    }


def default_listener() -> dict[str, Any]:
    """Return the default local listener used by bootstrapped imports."""

    return {
        "name": f"http-{DEFAULT_LISTENER_PORT}",
        "address": "0.0.0.0",
        "port": DEFAULT_LISTENER_PORT,
        "timeout": "300s",
    }


def merge_openclaw_models_into_target(
    target: dict[str, Any],
    imported_models: list[ImportedModel],
) -> None:
    """Merge imported models into the canonical target config."""

    providers = target["providers"]
    routing = target["routing"]
    provider_models = providers["models"]
    routing_cards = routing["modelCards"]
    defaults = providers["defaults"]

    provider_models_by_name = {
        str(model.get("name", "")).strip(): model
        for model in provider_models
        if isinstance(model, dict) and str(model.get("name", "")).strip()
    }
    routing_cards_by_name = {
        str(card.get("name", "")).strip(): card
        for card in routing_cards
        if isinstance(card, dict) and str(card.get("name", "")).strip()
    }

    for imported_model in imported_models:
        provider_model = provider_models_by_name.get(imported_model.logical_name)
        if provider_model is None:
            provider_model = {"name": imported_model.logical_name}
            provider_models.append(provider_model)
            provider_models_by_name[imported_model.logical_name] = provider_model

        provider_model["provider_model_id"] = imported_model.source_model_id
        provider_model["api_format"] = provider_model.get("api_format") or "openai"

        external_model_ids = provider_model.get("external_model_ids")
        if not isinstance(external_model_ids, dict):
            external_model_ids = {}
            provider_model["external_model_ids"] = external_model_ids
        external_model_ids["openai"] = imported_model.source_model_id

        provider_model["backend_refs"] = [build_backend_ref(imported_model)]

        routing_card = routing_cards_by_name.get(imported_model.logical_name)
        if routing_card is None:
            routing_card = {"name": imported_model.logical_name}
            routing_cards.append(routing_card)
            routing_cards_by_name[imported_model.logical_name] = routing_card

        routing_card["name"] = imported_model.logical_name
        context_window = positive_int(imported_model.model_config.get("contextWindow"))
        if context_window is not None:
            routing_card["context_window_size"] = context_window
        capabilities = build_capabilities(imported_model.model_config)
        if capabilities:
            routing_card["capabilities"] = capabilities
        routing_card.setdefault(
            "description",
            f"Imported from OpenClaw provider '{imported_model.provider_key}'.",
        )

    provider_names = [
        str(model.get("name", "")).strip()
        for model in provider_models
        if isinstance(model, dict) and str(model.get("name", "")).strip()
    ]
    default_model = str(defaults.get("default_model", "") or "").strip()
    if not default_model or default_model not in provider_names:
        defaults["default_model"] = imported_models[0].logical_name

    decisions = routing["decisions"]
    if not decisions:
        routing["decisions"] = [build_default_decision(defaults["default_model"])]


def build_backend_ref(imported_model: ImportedModel) -> dict[str, Any]:
    """Build one canonical backend ref from an OpenClaw provider definition."""

    backend_ref: dict[str, Any] = {
        "name": imported_model.provider_key,
        "base_url": str(
            imported_model.provider_config.get("baseUrl", "") or ""
        ).strip(),
        "provider": "openai",
        "weight": 1,
    }

    api_key = str(imported_model.provider_config.get("apiKey", "") or "").strip()
    if api_key and api_key != "not-needed":
        backend_ref["api_key"] = api_key
        backend_ref["auth_header"] = "Authorization"
        backend_ref["auth_prefix"] = "Bearer"

    headers = imported_model.provider_config.get("headers")
    if isinstance(headers, dict):
        extra_headers = {
            str(key): str(value)
            for key, value in headers.items()
            if value is not None and str(key)
        }
        if extra_headers:
            backend_ref["extra_headers"] = extra_headers

    return backend_ref


def build_capabilities(model_config: dict[str, Any]) -> list[str]:
    """Map OpenClaw model metadata into routing.modelCards capabilities."""

    capabilities: list[str] = []
    inputs = model_config.get("input")
    normalized_inputs = []
    if isinstance(inputs, list):
        normalized_inputs = [
            str(item).strip().lower() for item in inputs if str(item).strip()
        ]

    if not normalized_inputs or "text" in normalized_inputs:
        capabilities.append("chat")
    if "image" in normalized_inputs:
        capabilities.append("vision")
    if "audio" in normalized_inputs:
        capabilities.append("audio")
    if bool(model_config.get("reasoning")):
        capabilities.append("reasoning")

    seen: set[str] = set()
    unique_capabilities: list[str] = []
    for capability in capabilities:
        if capability in seen:
            continue
        seen.add(capability)
        unique_capabilities.append(capability)
    return unique_capabilities


def build_default_decision(default_model: str) -> dict[str, Any]:
    """Build the catch-all decision used for a bootstrapped target."""

    return {
        "name": "openclaw-import-default",
        "description": "Catch-all route bootstrapped from OpenClaw import.",
        "priority": 100,
        "rules": {
            "operator": "AND",
            "conditions": [],
        },
        "modelRefs": [
            {
                "model": default_model,
            }
        ],
    }


def build_listener_base_url(target: dict[str, Any]) -> str:
    """Resolve the first listener into a local OpenClaw-compatible base URL."""

    listeners = target.get("listeners")
    if not isinstance(listeners, list) or not listeners:
        raise ConfigImportError(
            "Imported config must declare at least one listener before OpenClaw can be rewritten."
        )

    first_listener = listeners[0]
    if not isinstance(first_listener, dict):
        raise ConfigImportError("The first listener must be a mapping.")

    port = positive_int(first_listener.get("port")) or DEFAULT_LISTENER_PORT
    raw_address = str(first_listener.get("address", "") or "").strip()
    host = "127.0.0.1" if raw_address in {"", "0.0.0.0", "::", "[::]"} else raw_address
    return f"http://{host}:{port}/v1"


def rewrite_openclaw_source(
    source_data: dict[str, Any],
    imported_models: list[ImportedModel],
    rewritten_base_url: str,
) -> None:
    """Rewrite imported OpenClaw provider base URLs and collision-renamed model ids."""

    providers = source_data["models"]["providers"]
    replacements: dict[str, str] = {}
    renamed_model_ids: list[tuple[str, str, str]] = []

    for imported_model in imported_models:
        provider_config = providers.get(imported_model.provider_key)
        if not isinstance(provider_config, dict):
            continue
        provider_config["baseUrl"] = rewritten_base_url
        collect_rewrite_changes(
            provider_config.get("models"),
            imported_model,
            replacements,
            renamed_model_ids,
        )

    if replacements:
        replace_openclaw_model_refs(source_data, replacements)

    apply_renamed_model_ids(providers, renamed_model_ids)


def replace_openclaw_model_refs(node: Any, replacements: dict[str, str]) -> Any:
    """Recursively replace OpenClaw provider/model reference strings."""

    if isinstance(node, dict):
        for key, value in list(node.items()):
            node[key] = replace_openclaw_model_refs(value, replacements)
        return node
    if isinstance(node, list):
        for index, value in enumerate(list(node)):
            node[index] = replace_openclaw_model_refs(value, replacements)
        return node
    if isinstance(node, str):
        return replacements.get(node, node)
    return node


def validate_import_result(target: dict[str, Any]) -> None:
    """Validate the merged canonical config before writing it to disk."""

    try:
        config = UserConfig(**target)
    except PydanticValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigImportError(
            f"Imported config does not satisfy the canonical schema: {details}"
        ) from exc

    errors = validate_user_config(config)
    if errors:
        rendered = "; ".join(str(error) for error in errors)
        raise ConfigImportError(f"Imported config failed validation: {rendered}")


def backup_path_for(path: Path) -> Path:
    """Return the default backup path for an imported or rewritten file."""

    return path.with_name(f"{path.name}.bak")


def ensure_backup_paths_available(
    target_backup_path: Path | None,
    source_backup_path: Path,
    force: bool,
) -> None:
    """Fail before writing when backup files already exist and --force is absent."""

    existing: list[str] = []
    for path in (target_backup_path, source_backup_path):
        if path is not None and path.exists():
            existing.append(str(path))

    if existing and not force:
        raise ConfigImportError(
            "Backup file already exists: "
            f"{', '.join(existing)}. Use --force to overwrite backup files."
        )


def positive_int(value: Any) -> int | None:
    """Return a positive integer value or None."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def openclaw_provider_block(source_data: dict[str, Any]) -> dict[str, Any]:
    """Return the OpenClaw provider block or raise an actionable error."""

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


def collect_provider_entries(
    provider_key: str,
    provider_config: Any,
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], str | None, list[str]]:
    """Collect valid model entries for one OpenClaw provider."""

    if not isinstance(provider_config, dict):
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' must be a JSON object."
        )

    provider_api = str(provider_config.get("api", "") or "").strip().lower()
    if provider_api and not provider_api.startswith(SUPPORTED_OPENCLAW_API_PREFIXES):
        return [], f"{provider_key} ({provider_api})", []

    base_url = str(provider_config.get("baseUrl", "") or "").strip()
    if not base_url:
        raise ConfigImportError(
            f"OpenClaw provider '{provider_key}' is missing baseUrl."
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
        model_id = validate_provider_model(provider_key, model)
        if model_id in seen_ids:
            duplicates.append(f"{provider_key}/{model_id}")
            continue
        seen_ids.add(model_id)
        entries.append((provider_key, provider_config, model))

    return entries, None, duplicates


def validate_provider_model(provider_key: str, model: Any) -> str:
    """Validate one OpenClaw provider model entry and return its id."""

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


def normalize_target_shape(target: dict[str, Any]) -> None:
    """Normalize a loaded target into the canonical blocks required by import."""

    target["version"] = "v0.3"
    target.pop("setup", None)

    listeners = ensure_list(target, "listeners")
    if not listeners:
        target["listeners"] = [default_listener()]

    providers = ensure_mapping(target, "providers")
    ensure_mapping(providers, "defaults")
    ensure_list(providers, "models")

    routing = ensure_mapping(target, "routing")
    ensure_list(routing, "modelCards")
    ensure_mapping(routing, "signals")
    ensure_list(routing, "decisions")


def ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Ensure a nested mapping key exists."""

    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def ensure_list(parent: dict[str, Any], key: str) -> list[Any]:
    """Ensure a nested list key exists."""

    value = parent.get(key)
    if not isinstance(value, list):
        value = []
        parent[key] = value
    return value


def collect_rewrite_changes(
    models: Any,
    imported_model: ImportedModel,
    replacements: dict[str, str],
    renamed_model_ids: list[tuple[str, str, str]],
) -> None:
    """Collect provider/model ref replacements and queued model-id renames."""

    if not isinstance(models, list):
        return

    for model in models:
        if (
            isinstance(model, dict)
            and str(model.get("id", "") or "").strip() == imported_model.source_model_id
        ):
            if imported_model.logical_name != imported_model.source_model_id:
                old_ref = (
                    f"{imported_model.provider_key}/{imported_model.source_model_id}"
                )
                new_ref = f"{imported_model.provider_key}/{imported_model.logical_name}"
                replacements[old_ref] = new_ref
                renamed_model_ids.append(
                    (
                        imported_model.provider_key,
                        imported_model.source_model_id,
                        imported_model.logical_name,
                    )
                )
            return


def apply_renamed_model_ids(
    providers: dict[str, Any],
    renamed_model_ids: list[tuple[str, str, str]],
) -> None:
    """Apply queued OpenClaw model-id renames after reference replacement."""

    for provider_key, source_model_id, logical_name in renamed_model_ids:
        provider_config = providers.get(provider_key)
        if not isinstance(provider_config, dict):
            continue
        models = provider_config.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if (
                isinstance(model, dict)
                and str(model.get("id", "") or "").strip() == source_model_id
            ):
                model["id"] = logical_name
                break
