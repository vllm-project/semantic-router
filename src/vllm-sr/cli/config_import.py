"""Helpers for importing external config sources into canonical VSR config."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from cli.config_contract import (
    DEFAULT_BACKEND_DISPATCH,
    DEFAULT_BACKEND_EGRESS_POLICY_FILE,
)
from cli.consts import DEFAULT_LISTENER_PORT
from cli.models import UserConfig
from cli.parser import ConfigParseError, load_config_file
from cli.terminal import fields, heading, success
from cli.validator import validate_user_config

OPENCLAW_CONFIG_ENV = "OPENCLAW_CONFIG_PATH"
SUPPORTED_OPENCLAW_API_PREFIXES = ("openai",)
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


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

    if is_managed_config(target):
        raise ConfigImportError(
            "Managed configuration is published through the Management API; "
            "config import will not rewrite a mounted runtime bootstrap file."
        )

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

    success("Configuration imported")
    heading("Files")
    output_fields: list[tuple[str, object]] = [
        ("Source", resolved_source),
        ("Source backup", source_backup_path),
        ("Target", resolved_target),
    ]
    if target_backup_path is not None:
        output_fields.append(("Target backup", target_backup_path))
    output_fields.extend(
        (
            ("Base URL", rewritten_base_url),
            (
                "Models",
                ", ".join(model.logical_name for model in imported_models),
            ),
        )
    )
    fields(output_fields)

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
        target = data
    else:
        target = build_minimal_target_config()

    if not isinstance(target, dict):
        raise ConfigImportError(
            f"Target config {target_path} did not resolve to a mapping."
        )
    if target.get("version") != "v0.4":
        raise ConfigImportError(
            f"Target config {target_path} must use the v0.4 authoring contract."
        )

    normalize_target_shape(target)
    return target


def build_minimal_target_config() -> dict[str, Any]:
    """Build the minimal canonical config used when the target does not exist."""

    return {
        "version": "v0.4",
        "listeners": [default_listener()],
        "models": [],
        "recipes": [],
        "entrypoints": [],
        "global": {
            "services": {
                "backend_dispatch": dict(DEFAULT_BACKEND_DISPATCH),
                "backend_egress": {"policy_file": DEFAULT_BACKEND_EGRESS_POLICY_FILE},
            }
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

    models = target["models"]
    models_by_name = {
        str(model.get("name", "")).strip(): model
        for model in models
        if isinstance(model, dict) and str(model.get("name", "")).strip()
    }

    for imported_model in imported_models:
        compiled = build_model(target, imported_model)
        current = models_by_name.get(imported_model.logical_name)
        if current is None:
            models.append(compiled)
        else:
            current.clear()
            current.update(compiled)
        models_by_name[imported_model.logical_name] = compiled
        ensure_import_entrypoint(target, compiled)


def build_model(
    target: dict[str, Any], imported_model: ImportedModel
) -> dict[str, Any]:
    """Translate one external provider value into readable Model authoring YAML."""

    connection = build_connection(target, imported_model)
    card: dict[str, Any] = {
        "capabilities": build_capabilities(imported_model.model_config),
    }
    description = str(imported_model.model_config.get("name", "") or "").strip()
    if description:
        card["description"] = description
    context_window = positive_int(imported_model.model_config.get("contextWindow"))
    if context_window is not None:
        card["context_window_size"] = context_window
    return {
        "name": imported_model.logical_name,
        "card": card,
        "connections": [connection],
    }


def build_connection(
    target: dict[str, Any], imported_model: ImportedModel
) -> dict[str, Any]:
    """Build one concise OpenAI-compatible Provider Integration binding."""

    raw_base_url = str(imported_model.provider_config.get("baseUrl", "") or "").strip()
    connection: dict[str, Any] = {
        "provider": "openai-compatible",
        "interface": "chat",
        "endpoint": raw_base_url,
        "model": imported_model.source_model_id,
    }
    credential_ref = import_credential_ref(target, imported_model)
    if credential_ref:
        connection["credential"] = credential_ref
    return connection


def import_credential_ref(
    target: dict[str, Any], imported_model: ImportedModel
) -> str | None:
    """Create a named environment-backed credential or reject plaintext."""

    api_key = str(imported_model.provider_config.get("apiKey", "") or "").strip()
    if not api_key or api_key == "not-needed":
        return None
    match = _ENV_REFERENCE.fullmatch(api_key)
    if match is None:
        raise ConfigImportError(
            f"OpenClaw provider '{imported_model.provider_key}' contains a plaintext "
            "API key. Replace it with an environment reference such as "
            "${PROVIDER_API_KEY} before importing."
        )
    credential_name = re.sub(
        r"[^a-z0-9_-]+", "_", f"{imported_model.provider_key}_credential".lower()
    ).strip("_")
    global_config = target.setdefault("global", {})
    services = global_config.setdefault("services", {})
    credentials = services.setdefault("backend_credentials", {})
    definition = {
        "credential_adapter_id": "bearer",
        "secret_env": match.group(1),
    }
    existing = credentials.get(credential_name)
    if existing not in (None, definition):
        raise ConfigImportError(
            f"Backend credential name collision for '{credential_name}'."
        )
    credentials[credential_name] = definition
    return credential_name


def ensure_import_entrypoint(target: dict[str, Any], model: dict[str, Any]) -> None:
    """Make each imported Model directly callable through a shared passthrough Recipe."""

    recipe_name = "openclaw-passthrough"
    decision_name = "route"
    recipes = target["recipes"]
    if not any(
        isinstance(recipe, dict) and recipe.get("name") == recipe_name
        for recipe in recipes
    ):
        recipes.append(
            {
                "name": recipe_name,
                "description": "Direct model routing for imported endpoints.",
                "document": {
                    "signals": {},
                    "projections": {},
                    "decisions": [
                        {
                            "name": decision_name,
                            "description": "Route every request.",
                            "priority": 100,
                            "rules": {"operator": "AND", "conditions": []},
                        }
                    ],
                },
            }
        )

    entrypoints = target["entrypoints"]
    public_name = imported_public_model_name(model["name"])
    compiled = {
        "name": public_name,
        "recipe": recipe_name,
        "assignments": {decision_name: {"models": [{"model": model["name"]}]}},
    }
    for index, entrypoint in enumerate(entrypoints):
        if isinstance(entrypoint, dict) and entrypoint.get("name") == public_name:
            entrypoints[index] = compiled
            break
    else:
        entrypoints.append(compiled)


def imported_public_model_name(logical_name: str) -> str:
    """Return the request-facing alias without colliding with physical Models."""

    return f"vllm-sr/imported/{logical_name}"


def build_capabilities(model_config: dict[str, Any]) -> list[str]:
    """Map external model metadata into canonical Model capabilities."""

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
    """Populate omitted v0.4 collection defaults required by import."""

    listeners = ensure_list(target, "listeners")
    if not listeners:
        target["listeners"] = [default_listener()]

    ensure_list(target, "models")
    ensure_list(target, "recipes")
    ensure_list(target, "entrypoints")
    global_config = target.setdefault("global", {})
    services = global_config.setdefault("services", {})
    services.setdefault("backend_dispatch", dict(DEFAULT_BACKEND_DISPATCH))
    services.setdefault(
        "backend_egress",
        {"policy_file": DEFAULT_BACKEND_EGRESS_POLICY_FILE},
    )


def is_managed_config(target: dict[str, Any]) -> bool:
    """Return whether a manifest is only a managed-mode bootstrap document."""

    global_config = target.get("global")
    if not isinstance(global_config, dict):
        return False
    control_plane = global_config.get("control_plane")
    return isinstance(control_plane, dict) and control_plane.get("mode") == "managed"


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
            public_name = imported_public_model_name(imported_model.logical_name)
            old_ref = f"{imported_model.provider_key}/{imported_model.source_model_id}"
            new_ref = f"{imported_model.provider_key}/{public_name}"
            replacements[old_ref] = new_ref
            renamed_model_ids.append(
                (
                    imported_model.provider_key,
                    imported_model.source_model_id,
                    public_name,
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
