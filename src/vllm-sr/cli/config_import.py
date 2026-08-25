"""Helpers for importing external config sources into canonical VSR config."""

from __future__ import annotations

import json
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
from cli.config_import_openclaw import (
    ConfigImportError,
    ImportedModel,
    collect_openclaw_models,
    discover_openclaw_config,
    load_openclaw_source,
)
from cli.consts import DEFAULT_LISTENER_PORT
from cli.models import UserConfig
from cli.parser import ConfigParseError, load_config_file
from cli.terminal import fields, heading, success
from cli.validator import validate_user_config

_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


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

    if has_management_store(target):
        raise ConfigImportError(
            "This config uses a Management store. Import the generated v0.3 "
            "resources through POST /management/v1/routing/imports instead of "
            "rewriting the bootstrap file."
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
    if target.get("version") != "v0.3":
        raise ConfigImportError(
            f"Target config {target_path} must use the strict v0.3 authoring contract."
        )

    normalize_target_shape(target)
    return target


def build_minimal_target_config() -> dict[str, Any]:
    """Build the minimal canonical config used when the target does not exist."""

    return {
        "version": "v0.3",
        "listeners": [default_listener()],
        "providers": {"defaults": {}, "models": []},
        "routing": {"modelCards": []},
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

    models = target["providers"]["models"]
    model_cards = target["routing"]["modelCards"]
    models_by_name = {
        str(model.get("name", "")).strip(): model
        for model in models
        if isinstance(model, dict) and str(model.get("name", "")).strip()
    }
    cards_by_name = {
        str(card.get("name", "")).strip(): card
        for card in model_cards
        if isinstance(card, dict) and str(card.get("name", "")).strip()
    }

    for imported_model in imported_models:
        compiled, card = build_model(imported_model)
        current = models_by_name.get(imported_model.logical_name)
        if current is None:
            models.append(compiled)
        else:
            current.clear()
            current.update(compiled)
        models_by_name[imported_model.logical_name] = compiled
        current_card = cards_by_name.get(imported_model.logical_name)
        if current_card is None:
            model_cards.append(card)
        else:
            current_card.clear()
            current_card.update(card)
        cards_by_name[imported_model.logical_name] = card
        ensure_import_entrypoint(target, compiled)


def build_model(imported_model: ImportedModel) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate one external provider value into Provider Model plus Model card."""

    backend_ref = build_backend_ref(imported_model)
    card: dict[str, Any] = {
        "name": imported_model.logical_name,
        "capabilities": build_capabilities(imported_model.model_config),
    }
    description = str(imported_model.model_config.get("name", "") or "").strip()
    if description:
        card["description"] = description
    context_window = positive_int(imported_model.model_config.get("contextWindow"))
    if context_window is not None:
        card["context_window_size"] = context_window
    model = {
        "name": imported_model.logical_name,
        "provider_model_id": imported_model.source_model_id,
        "api_format": "openai",
        "backend_refs": [backend_ref],
    }
    return model, card


def build_backend_ref(imported_model: ImportedModel) -> dict[str, Any]:
    """Build one secret-safe OpenAI-compatible backend reference."""

    raw_base_url = str(imported_model.provider_config.get("baseUrl", "") or "").strip()
    backend_ref: dict[str, Any] = {
        "provider": "openai-compatible",
        "base_url": raw_base_url,
        "protocol": "https" if raw_base_url.startswith("https://") else "http",
    }
    api_key_env = imported_api_key_env(imported_model)
    if api_key_env:
        backend_ref["api_key_env"] = api_key_env
    return backend_ref


def imported_api_key_env(imported_model: ImportedModel) -> str | None:
    """Return one environment-backed provider credential or reject plaintext."""

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
    return match.group(1)


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
                "routing": {
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
        "model_names": [public_name],
        "recipe": recipe_name,
        "assignments": {decision_name: {"models": [{"model": model["name"]}]}},
    }
    for index, entrypoint in enumerate(entrypoints):
        if isinstance(entrypoint, dict) and public_name in (
            entrypoint.get("model_names") or []
        ):
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


def normalize_target_shape(target: dict[str, Any]) -> None:
    """Populate omitted v0.3 collection defaults required by import."""

    listeners = ensure_list(target, "listeners")
    if not listeners:
        target["listeners"] = [default_listener()]

    providers = target.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ConfigImportError("Target providers must be a mapping.")
    providers.setdefault("defaults", {})
    ensure_list(providers, "models")
    routing = target.setdefault("routing", {})
    if not isinstance(routing, dict):
        raise ConfigImportError("Target routing must be a mapping.")
    ensure_list(routing, "modelCards")
    ensure_list(target, "recipes")
    ensure_list(target, "entrypoints")
    global_config = target.setdefault("global", {})
    services = global_config.setdefault("services", {})
    services.setdefault("backend_dispatch", dict(DEFAULT_BACKEND_DISPATCH))
    services.setdefault(
        "backend_egress",
        {"policy_file": DEFAULT_BACKEND_EGRESS_POLICY_FILE},
    )


def has_management_store(target: dict[str, Any]) -> bool:
    """Return whether dynamic desired state is backed by a Management store."""

    global_config = target.get("global")
    if not isinstance(global_config, dict):
        return False
    stores = global_config.get("stores")
    return isinstance(stores, dict) and stores.get("management") is not None


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
