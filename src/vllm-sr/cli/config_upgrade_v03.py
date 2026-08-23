"""Strict offline upgrade from canonical v0.3 YAML to v0.4 YAML."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from cli.config_contract import (
    DEFAULT_BACKEND_DISPATCH,
    DEFAULT_BACKEND_EGRESS_POLICY_FILE,
)
from cli.config_upgrade_v03_models import (
    translate_v03_models,
    v03_default_model,
)
from cli.config_upgrade_v03_routing import translate_v03_routing
from cli.config_upgrade_v03_support import (
    ConfigMigrationError,
    MigrationContext,
    MigrationIssue,
    as_mapping,
    reject_unknown_fields,
    scan_plaintext_secrets,
)
from cli.models import UserConfig
from cli.terminal import fields, heading, success
from cli.validator import validate_user_config

SOURCE_VERSION = "v0.3"
TARGET_VERSION = "v0.4"
_V03_TOP_LEVEL_FIELDS = frozenset(
    {
        "version",
        "listeners",
        "providers",
        "routing",
        "entrypoints",
        "recipes",
        "global",
        "setup",
    }
)


@dataclass(frozen=True)
class MigrationSummary:
    """Bounded success metadata for a completed translation."""

    source_version: str
    target_version: str
    models: int
    recipes: int
    entrypoints: int
    credential_references: int


@dataclass(frozen=True)
class MigrationResult:
    """Validated v0.4 document and its success metadata."""

    document: dict[str, Any]
    summary: MigrationSummary


def migrate_v03_config_data(source: dict[str, Any]) -> MigrationResult:
    """Compile exactly one v0.3 mapping into a validated v0.4 mapping."""

    context = MigrationContext()
    reject_unknown_fields(source, _V03_TOP_LEVEL_FIELDS, "config", context)
    version = source.get("version")
    if version != SOURCE_VERSION:
        if version == TARGET_VERSION:
            resolution = "use the file directly; it already uses the v0.4 contract"
        else:
            resolution = "first upgrade to canonical v0.3, then run this converter"
        context.add(
            "unsupported_source_version",
            "version",
            f"expected exactly {SOURCE_VERSION}, found {version!r}",
            resolution,
        )
    if source.get("setup") not in (None, {}):
        context.add(
            "removed_setup_block",
            "setup",
            "CLI setup state is not part of the v0.4 Router manifest",
            "remove setup after moving deployment choices to CLI flags or deployment values",
        )
    scan_plaintext_secrets(source, context)

    providers = source.get("providers")
    default_model = v03_default_model(providers)
    routing_result = translate_v03_routing(source, default_model, context)
    top_routing = as_mapping(source.get("routing"), "routing", context)
    model_result = translate_v03_models(
        providers,
        top_routing.get("modelCards"),
        routing_result.reasoning_efforts,
        context,
    )

    global_config = routing_result.global_config
    _merge_runtime_services(global_config, model_result.credentials, context)
    document: dict[str, Any] = {
        "version": TARGET_VERSION,
        "listeners": source.get("listeners") or [],
        "models": model_result.models,
        "recipes": routing_result.recipes,
        "entrypoints": routing_result.entrypoints,
        "global": global_config,
    }
    if model_result.billing_currency:
        document["billing_currency"] = model_result.billing_currency

    context.raise_if_blocked()
    _validate_v04_document(document)
    return MigrationResult(
        document=document,
        summary=MigrationSummary(
            source_version=SOURCE_VERSION,
            target_version=TARGET_VERSION,
            models=len(model_result.models),
            recipes=len(routing_result.recipes),
            entrypoints=len(routing_result.entrypoints),
            credential_references=len(model_result.credentials),
        ),
    )


def migrate_config_command(
    *,
    config_path: str,
    output_path: str | None = None,
    force: bool = False,
) -> MigrationResult:
    """Read, translate, validate, and atomically write one v0.3 config."""

    source_path = Path(config_path).expanduser()
    source = _load_source(source_path)
    result = migrate_v03_config_data(source)
    target_path = (
        Path(output_path).expanduser()
        if output_path
        else _default_output_path(source_path)
    )
    if source_path.resolve() == target_path.resolve():
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    code="source_overwrite_forbidden",
                    path=str(target_path),
                    message="the migration output cannot replace its v0.3 source",
                    resolution="choose a distinct --output path and retain the source for rollback",
                )
            ]
        )
    _write_output(target_path, result.document, force=force)

    success("Configuration migrated")
    heading("Files")
    fields(
        [
            ("Source", source_path),
            ("Output", target_path),
            ("Contract", f"{SOURCE_VERSION} → {TARGET_VERSION}"),
            ("Models", result.summary.models),
            ("Recipes", result.summary.recipes),
            ("Entrypoints", result.summary.entrypoints),
        ]
    )
    return result


def _merge_runtime_services(
    global_config: dict[str, Any],
    migrated_credentials: dict[str, dict[str, str]],
    context: MigrationContext,
) -> None:
    services_value = global_config.get("services")
    services = as_mapping(services_value, "global.services", context)
    if not isinstance(services_value, dict):
        global_config["services"] = services
    services.setdefault("backend_dispatch", dict(DEFAULT_BACKEND_DISPATCH))
    services.setdefault(
        "backend_egress",
        {"policy_file": DEFAULT_BACKEND_EGRESS_POLICY_FILE},
    )
    if not migrated_credentials:
        return

    credentials_value = services.get("backend_credentials")
    credentials = as_mapping(
        credentials_value,
        "global.services.backend_credentials",
        context,
    )
    if not isinstance(credentials_value, dict):
        services["backend_credentials"] = credentials
    for name, definition in migrated_credentials.items():
        current = credentials.get(name)
        if current not in (None, definition):
            context.add(
                "credential_name_collision",
                f"global.services.backend_credentials.{name}",
                "the generated environment-backed credential conflicts with an existing definition",
                "rename or consolidate the existing credential reference",
            )
            continue
        credentials[name] = definition


def _validate_v04_document(document: dict[str, Any]) -> None:
    issues: list[MigrationIssue] = []
    try:
        config = UserConfig.model_validate(document)
    except PydanticValidationError as error:
        for item in error.errors():
            path = ".".join(str(part) for part in item["loc"]) or "config"
            issues.append(
                MigrationIssue(
                    code="invalid_v04_schema",
                    path=path,
                    message=item["msg"],
                    resolution="update the named v0.3 source field to satisfy the documented v0.4 shape",
                )
            )
        raise ConfigMigrationError(issues) from error

    validation_errors = validate_user_config(config, log_summary=False)
    for error in validation_errors:
        issues.append(
            MigrationIssue(
                code="invalid_v04_semantics",
                path=error.field or "config",
                message=error.message,
                resolution="repair the referenced Model, Recipe, or Entrypoint relationship and rerun",
            )
        )
    if issues:
        raise ConfigMigrationError(issues)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "source_not_found",
                    str(path),
                    "configuration file does not exist",
                    "pass an existing v0.3 YAML file with --config",
                )
            ]
        )
    if not path.is_file():
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "source_not_file",
                    str(path),
                    "configuration path is not a regular file",
                    "pass a v0.3 YAML file",
                )
            ]
        )
    try:
        source_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_source_yaml",
                    str(path),
                    "the source could not be read as UTF-8 YAML",
                    "check the file encoding and permissions, then rerun",
                )
            ]
        ) from error
    try:
        value = yaml.load(source_text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f" near line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else ""
        )
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_source_yaml",
                    str(path),
                    f"the source is not unambiguous YAML{location}",
                    "repair the YAML or duplicate mapping key and rerun",
                )
            ]
        ) from error
    if not isinstance(value, dict):
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_source_root",
                    str(path),
                    "configuration root must be a YAML mapping",
                    "replace the document root with a v0.3 mapping",
                )
            ]
        )
    return value


def _default_output_path(source_path: Path) -> Path:
    suffix = source_path.suffix or ".yaml"
    stem = source_path.stem if source_path.suffix else source_path.name
    return source_path.with_name(f"{stem}.v0.4{suffix}")


def _write_output(path: Path, document: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "output_exists",
                    str(path),
                    "migration output already exists",
                    "choose another --output path or pass --force to replace only the output",
                )
            ]
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(
            document,
            sort_keys=False,
            allow_unicode=True,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(rendered)
                output.flush()
                os.fsync(output.fileno())
            if force:
                os.replace(temporary_path, path)
            else:
                try:
                    os.link(temporary_path, path)
                except FileExistsError as error:
                    raise ConfigMigrationError(
                        [
                            MigrationIssue(
                                "output_exists",
                                str(path),
                                "migration output appeared while the command was running",
                                "choose another --output path or rerun with --force",
                            )
                        ]
                    ) from error
                temporary_path.unlink()
        finally:
            temporary_path.unlink(missing_ok=True)
    except ConfigMigrationError:
        raise
    except OSError as error:
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "output_write_failed",
                    str(path),
                    str(error),
                    "check the output directory and permissions, then rerun",
                )
            ]
        ) from error
