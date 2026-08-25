"""File-system boundary for the offline v0.3 configuration migration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cli.config_upgrade_v03 import (
    CONTRACT_VERSION,
    ConfigMigrationError,
    MigrationIssue,
    MigrationResult,
    migrate_v03_config_data,
)
from cli.terminal import fields, heading, success


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


def migrate_config_command(
    *,
    config_path: str,
    output_path: str | None = None,
    force: bool = False,
) -> MigrationResult:
    """Read, rewrite, validate, and atomically write one previous v0.3 file."""

    source_path = Path(config_path).expanduser()
    source = _load_source(source_path)
    result = migrate_v03_config_data(source)
    target_path = (
        Path(output_path).expanduser()
        if output_path
        else source_path.with_name(
            f"{source_path.stem}.migrated{source_path.suffix or '.yaml'}"
        )
    )
    if source_path.resolve() == target_path.resolve():
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "source_overwrite_forbidden",
                    str(target_path),
                    "migration output cannot replace its source",
                    "choose a distinct --output path and retain the source for rollback",
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
            ("Contract", f"{CONTRACT_VERSION} previous → {CONTRACT_VERSION} current"),
            ("Models", result.summary.models),
            ("Pricing", result.summary.pricing_blocks),
            ("Control", result.summary.control_blocks),
            ("Removed no-op fields", result.summary.removed_noop_fields),
        ]
    )
    return result


def _load_source(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_source",
                    str(path),
                    "source must be a regular file",
                    "provide a readable previous-release v0.3 file",
                )
            ]
        )
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_yaml",
                    str(path),
                    str(error),
                    "repair the source YAML and rerun",
                )
            ]
        ) from error
    if not isinstance(value, dict):
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_root",
                    "config",
                    "configuration root must be a mapping",
                    "provide one canonical v0.3 document",
                )
            ]
        )
    return value


def _write_output(path: Path, document: dict[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not force or not path.is_file() or path.is_symlink()):
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "output_exists",
                    str(path),
                    "output already exists or is not a regular file",
                    "choose another path or pass --force for a regular file",
                )
            ]
        )
    rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
