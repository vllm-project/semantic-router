"""YAML, JSON, and JSON Schema boundaries for catalog compilation."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from catalog_common import CatalogBuildError
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogBuildError(
            f"cannot load {path.relative_to(REPO_ROOT)}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CatalogBuildError(f"{path.relative_to(REPO_ROOT)} must be an object")
    return value


def validate_schema(value: Any, schema: dict[str, Any], path: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
    except Exception as error:  # pragma: no cover - checked-in schema authoring failure
        raise CatalogBuildError(f"invalid JSON Schema for {path}: {error}") from error
    errors = sorted(
        validator.iter_errors(value), key=lambda item: list(item.absolute_path)
    )
    if not errors:
        return
    error = errors[0]
    suffix = "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise CatalogBuildError(f"{path}{suffix}: {error.message}")


def load_yaml(path: Path) -> Any:
    try:
        return _plain_yaml_value(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as error:
        raise CatalogBuildError(
            f"cannot load {path.relative_to(REPO_ROOT)}: {error}"
        ) from error


def _plain_yaml_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain_yaml_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_yaml_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
