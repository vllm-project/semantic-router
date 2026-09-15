"""JSON Schema validation for managed recipe metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "config" / "schemas" / "recipe-metadata-v1.schema.json"
METADATA_SCHEMA_VERSION = "vllm-sr/recipe-metadata/v1"
MAX_SCHEMA_ERRORS = 10


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
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


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_recipe_metadata_document(document: str) -> dict[str, Any]:
    metadata = yaml.load(document, Loader=UniqueKeyLoader) or {}
    if not isinstance(metadata, dict):
        raise TypeError("recipe metadata must contain a YAML mapping")
    return metadata


@lru_cache(maxsize=1)
def recipe_metadata_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_recipe_metadata_schema(
    metadata: dict[str, Any], metadata_path: Path
) -> None:
    errors = sorted(
        recipe_metadata_validator().iter_errors(metadata),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    details = []
    for error in errors[:MAX_SCHEMA_ERRORS]:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    if len(errors) > MAX_SCHEMA_ERRORS:
        details.append(f"... and {len(errors) - MAX_SCHEMA_ERRORS} more errors")
    raise ValueError(
        f"{metadata_path} does not satisfy recipe metadata schema: "
        + "; ".join(details)
    )
