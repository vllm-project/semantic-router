"""JSON Schema validation for versioned recipe probe manifests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "tools" / "agent" / "schemas" / "recipe-probes-v1.schema.json"
MAX_SCHEMA_ERRORS = 10


@lru_cache(maxsize=1)
def probe_manifest_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_probe_manifest_schema(
    manifest: dict[str, Any], manifest_path: Path
) -> None:
    _reject_raw_probe_image_sources(manifest, manifest_path)
    errors = sorted(
        probe_manifest_validator().iter_errors(manifest),
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
        f"{manifest_path} does not satisfy recipe probe schema: " + "; ".join(details)
    )


def _reject_raw_probe_image_sources(
    manifest: dict[str, Any], manifest_path: Path
) -> None:
    """Fail before jsonschema can echo a credential-bearing image source."""
    for location, item in _probe_content_entries(manifest):
        decision_index, variant_index, message_index, content_index = location
        label = (
            f"{manifest_path} decisions[{decision_index}].variants"
            f"[{variant_index}].messages[{message_index}].content[{content_index}]"
        )
        raw_type = item.get("type")
        item_type = str(raw_type or "").strip().lower()
        if item_type in {"image_url", "input_image"}:
            raise ValueError(f"{label} must use a declared image_fixture")
        if item_type == "image_fixture" and raw_type != "image_fixture":
            raise ValueError(f"{label}.type must be exactly 'image_fixture'")


def _probe_content_entries(
    manifest: dict[str, Any],
) -> Iterator[tuple[tuple[int, int, int, int], dict[str, Any]]]:
    for decision_index, variant_index, message_index, message in _message_entries(
        manifest
    ):
        for content_index, item in _mapping_entries(message.get("content")):
            yield (
                decision_index,
                variant_index,
                message_index,
                content_index,
            ), item


def _message_entries(
    manifest: dict[str, Any],
) -> Iterator[tuple[int, int, int, dict[str, Any]]]:
    for decision_index, variant_index, variant in _variant_entries(manifest):
        for message_index, message in _mapping_entries(variant.get("messages")):
            yield decision_index, variant_index, message_index, message


def _variant_entries(
    manifest: dict[str, Any],
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    for decision_index, decision in _mapping_entries(manifest.get("decisions")):
        for variant_index, variant in _mapping_entries(decision.get("variants")):
            yield decision_index, variant_index, variant


def _mapping_entries(value: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield indexed mappings while preserving schema locations."""
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if isinstance(item, dict):
            yield index, item
