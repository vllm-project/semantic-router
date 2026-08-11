"""JSON Schema validation for versioned recipe probe manifests."""

from __future__ import annotations

import json
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
