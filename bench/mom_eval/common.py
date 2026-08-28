"""Shared helpers for MoM evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "config" / "evaluation" / "schema"
CORE_SUITE_MANIFEST = REPO_ROOT / "config" / "evaluation" / "mom-core-suite" / "v1" / "manifest.yaml"
PACK_REGISTRY = REPO_ROOT / "config" / "evaluation" / "packs" / "registry.yaml"
SCORECARD_INDEX = REPO_ROOT / "config" / "evaluation" / "scorecards" / "index.yaml"


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def recipe_digest(recipe_dir: Path) -> str:
    hasher = hashlib.sha256()
    for name in sorted(("config.yaml", "metadata.yaml", "recipe.dsl", "mom-evaluation.yaml")):
        path = recipe_dir / name
        if path.is_file():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def resolve_manifest_path(manifest_arg: str | Path) -> Path:
    path = Path(manifest_arg)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_mom_manifest(manifest_path: Path) -> dict[str, Any]:
    return load_yaml(manifest_path)


def entrypoint_config(manifest: dict[str, Any], entrypoint: str) -> dict[str, Any]:
    entrypoints = manifest.get("entrypoints") or {}
    if entrypoint not in entrypoints:
        known = ", ".join(sorted(entrypoints))
        raise KeyError(f"entrypoint {entrypoint!r} not found; known: {known}")
    return entrypoints[entrypoint]


def load_core_suite() -> dict[str, Any]:
    return load_yaml(CORE_SUITE_MANIFEST)


def load_pack_registry() -> dict[str, Any]:
    return load_yaml(PACK_REGISTRY)
