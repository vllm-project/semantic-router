"""Loading and schema validation for Router Model provenance manifests.

Every manifest is a YAML mapping that declares ``schema_version`` and ``kind``.
Validation is deliberately strict: an unknown field, a mutable revision, or a
YAML construct that can hide data is a failure, not a warning. Provenance that
silently accepts partial input is not provenance.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from yaml.nodes import MappingNode, Node, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

from .redaction import assert_publishable

SCHEMA_VERSION = "v1"
MANIFEST_KINDS = ("dataset", "run", "artifact", "evaluation")
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
MAX_SCHEMA_ERRORS = 10
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class ManifestError(ValueError):
    """Raised when a manifest is unreadable, invalid, or internally inconsistent."""


@lru_cache(maxsize=1)
def _registry() -> Registry:
    """Resolve the relative ``$ref`` targets used across the schema files."""
    resources = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((path.name, Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@lru_cache(maxsize=len(MANIFEST_KINDS))
def _validator(kind: str) -> Draft202012Validator:
    path = SCHEMA_DIR / f"router-model-{kind}-{SCHEMA_VERSION}.schema.json"
    if not path.is_file():
        raise ManifestError(f"no schema registered for manifest kind {kind!r}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_registry())


def load_manifest(path: Path, expected_kind: str | None = None) -> dict[str, Any]:
    """Read one manifest and return it only if it fully satisfies the contract."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{path} could not be read: {exc}") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestError(
            f"{path} is {len(raw)} bytes, above the {MAX_MANIFEST_BYTES} byte limit"
        )

    document = raw.decode("utf-8")
    _reject_yaml_indirection(document, path)
    try:
        manifest = yaml.safe_load(document)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"{path} must contain a mapping")

    kind = _declared_kind(manifest, path)
    if expected_kind is not None and kind != expected_kind:
        raise ManifestError(
            f"{path} declares kind {kind!r}, expected {expected_kind!r}"
        )

    version = str(manifest.get("schema_version") or "").strip()
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path} schema_version must be {SCHEMA_VERSION!r}, got {version!r}"
        )

    assert_publishable(manifest, path)
    _validate_against_schema(manifest, kind, path)
    _validate_label_mapping(manifest, path)
    return manifest


def load_manifests(directory: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """Load every ``*.manifest.yaml`` under ``directory``, grouped by kind."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ManifestError(f"{directory} is not a directory")

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        kind: [] for kind in MANIFEST_KINDS
    }
    paths = sorted(directory.rglob("*.manifest.yaml"))
    if not paths:
        raise ManifestError(f"{directory} contains no *.manifest.yaml files")
    for path in paths:
        manifest = load_manifest(path)
        grouped[manifest["kind"]].append((path, manifest))
    return grouped


def _declared_kind(manifest: dict[str, Any], path: Path) -> str:
    kind = str(manifest.get("kind") or "").strip()
    if kind not in MANIFEST_KINDS:
        raise ManifestError(
            f"{path} kind must be one of {', '.join(MANIFEST_KINDS)}, got {kind!r}"
        )
    return kind


def _validate_against_schema(manifest: dict[str, Any], kind: str, path: Path) -> None:
    errors = sorted(
        _validator(kind).iter_errors(manifest),
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
    raise ManifestError(
        f"{path} does not satisfy the {kind} manifest schema: " + "; ".join(details)
    )


def _validate_label_mapping(manifest: dict[str, Any], path: Path) -> None:
    """Require a contiguous 0..n-1 index space so class order is unambiguous."""
    mapping = manifest.get("label_mapping")
    if not isinstance(mapping, dict):
        return
    indices = sorted(mapping.values())
    expected = list(range(len(mapping)))
    if indices != expected:
        raise ManifestError(
            f"{path} label_mapping must cover indices 0..{len(mapping) - 1} exactly "
            f"once, got {indices}"
        )


def _reject_yaml_indirection(document: str, path: Path) -> None:
    """Reject YAML features that let a manifest hide or duplicate provenance."""
    try:
        tokens = list(yaml.scan(document))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc
    for token in tokens:
        if isinstance(token, AnchorToken):
            raise ManifestError(f"{path} must not contain YAML anchors")
        if isinstance(token, AliasToken):
            raise ManifestError(f"{path} must not contain YAML aliases")
        if isinstance(token, TagToken):
            raise ManifestError(f"{path} must not contain explicit YAML tags")
    root = yaml.compose(document)
    if root is not None and _contains_yaml_merge(root):
        raise ManifestError(f"{path} must not contain YAML merge keys")


def _contains_yaml_merge(node: Node) -> bool:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                return True
            if _contains_yaml_merge(key) or _contains_yaml_merge(value):
                return True
    elif isinstance(node, SequenceNode):
        return any(_contains_yaml_merge(item) for item in node.value)
    return False
