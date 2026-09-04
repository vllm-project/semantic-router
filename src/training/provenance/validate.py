from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenance.digest import tree_digest, tree_files
from provenance.manifests import (
    SCHEMA_VERSION,
    ArtifactManifest,
    DatasetManifest,
    EvaluationManifest,
    Manifest,
    RunManifest,
    load_manifest,
    manifest_id,
)

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": ("name", "sources", "splits"),
    "run": ("name", "task", "dataset_id", "base_model", "code_revision", "seed"),
    "evaluation": (
        "run_id",
        "dataset_id",
        "split",
        "sample_count",
        "command",
        "metrics",
    ),
    "artifact": ("run_id", "evaluation_id", "format", "files", "digest"),
}
REQUIRED_SOURCE_KEYS = ("name", "revision", "license")
REQUIRED_BASE_MODEL_KEYS = ("name", "revision")
SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|api_key|apikey|credential)", re.I
)
SECRET_VALUE_PATTERN = re.compile(
    r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"
)
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(DIGEST_PATTERN.match(value))


def _is_str_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _is_count_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and _is_int(v) and v >= 0 for k, v in value.items()
    )


def _is_int_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and _is_int(v) for k, v in value.items()
    )


def _is_metric_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and _is_number(v) for k, v in value.items()
    )


def _is_digest_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and _is_digest(v) for k, v in value.items()
    )


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_dict_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, dict) for v in value)


FIELD_CHECKS: dict[str, dict[str, tuple[Callable[[Any], bool], str]]] = {
    "dataset": {
        "sources": (_is_dict_list, "a list of objects"),
        "splits": (_is_count_map, "a map of split name to non-negative int"),
        "preprocessing": (_is_str_list, "a list of strings"),
        "label_mapping": (_is_int_map, "a map of label to int"),
    },
    "run": {
        "dataset_id": (_is_digest, "a sha256:<64 hex> identity"),
        "base_model": (_is_str_map, "a map of string to string"),
        "dependencies": (_is_str_map, "a map of string to string"),
        "seed": (lambda v: _is_int(v) and v >= 0, "a non-negative int"),
        "hyperparameters": (lambda v: isinstance(v, dict), "an object"),
    },
    "evaluation": {
        "run_id": (_is_digest, "a sha256:<64 hex> identity"),
        "dataset_id": (_is_digest, "a sha256:<64 hex> identity"),
        "sample_count": (lambda v: _is_int(v) and v > 0, "a positive int"),
        "metrics": (_is_metric_map, "a map of metric to finite number"),
    },
    "artifact": {
        "run_id": (_is_digest, "a sha256:<64 hex> identity"),
        "evaluation_id": (_is_digest, "a sha256:<64 hex> identity"),
        "files": (_is_digest_map, "a map of path to sha256:<64 hex> digest"),
        "digest": (_is_digest, "a sha256:<64 hex> digest"),
        "label_mapping": (_is_int_map, "a map of label to int"),
    },
}
STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": ("name",),
    "run": ("name", "task", "code_revision"),
    "evaluation": ("name", "split", "command"),
    "artifact": ("name", "format"),
}
BUNDLE_FILES = {
    "dataset": "dataset.json",
    "run": "run.json",
    "evaluation": "evaluation.json",
    "artifact": "artifact.json",
}


class ProvenanceError(ValueError):
    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass
class Bundle:
    dataset: DatasetManifest
    run: RunManifest
    evaluation: EvaluationManifest
    artifact: ArtifactManifest


def _is_missing(value: Any) -> bool:
    return value is None or value in ("", {}, [])


def _completeness_errors(manifest: Manifest) -> list[str]:
    errors = []
    if manifest.schema_version != SCHEMA_VERSION:
        errors.append(
            f"{manifest.kind}: unsupported schema_version {manifest.schema_version}"
        )
    for name in REQUIRED_FIELDS[manifest.kind]:
        if _is_missing(getattr(manifest, name)):
            errors.append(f"{manifest.kind}: missing required field {name!r}")
    return errors


def _type_errors(manifest: Manifest) -> list[str]:
    errors = []
    if not _is_int(manifest.schema_version):
        errors.append(f"{manifest.kind}: schema_version must be an int")
    if not isinstance(manifest.extra, dict):
        errors.append(f"{manifest.kind}: extra must be an object")
    for name in STRING_FIELDS[manifest.kind]:
        if not isinstance(getattr(manifest, name), str):
            errors.append(f"{manifest.kind}: {name} must be a string")
    for name, (check, expected) in FIELD_CHECKS[manifest.kind].items():
        value = getattr(manifest, name)
        if not _is_missing(value) and not check(value):
            errors.append(f"{manifest.kind}: {name} must be {expected}")
    return errors


def _well_typed(manifest: Manifest, name: str) -> bool:
    check, _ = FIELD_CHECKS[manifest.kind][name]
    return check(getattr(manifest, name))


def _source_errors(sources: list[dict[str, Any]]) -> list[str]:
    errors = []
    for index, source in enumerate(sources):
        for key in REQUIRED_SOURCE_KEYS:
            value = source.get(key)
            if _is_missing(value):
                errors.append(f"dataset: sources[{index}] missing {key!r}")
            elif not isinstance(value, str):
                errors.append(f"dataset: sources[{index}].{key} must be a string")
    return errors


def _nested_errors(manifest: Manifest) -> list[str]:
    errors = []
    if isinstance(manifest, DatasetManifest) and _well_typed(manifest, "sources"):
        errors.extend(_source_errors(manifest.sources))
    if isinstance(manifest, RunManifest) and _well_typed(manifest, "base_model"):
        for key in REQUIRED_BASE_MODEL_KEYS:
            if _is_missing(manifest.base_model.get(key)):
                errors.append(f"run: base_model missing {key!r}")
    if isinstance(manifest, ArtifactManifest) and _well_typed(manifest, "files"):
        errors.extend(_artifact_path_errors(manifest))
    return errors


def _artifact_path_errors(manifest: ArtifactManifest) -> list[str]:
    errors = []
    for name in manifest.files:
        if name.startswith("/") or ".." in Path(name).parts:
            errors.append(
                f"artifact: file path must be relative and inside the artifact: {name!r}"
            )
    if manifest.files and manifest.digest != tree_digest(manifest.files):
        errors.append("artifact: digest does not match files")
    return errors


def _secret_errors(kind: str, value: Any, path: str = "") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY_PATTERN.search(str(key)):
                errors.append(f"{kind}: secret-like key {path}{key!r}")
            errors.extend(_secret_errors(kind, item, f"{path}{key}."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_secret_errors(kind, item, f"{path}[{index}]."))
    elif isinstance(value, str) and SECRET_VALUE_PATTERN.search(value):
        errors.append(f"{kind}: secret-like value at {path.rstrip('.')}")
    return errors


def manifest_errors(manifest: Manifest) -> list[str]:
    return (
        _completeness_errors(manifest)
        + _type_errors(manifest)
        + _nested_errors(manifest)
        + _secret_errors(manifest.kind, manifest.to_dict())
    )


def validate_manifest(manifest: Manifest) -> None:
    errors = manifest_errors(manifest)
    if errors:
        raise ProvenanceError(errors)


def _cross_reference_errors(bundle: Bundle) -> list[str]:
    dataset_id = manifest_id(bundle.dataset)
    run_id = manifest_id(bundle.run)
    evaluation_id = manifest_id(bundle.evaluation)
    checks = (
        ("run.dataset_id", bundle.run.dataset_id, dataset_id),
        ("evaluation.run_id", bundle.evaluation.run_id, run_id),
        ("evaluation.dataset_id", bundle.evaluation.dataset_id, dataset_id),
        ("artifact.run_id", bundle.artifact.run_id, run_id),
        ("artifact.evaluation_id", bundle.artifact.evaluation_id, evaluation_id),
    )
    return [
        f"{field}: expected {expected} got {actual or '<empty>'}"
        for field, actual, expected in checks
        if actual != expected
    ]


def _artifact_dir_errors(
    artifact: ArtifactManifest, artifact_dir: Path | None
) -> list[str]:
    if artifact_dir is None:
        return []
    actual = tree_files(artifact_dir)
    if actual != artifact.files:
        changed = sorted(set(actual) ^ set(artifact.files))
        changed += sorted(
            k
            for k in actual.keys() & artifact.files.keys()
            if actual[k] != artifact.files[k]
        )
        return [f"artifact: files differ from {artifact_dir}: {', '.join(changed)}"]
    return []


def bundle_errors(bundle: Bundle, artifact_dir: Path | None = None) -> list[str]:
    errors = []
    for manifest in (bundle.dataset, bundle.run, bundle.evaluation, bundle.artifact):
        errors.extend(manifest_errors(manifest))
    errors.extend(_cross_reference_errors(bundle))
    errors.extend(_artifact_dir_errors(bundle.artifact, artifact_dir))
    return errors


def validate_bundle(bundle: Bundle, artifact_dir: Path | None = None) -> None:
    errors = bundle_errors(bundle, artifact_dir)
    if errors:
        raise ProvenanceError(errors)


def load_bundle(directory: Path) -> Bundle:
    loaded = {
        kind: load_manifest(directory / name) for kind, name in BUNDLE_FILES.items()
    }
    for kind, manifest in loaded.items():
        if manifest.kind != kind:
            raise ProvenanceError(
                [f"{BUNDLE_FILES[kind]}: expected kind {kind!r} got {manifest.kind!r}"]
            )
    return Bundle(**loaded)
