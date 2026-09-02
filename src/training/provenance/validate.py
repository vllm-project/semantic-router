from __future__ import annotations

import re
from collections.abc import Iterable
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
    "evaluation": ("run_id", "dataset_id", "split", "metrics"),
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


def _nested_errors(manifest: Manifest) -> list[str]:
    errors = []
    if isinstance(manifest, DatasetManifest):
        for index, source in enumerate(manifest.sources):
            for key in REQUIRED_SOURCE_KEYS:
                if _is_missing(source.get(key)):
                    errors.append(f"dataset: sources[{index}] missing {key!r}")
    if isinstance(manifest, RunManifest):
        for key in REQUIRED_BASE_MODEL_KEYS:
            if _is_missing(manifest.base_model.get(key)):
                errors.append(f"run: base_model missing {key!r}")
    if isinstance(manifest, ArtifactManifest):
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
