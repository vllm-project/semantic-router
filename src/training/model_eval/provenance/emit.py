"""Builders that turn a live training or evaluation workflow into manifests.

Nothing here invents provenance. Every value is read from the running process,
the resolved Hugging Face revision, or the artifact bytes on disk. When a fact
is unavailable the builder raises instead of writing a placeholder, because a
manifest that guesses is worse than no manifest.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

from .crossref import artifact_identity_digest
from .manifest import load_manifest

__all__ = [
    "ProvenanceError",
    "build_artifact_manifest",
    "build_dataset_manifest",
    "build_evaluation_manifest",
    "build_run_manifest",
    "code_revision",
    "dependency_versions",
    "file_digest",
    "resolve_hf_revision",
    "split_digest",
    "write_manifest",
]

CODE_REPO = "vllm-project/semantic-router"
DIGEST_CHUNK_BYTES = 1024 * 1024
# Length of a full git commit sha. Anything shorter is a prefix or a branch name.
REVISION_LENGTH = 40
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
    "peft",
    "numpy",
    "scikit-learn",
)


class ProvenanceError(RuntimeError):
    """Raised when a required provenance fact cannot be established."""


def resolve_hf_revision(repo_id: str, repo_type: str = "model") -> str:
    """Resolve a Hugging Face repo to the immutable commit sha it currently points at."""
    api = HfApi()
    if repo_type == "dataset":
        info = api.dataset_info(repo_id)
    else:
        info = api.model_info(repo_id)
    sha = str(getattr(info, "sha", "") or "")
    if len(sha) != REVISION_LENGTH:
        raise ProvenanceError(
            f"{repo_id} did not resolve to a 40-character commit sha (got {sha!r})"
        )
    return sha


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(DIGEST_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def split_digest(rows: Iterable[tuple[str, int]]) -> str:
    """Digest a split from its (text, label) content, independent of row order.

    Ordering is deliberately removed so that a shuffle is not reported as a
    different dataset, while any change to text or labels is.
    """
    hasher = hashlib.sha256()
    for row_digest in sorted(
        hashlib.sha256(f"{label}\0{text}".encode()).hexdigest() for text, label in rows
    ):
        hasher.update(row_digest.encode("ascii"))
    return f"sha256:{hasher.hexdigest()}"


def dependency_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, Any]:
    resolved: dict[str, str] = {}
    for name in packages:
        try:
            resolved[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if not resolved:
        raise ProvenanceError("no tracked package versions could be resolved")
    return {"python": platform.python_version(), "packages": resolved}


def code_revision(entrypoint: str, repo_root: Path) -> dict[str, Any]:
    """Record the commit the workflow ran from, and whether the tree was dirty."""
    revision = _git(["rev-parse", "HEAD"], repo_root)
    if revision is None or len(revision) != REVISION_LENGTH:
        raise ProvenanceError(
            f"{repo_root} is not a git checkout; provenance requires a code revision"
        )
    status = _git(["status", "--porcelain"], repo_root)
    return {
        "repo": CODE_REPO,
        "revision": revision,
        "entrypoint": entrypoint,
        "dirty": bool(status),
    }


def build_dataset_manifest(
    *,
    manifest_id: str,
    task: str,
    source_type: str,
    locator: str,
    revision: str,
    license_id: str,
    splits: list[dict[str, Any]],
    text_field: str,
    label_field: str,
    preprocessing_steps: list[str],
    label_mapping: dict[str, int],
    description: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "v1",
        "kind": "dataset",
        "id": manifest_id,
        "task": task,
        "source": {
            "type": source_type,
            "locator": locator,
            "revision": revision,
        },
        "license": license_id,
        "splits": splits,
        "preprocessing": {
            "text_field": text_field,
            "label_field": label_field,
            "steps": preprocessing_steps,
        },
        "label_mapping": label_mapping,
    }
    if description:
        manifest["description"] = description
    return manifest


def build_run_manifest(
    *,
    manifest_id: str,
    task: str,
    base_model_repo: str,
    base_model_revision: str,
    entrypoint: str,
    repo_root: Path,
    dataset_refs: list[dict[str, Any]],
    seed: int,
    hyperparameters: dict[str, Any],
    label_mapping: dict[str, int],
    description: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "v1",
        "kind": "run",
        "id": manifest_id,
        "task": task,
        "base_model": {"repo": base_model_repo, "revision": base_model_revision},
        "code": code_revision(entrypoint, repo_root),
        "dataset_refs": dataset_refs,
        "seed": seed,
        "hyperparameters": _scalarize(hyperparameters),
        "dependencies": dependency_versions(),
        "label_mapping": label_mapping,
    }
    if description:
        manifest["description"] = description
    return manifest


def build_artifact_manifest(
    *,
    manifest_id: str,
    task: str,
    repo: str,
    revision: str,
    artifact_dir: Path,
    label_mapping: dict[str, int],
    architecture: str,
    max_position_embeddings: int,
    run_id: str | None = None,
    torch_dtype: str | None = None,
    tokenizer_class: str | None = None,
    served_paths: Sequence[str] = (),
    include_globs: Sequence[str] = (
        "*.json",
        "*.safetensors",
        "*.bin",
        "*.txt",
        "*.model",
    ),
    description: str | None = None,
) -> dict[str, Any]:
    """Hash the artifact on disk into a manifest whose identity is reproducible."""
    artifact_dir = Path(artifact_dir)
    files = []
    seen: set[str] = set()
    for pattern in include_globs:
        for path in sorted(artifact_dir.rglob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(artifact_dir).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "digest": file_digest(path),
                }
            )
    if not files:
        raise ProvenanceError(f"{artifact_dir} contains no artifact files to hash")
    files.sort(key=lambda entry: entry["path"])

    runtime: dict[str, Any] = {
        "architecture": architecture,
        "max_position_embeddings": max_position_embeddings,
        "num_labels": len(label_mapping),
    }
    if torch_dtype:
        runtime["torch_dtype"] = torch_dtype
    if tokenizer_class:
        runtime["tokenizer_class"] = tokenizer_class
    if served_paths:
        runtime["served_paths"] = sorted(set(served_paths))

    manifest = {
        "schema_version": "v1",
        "kind": "artifact",
        "id": manifest_id,
        "task": task,
        "identity": {
            "repo": repo,
            "revision": revision,
            "digest": artifact_identity_digest(files),
        },
        "files": files,
        "label_mapping": label_mapping,
        "runtime": runtime,
    }
    if run_id:
        manifest["run_ref"] = {"id": run_id}
    if description:
        manifest["description"] = description
    return manifest


def build_evaluation_manifest(
    *,
    manifest_id: str,
    task: str,
    artifact_ref: dict[str, Any],
    dataset_ref: dict[str, Any],
    entrypoint: str,
    repo_root: Path,
    device: str,
    device_name: str | None,
    batch_size: int,
    max_length: int,
    sample_limit: int | None,
    seed: int,
    label_mapping: dict[str, int],
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    abstention: dict[str, Any],
    performance: dict[str, Any],
    slices: list[dict[str, Any]] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    harness: dict[str, Any] = {
        "code": code_revision(entrypoint, repo_root),
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "sample_limit": sample_limit,
        "seed": seed,
        "dependencies": dependency_versions(),
    }
    if device_name:
        harness["device_name"] = device_name

    manifest = {
        "schema_version": "v1",
        "kind": "evaluation",
        "id": manifest_id,
        "task": task,
        "artifact_ref": artifact_ref,
        "dataset_ref": dataset_ref,
        "harness": harness,
        "label_mapping": label_mapping,
        "metrics": metrics,
        "calibration": calibration,
        "abstention": abstention,
        "performance": performance,
    }
    if slices:
        manifest["slices"] = slices
    if description:
        manifest["description"] = description
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write a manifest as plain YAML and re-validate it before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            _plain_tree(manifest),
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    load_manifest(path, expected_kind=manifest["kind"])
    return path


def _scalarize(values: dict[str, Any]) -> dict[str, Any]:
    """Flatten hyperparameters to the scalar types the schema accepts."""
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            flattened[str(key)] = ",".join(str(_plain(item)) for item in value)
        else:
            flattened[str(key)] = _plain(value)
    return flattened


def _plain(value: Any) -> Any:
    """Reduce a value to a plain YAML-serialisable scalar.

    Library types such as ``transformers.SchedulerType`` subclass ``str`` and
    ``Enum``, which pass an ``isinstance`` check but cannot be serialised. They
    are reduced here so a manifest never fails to write over a type detail.
    """
    if isinstance(value, Enum):
        return _plain(value.value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    for plain_type in (bool, int, float):
        if isinstance(value, plain_type):
            return plain_type(value)
    return str(value)


def _plain_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(_plain(key)): _plain_tree(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain_tree(item) for item in node]
    return _plain(node)


def _git(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
