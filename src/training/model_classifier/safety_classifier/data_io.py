"""Dependency-light source I/O and contract validation for safety data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

if __package__:
    from .data_contract import (
        NORMALIZATION_VERSION,
        SHA256_HEX_LENGTH,
        SPLIT_PRECEDENCE,
        SPLITS,
        DataContractError,
        DataPreparationError,
        SchemaError,
    )
    from .taxonomy import LEVEL1_LABEL_TO_ID, LEVEL2_LABEL_TO_ID, TAXONOMY_VERSION
else:  # Allow importing from a directly executed sibling module.
    from data_contract import (  # type: ignore[no-redef]
        NORMALIZATION_VERSION,
        SHA256_HEX_LENGTH,
        SPLIT_PRECEDENCE,
        SPLITS,
        DataContractError,
        DataPreparationError,
        SchemaError,
    )
    from taxonomy import (  # type: ignore[no-redef]
        LEVEL1_LABEL_TO_ID,
        LEVEL2_LABEL_TO_ID,
        TAXONOMY_VERSION,
    )


def load_json_array(path: str | Path) -> list[Mapping[str, object]]:
    """Load a JSON array without importing a dataset framework."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise SchemaError(f"{source} must contain a JSON array of objects")
    return value


def load_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    """Load JSONL records without importing a dataset framework."""
    source = Path(path)
    rows: list[Mapping[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"Invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise SchemaError(f"{source}:{line_number} must contain an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """Atomically replace ``path`` with ``payload`` on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def materialize_dataset(
    build: Any,
    output_dir: str | Path,
    *,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Write split JSONL files and a self-auditing deterministic manifest."""
    target = Path(output_dir)
    files: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        rows = build.split_samples(split)
        payload = b"".join(
            _canonical_json_bytes(sample.to_dict()) + b"\n" for sample in rows
        )
        filename = f"{split}.jsonl"
        atomic_write(target / filename, payload)
        files[split] = {
            "path": filename,
            "rows": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest = json.loads(json.dumps(build.manifest))
    manifest["files"] = files
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    atomic_write(
        target / "data_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )
    return manifest


def validate_data_contract(contract: Mapping[str, object]) -> None:
    """Validate data and taxonomy fields consumed by the materializer."""
    if contract.get("taxonomy_version") != TAXONOMY_VERSION:
        raise DataContractError(
            f"Contract taxonomy_version must be {TAXONOMY_VERSION!r}"
        )
    data = contract.get("data")
    if not isinstance(data, Mapping):
        raise DataContractError("Contract data must be an object")
    expected = {
        "input_field": "prompt",
        "normalization": NORMALIZATION_VERSION,
        "split_precedence": list(SPLIT_PRECEDENCE),
        "single_label_strategy": "first-mapped-source-order",
        "drop_redacted": True,
        "use_refusal_splits": False,
        "use_responses": False,
    }
    for key, expected_value in expected.items():
        if data.get(key) != expected_value:
            raise DataContractError(
                f"Contract data.{key} must be {expected_value!r}, got {data.get(key)!r}"
            )

    tasks = contract.get("tasks")
    if not isinstance(tasks, Mapping):
        raise DataContractError("Contract tasks must be an object")
    expected_labels = {
        "level1": LEVEL1_LABEL_TO_ID,
        "level2": LEVEL2_LABEL_TO_ID,
    }
    for task, label2id in expected_labels.items():
        task_contract = tasks.get(task)
        if not isinstance(task_contract, Mapping):
            raise DataContractError(f"Contract tasks.{task} must be an object")
        if task_contract.get("label2id") != label2id:
            raise DataContractError(
                f"Contract tasks.{task}.label2id disagrees with {TAXONOMY_VERSION}"
            )


def positive_contract_int(data: Mapping[str, object], key: str) -> int:
    """Read one positive integer from the data contract."""
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DataContractError(f"Contract data.{key} must be a positive integer")
    return value


def download_contract_file(
    dataset: Mapping[str, object],
    file_spec: Mapping[str, object],
    downloader: Callable[..., str],
) -> Path:
    """Download one pinned source file and verify its content digest."""
    repo_id = dataset.get("id")
    revision = dataset.get("revision")
    filename = file_spec.get("path")
    expected_sha256 = file_spec.get("sha256")
    if not all(isinstance(value, str) and value for value in (repo_id, revision)):
        raise DataContractError("Dataset id and revision must be non-empty strings")
    if not isinstance(filename, str) or not filename:
        raise DataContractError("Dataset file path must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != SHA256_HEX_LENGTH
    ):
        raise DataContractError("Dataset file sha256 must be a 64-character string")

    downloaded = Path(
        downloader(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    actual_sha256 = sha256_file(downloaded)
    if actual_sha256 != expected_sha256:
        raise DataPreparationError(
            f"SHA-256 mismatch for {repo_id}/{filename}@{revision}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return downloaded
