"""Content identity of an immutable local model initialization source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import file_sha256

SOURCE_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}


def source_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"Model source does not exist: {path}")
    # The Qwen base loader reads root files; Decision 1.0/2.0 also read the
    # backbone/ subtree. Ignore model-card assets, metrics, HF cache, and other
    # unrelated directories that may change during a long experiment.
    candidates = list(path.iterdir())
    if (path / "backbone").is_dir():
        candidates.extend((path / "backbone").rglob("*"))
    files = [
        file
        for file in candidates
        if file.is_file()
        and file.suffix in SOURCE_SUFFIXES
        and file.name not in {"trainer_state.pt", "checkpoint.json"}
    ]
    if not any(file.suffix in {".safetensors", ".bin"} for file in files):
        raise ValueError("Model source contains no local weight files")
    return {
        "source_name": path.name,
        "files_sha256": {
            str(file.relative_to(path)): file_sha256(file) for file in sorted(files)
        },
    }


def verify_source(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(expected, dict) or not isinstance(
        expected.get("files_sha256"), dict
    ):
        raise ValueError("LoRA checkpoint has no valid source fingerprint")
    actual = source_fingerprint(path)
    if actual["files_sha256"] != expected["files_sha256"]:
        raise ValueError(
            "Local source model files differ from the LoRA checkpoint fingerprint"
        )
    return actual
