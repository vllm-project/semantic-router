"""Small release fixtures exercise the same hash checks as full model files."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.release_artifacts import (  # noqa: E402
    ReleaseArtifactError,
    verify_release_manifest,
)


def _release(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "release"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"verified model bytes")
    body = (root / "model.safetensors").read_bytes()
    manifest = json.dumps(
        {
            "files": {
                "model.safetensors": {
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            }
        }
    ).encode()
    (root / "MANIFEST.json").write_bytes(manifest)
    return root, hashlib.sha256(manifest).hexdigest()


def test_pinned_manifest_verifies_file_bytes(tmp_path: Path) -> None:
    root, digest = _release(tmp_path)
    verify_release_manifest(
        root,
        manifest_name="MANIFEST.json",
        expected_sha256=digest,
        required_files=("model.safetensors",),
    )

    (root / "model.safetensors").write_bytes(b"tampered model bytes")
    with pytest.raises(ReleaseArtifactError, match="digest mismatch"):
        verify_release_manifest(
            root,
            manifest_name="MANIFEST.json",
            expected_sha256=digest,
            required_files=("model.safetensors",),
        )


def test_manifest_identity_and_missing_file_fail_closed(tmp_path: Path) -> None:
    root, digest = _release(tmp_path)
    with pytest.raises(ReleaseArtifactError, match="manifest digest mismatch"):
        verify_release_manifest(
            root,
            manifest_name="MANIFEST.json",
            expected_sha256="0" * 64,
            required_files=("model.safetensors",),
        )
    with pytest.raises(ReleaseArtifactError, match="missing tokenizer.json"):
        verify_release_manifest(
            root,
            manifest_name="MANIFEST.json",
            expected_sha256=digest,
            required_files=("tokenizer.json",),
        )
