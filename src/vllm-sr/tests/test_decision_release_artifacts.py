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
    with pytest.raises(ReleaseArtifactError, match=r"missing tokenizer\.json"):
        verify_release_manifest(
            root,
            manifest_name="MANIFEST.json",
            expected_sha256=digest,
            required_files=("tokenizer.json",),
        )


def test_direct_qwen_loader_verifies_new_shards_before_importing_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decision_runtime import qwen35_torch  # noqa: PLC0415

    root = tmp_path / "qwen"
    root.mkdir()
    names = (
        *qwen35_torch.INFERENCE_FILES,
        "backbone/model.safetensors.index.json",
        "backbone/model-00001-of-00002.safetensors",
        "backbone/model-00002-of-00002.safetensors",
        "code/untrusted.py",
    )
    inventory = {}
    for name in names:
        payload = (
            b"raise AssertionError('repository code executed')"
            if name == "code/untrusted.py"
            else name.encode()
        )
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = json.dumps({"files": inventory}, sort_keys=True).encode()
    (root / "MODEL_MANIFEST.json").write_bytes(manifest)
    digest = hashlib.sha256(manifest).hexdigest()

    monkeypatch.setattr(qwen35_torch, "_validate_configuration", lambda *a, **k: None)
    monkeypatch.setattr(
        qwen35_torch,
        "_required_module",
        lambda name: (_ for _ in ()).throw(AssertionError("framework import reached")),
    )
    with pytest.raises(AssertionError, match="framework import reached"):
        qwen35_torch.Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=1024,
            backend="cpu",
            expected_manifest_sha256=digest,
        )

    (root / "backbone/model-00002-of-00002.safetensors").write_bytes(b"tampered")
    with pytest.raises(qwen35_torch.Qwen35RuntimeError, match="integrity failed"):
        qwen35_torch.Qwen35TorchRuntime.load(
            root,
            temperature=1.0,
            max_length=1024,
            backend="cpu",
            expected_manifest_sha256=digest,
        )
