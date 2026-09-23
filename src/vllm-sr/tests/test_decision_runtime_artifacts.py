"""Security and integrity tests for Decision artifact materialization."""

from __future__ import annotations

import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from decision_runtime.artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactManifestError,
    ArtifactResolver,
    open_verified_artifact,
    parse_artifact_manifest,
)
from decision_runtime.catalog_adapter import (
    ResolvedRuntimeModel,
    resolve_decision_runtime_model,
)
from decision_runtime.runtime_profile import (
    ArtifactManifestIdentity,
    ArtifactSelection,
)

MODEL_ID = "llm-semantic-router/Decision-1.0-Kai-0.6B"


@dataclass
class FakeFetcher:
    files: dict[str, Path]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, *, repository_id: str, revision: str, filename: str) -> Path:
        self.calls.append((repository_id, revision, filename))
        try:
            return self.files[filename]
        except KeyError as error:
            raise AssertionError(f"unexpected fetch: {filename}") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture_model(
    tmp_path: Path,
    *,
    payloads: dict[str, bytes] | None = None,
    selected: tuple[str, ...] = ("config.json", "weights/model.safetensors"),
) -> tuple[ResolvedRuntimeModel, FakeFetcher]:
    payloads = payloads or {
        "config.json": b'{"hidden_size": 8}\n',
        "weights/model.safetensors": b"safe tensor bytes",
    }
    source = tmp_path / "source"
    manifest_document = {
        "files": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in payloads.items()
        }
    }
    manifest_payload = json.dumps(
        manifest_document, sort_keys=True, separators=(",", ":")
    ).encode()
    repository_files: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = source / "artifact" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        repository_files[f"artifact/{name}"] = path
    manifest_path = source / "artifact" / "MANIFEST.json"
    manifest_path.write_bytes(manifest_payload)
    repository_files["artifact/MANIFEST.json"] = manifest_path

    base = resolve_decision_runtime_model(MODEL_ID)
    artifact = ArtifactSelection(
        manifest=ArtifactManifestIdentity(
            path="artifact/MANIFEST.json",
            sha256=_sha256(manifest_payload),
            size_bytes=len(manifest_payload),
        ),
        files=selected,
    )
    model = replace(base, profile=replace(base.profile, artifact=artifact))
    return model, FakeFetcher(repository_files)


def test_materialize_verifies_exact_revision_and_creates_read_only_view(
    tmp_path: Path,
) -> None:
    model, fetcher = _fixture_model(tmp_path)
    resolver = ArtifactResolver(fetcher=fetcher, cache_root=tmp_path / "cache")

    artifact = resolver.materialize(model)

    assert artifact.repository_id == MODEL_ID
    assert artifact.revision == model.catalog.revision
    assert len(artifact.content_id) == 64
    assert artifact.data_root == artifact.root / "artifact"
    assert (artifact.root / "artifact/config.json").read_text() == (
        '{"hidden_size": 8}\n'
    )
    assert (artifact.root / "artifact/weights/model.safetensors").read_bytes() == (
        b"safe tensor bytes"
    )
    assert all(call[0] == MODEL_ID for call in fetcher.calls)
    assert all(call[1] == model.catalog.revision for call in fetcher.calls)
    assert all(
        not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        for path in (artifact.root, *artifact.root.rglob("*"))
    )

    required_file_fetches = len(fetcher.calls)
    assert resolver.materialize(model).root == artifact.root
    assert len(fetcher.calls) == required_file_fetches + 1


def test_open_verified_artifact_rechecks_mounted_content_without_fetching(
    tmp_path: Path,
) -> None:
    model, fetcher = _fixture_model(tmp_path)
    artifact = ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)
    fetch_count = len(fetcher.calls)

    reopened = open_verified_artifact(
        artifact.root,
        model,
        expected_content_id=artifact.content_id,
    )

    assert reopened == artifact
    assert len(fetcher.calls) == fetch_count


def test_open_verified_artifact_rejects_wrong_identity_and_corruption(
    tmp_path: Path,
) -> None:
    model, fetcher = _fixture_model(tmp_path)
    artifact = ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)

    with pytest.raises(ArtifactIntegrityError, match="launch contract"):
        open_verified_artifact(
            artifact.root,
            model,
            expected_content_id="0" * 64,
        )

    selected = artifact.root / "artifact/config.json"
    selected.chmod(0o644)
    selected.write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match="size|digest"):
        open_verified_artifact(
            artifact.root,
            model,
            expected_content_id=artifact.content_id,
        )


def test_concurrent_materialization_converges_on_one_complete_view(
    tmp_path: Path,
) -> None:
    model, fetcher = _fixture_model(tmp_path)
    resolver = ArtifactResolver(fetcher=fetcher, cache_root=tmp_path / "cache")

    with ThreadPoolExecutor(max_workers=8) as executor:
        artifacts = list(executor.map(lambda _: resolver.materialize(model), range(16)))

    assert len({artifact.root for artifact in artifacts}) == 1
    artifact = artifacts[0]
    assert (artifact.root / "artifact/config.json").is_file()
    assert (artifact.root / "artifact/weights/model.safetensors").is_file()
    assert (artifact.root / ".vllm-sr-artifact.json").is_file()


def test_materialization_detects_cached_corruption(tmp_path: Path) -> None:
    model, fetcher = _fixture_model(tmp_path)
    resolver = ArtifactResolver(fetcher=fetcher, cache_root=tmp_path / "cache")
    artifact = resolver.materialize(model)
    corrupted = artifact.root / "artifact/config.json"
    corrupted.chmod(0o644)
    corrupted.write_bytes(b"corrupted")

    with pytest.raises(ArtifactIntegrityError, match="size|digest"):
        resolver.materialize(model)


def test_fetched_file_corruption_is_rejected(tmp_path: Path) -> None:
    model, fetcher = _fixture_model(tmp_path)
    fetcher.files["artifact/config.json"].write_bytes(b"wrong")

    with pytest.raises(ArtifactIntegrityError, match="size|digest"):
        ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)


@pytest.mark.parametrize(
    "unsafe",
    ("../secret", "/absolute", "nested/../../secret", "nested\\secret", "C:/x"),
)
def test_manifest_rejects_unsafe_paths(unsafe: str) -> None:
    payload = json.dumps(
        {"files": {unsafe: {"bytes": 0, "sha256": _sha256(b"")}}}
    ).encode()

    with pytest.raises(ArtifactManifestError, match="unsafe path"):
        parse_artifact_manifest(payload, manifest_path="MANIFEST.json")


def test_manifest_list_shape_is_supported_and_strict() -> None:
    payload = json.dumps(
        {"files": [{"file": "model.safetensors", "bytes": 1, "sha256": _sha256(b"x")}]}
    ).encode()

    inventory = parse_artifact_manifest(payload, manifest_path="bundle-manifest.json")
    assert inventory["model.safetensors"].repository_path == "model.safetensors"

    payload = json.dumps(
        {
            "files": [
                {
                    "file": "model.safetensors",
                    "bytes": 1,
                    "sha256": _sha256(b"x"),
                    "url": "https://example.invalid",
                }
            ]
        }
    ).encode()
    with pytest.raises(ArtifactManifestError, match="fields are invalid"):
        parse_artifact_manifest(payload, manifest_path="bundle-manifest.json")


def test_missing_selected_file_and_repository_code_are_rejected(tmp_path: Path) -> None:
    missing, missing_fetcher = _fixture_model(
        tmp_path / "missing", selected=("missing.safetensors",)
    )
    with pytest.raises(ArtifactManifestError, match="absent"):
        ArtifactResolver(missing_fetcher, tmp_path / "missing-cache").materialize(
            missing
        )

    code, code_fetcher = _fixture_model(
        tmp_path / "code",
        payloads={"runtime.py": b"raise RuntimeError('must never run')"},
        selected=("runtime.py",),
    )
    with pytest.raises(ArtifactManifestError, match="may not select repository code"):
        ArtifactResolver(code_fetcher, tmp_path / "code-cache").materialize(code)


def test_manifest_identity_is_verified_before_json_is_parsed(tmp_path: Path) -> None:
    model, fetcher = _fixture_model(tmp_path)
    manifest = model.profile.artifact.manifest
    model = replace(
        model,
        profile=replace(
            model.profile,
            artifact=replace(
                model.profile.artifact,
                manifest=replace(manifest, sha256="0" * 64),
            ),
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="digest"):
        ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)


def test_materializer_revalidates_catalog_identity(tmp_path: Path) -> None:
    model, fetcher = _fixture_model(tmp_path)

    with pytest.raises(ArtifactError, match="does not match"):
        ArtifactResolver(fetcher, tmp_path / "cache").materialize(
            replace(model, repository_id="llm-semantic-router/Other")
        )

    with pytest.raises(ArtifactError, match="full lowercase Git SHA"):
        ArtifactResolver(fetcher, tmp_path / "cache").materialize(
            replace(model, catalog=replace(model.catalog, revision="main"))
        )
