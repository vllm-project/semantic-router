"""Verified, content-addressed materialization of Decision model data.

Only immutable Hugging Face revisions supplied by the catalog are fetched.
Repository Python is never selected or imported: inference implementations live
in this distribution and consume only the verified data files listed by a
packaged runtime profile.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .catalog_adapter import ResolvedRuntimeModel
from .runtime_profile import (
    ArtifactManifestIdentity,
    RuntimeProfileError,
    validate_catalog_revision,
    validate_relative_artifact_path,
)

_RECEIPT = ".vllm-sr-artifact.json"
_RECEIPT_SCHEMA = 1
_REPOSITORY_CODE_SUFFIXES = frozenset(
    {".py", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".sh", ".bash"}
)
_REPOSITORY_CODE_NAMES = frozenset({"pyproject.toml", "setup.py", "requirements.txt"})


class ArtifactError(RuntimeError):
    """A model artifact could not be safely resolved."""


class ArtifactManifestError(ArtifactError):
    """An artifact inventory is malformed or conflicts with its profile."""


class ArtifactIntegrityError(ArtifactError):
    """Artifact bytes differ from their immutable manifest."""


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One file identity from a repository artifact manifest."""

    manifest_path: str
    repository_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """Read-only local artifact view consumed by an owned inference backend."""

    root: Path
    content_id: str
    repository_id: str
    revision: str
    files: tuple[ArtifactFile, ...]


class ArtifactFetcher(Protocol):
    """Minimal fetch seam; implementations must preserve repo/revision identity."""

    def fetch(self, *, repository_id: str, revision: str, filename: str) -> Path:
        """Return one local file for the exact immutable repository revision."""


@dataclass(frozen=True, slots=True)
class HfHubArtifactFetcher:
    """Fetch through the Hugging Face SDK's cache and credential handling.

    No token is accepted by this interface. The SDK obtains authentication from
    its normal login/environment configuration, keeping credentials out of
    process arguments, receipts, exceptions, and runtime profiles.
    """

    local_files_only: bool = False

    def fetch(self, *, repository_id: str, revision: str, filename: str) -> Path:
        try:
            validate_catalog_revision(revision)
            validate_relative_artifact_path(filename, field="artifact filename")
        except RuntimeProfileError as error:
            raise ArtifactError(str(error)) from error
        # Keep the optional network client out of package import and contract-test
        # paths. The base CLI dependency provides it in real installations.
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repository_id,
                revision=revision,
                filename=filename,
                local_files_only=self.local_files_only,
            )
        )


@dataclass(slots=True)
class ArtifactResolver:
    """Verify a pinned manifest and materialize its selected data read-only."""

    fetcher: ArtifactFetcher
    cache_root: Path

    def materialize(self, model: ResolvedRuntimeModel) -> VerifiedArtifact:
        try:
            validate_catalog_revision(model.catalog.revision)
        except RuntimeProfileError as error:
            raise ArtifactError(str(error)) from error
        if (
            model.profile.revision != model.catalog.revision
            or model.repository_id != model.catalog.model_id
        ):
            raise ArtifactError(
                "resolved artifact identity does not match the catalog model"
            )
        identity = model.profile.artifact.manifest
        manifest_source = self.fetcher.fetch(
            repository_id=model.repository_id,
            revision=model.catalog.revision,
            filename=identity.path,
        )
        verify_file_identity(manifest_source, identity)
        inventory = parse_artifact_manifest(
            manifest_source.read_bytes(), manifest_path=identity.path
        )
        files = _select_profile_files(model, inventory)
        receipt = _receipt(identity, files)
        content_id = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        destination = self.cache_root / "sha256" / content_id

        with _content_lock(self.cache_root / ".locks" / f"{content_id}.lock"):
            if destination.exists() or destination.is_symlink():
                _verify_materialization(destination, receipt, files)
            else:
                self._create_materialization(model, destination, receipt, files)
        return VerifiedArtifact(
            root=destination,
            content_id=content_id,
            repository_id=model.repository_id,
            revision=model.catalog.revision,
            files=files,
        )

    def _create_materialization(
        self,
        model: ResolvedRuntimeModel,
        destination: Path,
        receipt: dict[str, Any],
        files: tuple[ArtifactFile, ...],
    ) -> None:
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent)
        )
        try:
            manifest = model.profile.artifact.manifest
            manifest_source = self.fetcher.fetch(
                repository_id=model.repository_id,
                revision=model.catalog.revision,
                filename=manifest.path,
            )
            verify_file_identity(manifest_source, manifest)
            _copy_verified(
                manifest_source,
                temporary / manifest.path,
                sha256=manifest.sha256,
                size_bytes=manifest.size_bytes,
            )
            for item in files:
                source = self.fetcher.fetch(
                    repository_id=model.repository_id,
                    revision=model.catalog.revision,
                    filename=item.repository_path,
                )
                _copy_verified(
                    source,
                    temporary / item.repository_path,
                    sha256=item.sha256,
                    size_bytes=item.size_bytes,
                )

            receipt_path = temporary / _RECEIPT
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _make_read_only(temporary)
            try:
                temporary.rename(destination)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _verify_materialization(destination, receipt, files)
            else:
                temporary = None
        finally:
            if temporary is not None and temporary.exists():
                _make_writable_for_cleanup(temporary)
                shutil.rmtree(temporary)


def parse_artifact_manifest(
    payload: bytes, *, manifest_path: str
) -> dict[str, ArtifactFile]:
    """Parse supported repository manifest shapes into one strict inventory."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactManifestError("artifact manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise ArtifactManifestError("artifact manifest must be an object")
    raw_files = document.get("files")
    if isinstance(raw_files, dict):
        entries = [
            _manifest_entry(path, value, include_path=False)
            for path, value in raw_files.items()
        ]
    elif isinstance(raw_files, list):
        entries = [
            _manifest_entry(None, value, include_path=True) for value in raw_files
        ]
    else:
        raise ArtifactManifestError("artifact manifest files must be an object or list")
    if not entries:
        raise ArtifactManifestError("artifact manifest has no files")

    safe_manifest_path = validate_relative_artifact_path(
        manifest_path, field="manifest path"
    )
    base = PurePosixPath(safe_manifest_path).parent
    inventory: dict[str, ArtifactFile] = {}
    for manifest_relative, digest, size_bytes in entries:
        if manifest_relative in inventory:
            raise ArtifactManifestError(
                f"artifact manifest contains duplicate path {manifest_relative!r}"
            )
        repository_path = (base / manifest_relative).as_posix()
        repository_path = validate_relative_artifact_path(
            repository_path, field="manifest repository path"
        )
        inventory[manifest_relative] = ArtifactFile(
            manifest_path=manifest_relative,
            repository_path=repository_path,
            sha256=digest,
            size_bytes=size_bytes,
        )
    return inventory


def verify_file_identity(path: Path, identity: ArtifactManifestIdentity) -> None:
    """Verify manifest bytes before parsing any repository-supplied metadata."""

    _verify_regular_file(
        path,
        sha256=identity.sha256,
        size_bytes=identity.size_bytes,
        label="artifact manifest",
    )


def _manifest_entry(
    path: object | None, value: object, *, include_path: bool
) -> tuple[str, str, int]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArtifactManifestError("artifact manifest file entry must be an object")
    expected = {"file", "sha256", "bytes"} if include_path else {"sha256", "bytes"}
    if set(value) != expected:
        raise ArtifactManifestError("artifact manifest file fields are invalid")
    if include_path:
        path = value["file"]
    try:
        safe_path = validate_relative_artifact_path(path, field="manifest file")
    except RuntimeProfileError as error:
        raise ArtifactManifestError(str(error)) from error
    digest = value["sha256"]
    size_bytes = value["bytes"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ArtifactManifestError("artifact manifest SHA-256 is invalid")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ArtifactManifestError("artifact manifest byte size is invalid")
    return safe_path, digest, size_bytes


def _select_profile_files(
    model: ResolvedRuntimeModel, inventory: dict[str, ArtifactFile]
) -> tuple[ArtifactFile, ...]:
    files: list[ArtifactFile] = []
    for path in model.profile.artifact.files:
        item = inventory.get(path)
        if item is None:
            raise ArtifactManifestError(
                f"runtime profile selects a file absent from the manifest: {path}"
            )
        _reject_repository_code(item.repository_path)
        files.append(item)
    return tuple(files)


def _reject_repository_code(path: str) -> None:
    name = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _REPOSITORY_CODE_SUFFIXES or name.lower() in _REPOSITORY_CODE_NAMES:
        raise ArtifactManifestError(
            f"runtime profiles may not select repository code: {path}"
        )


def _receipt(
    manifest: ArtifactManifestIdentity, files: tuple[ArtifactFile, ...]
) -> dict[str, Any]:
    return {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest": {
            "path": manifest.path,
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
        },
        "files": [
            {
                "path": item.repository_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in files
        ],
    }


def _copy_verified(
    source: Path, destination: Path, *, sha256: str, size_bytes: int
) -> None:
    _verify_regular_file(
        source, sha256=sha256, size_bytes=size_bytes, label="fetched artifact"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _verify_regular_file(
        destination, sha256=sha256, size_bytes=size_bytes, label="materialized artifact"
    )


def _verify_regular_file(
    path: Path, *, sha256: str, size_bytes: int, label: str
) -> None:
    try:
        actual_size = path.stat().st_size
    except (FileNotFoundError, OSError) as error:
        raise ArtifactIntegrityError(f"{label} is unavailable") from error
    if not path.is_file() or actual_size != size_bytes:
        raise ArtifactIntegrityError(f"{label} size does not match its manifest")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactIntegrityError(f"{label} could not be read") from error
    if digest.hexdigest() != sha256:
        raise ArtifactIntegrityError(f"{label} digest does not match its manifest")


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ArtifactIntegrityError(
                "materialized artifacts may not contain symlinks"
            )
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _make_writable_for_cleanup(root: Path) -> None:
    """Restore permissions only inside an abandoned task-owned temp tree."""

    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


@contextmanager
def _content_lock(path: Path):
    """Serialize one content ID across local processes without partial views."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ArtifactError(
            "could not open the artifact materialization lock"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactError("artifact materialization lock is not a regular file")
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - production is Linux/Darwin
            raise ArtifactError(
                "artifact materialization requires POSIX file locking"
            ) from error
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _verify_materialization(
    root: Path,
    receipt: dict[str, Any],
    files: tuple[ArtifactFile, ...],
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactIntegrityError("content-addressed artifact root is invalid")
    receipt_path = root / _RECEIPT
    try:
        actual_receipt = json.loads(receipt_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            "artifact receipt is unavailable or invalid"
        ) from error
    if actual_receipt != receipt:
        raise ArtifactIntegrityError(
            "artifact receipt differs from its content identity"
        )

    expected_paths = {item.repository_path for item in files}
    expected_paths.add(receipt["manifest"]["path"])
    expected_paths.add(_RECEIPT)
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ArtifactIntegrityError("materialized artifact file inventory drifted")

    manifest = receipt["manifest"]
    _verify_regular_file(
        root / manifest["path"],
        sha256=manifest["sha256"],
        size_bytes=manifest["size_bytes"],
        label="materialized artifact manifest",
    )
    for item in files:
        path = root / item.repository_path
        if path.is_symlink():
            raise ArtifactIntegrityError(
                "materialized artifacts may not contain symlinks"
            )
        _verify_regular_file(
            path,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
            label="materialized artifact",
        )
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ArtifactIntegrityError(
                "materialized artifacts may not contain symlinks"
            )
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ArtifactIntegrityError("materialized artifacts must be read-only")
