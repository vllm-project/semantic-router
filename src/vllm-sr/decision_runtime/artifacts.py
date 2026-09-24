"""Verified, content-addressed materialization of Decision model data.

Only immutable Hugging Face revisions supplied by the catalog are fetched.
Inference implementations live in this distribution. Repository Python is
never imported or executed. One hash-pinned Qwen guard contract is carried as
inert data because its release metadata references its exact bytes.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .catalog_adapter import ResolvedRuntimeModel
from .release_artifacts import ReleaseArtifactError, select_qwen_weight_files
from .runtime_profile import (
    ArtifactManifestIdentity,
    RuntimeProfileError,
    validate_catalog_revision,
    validate_relative_artifact_path,
)

_RECEIPT = ".vllm-sr-artifact.json"
_RECEIPT_SCHEMA = 2
_SHA256_HEX_LENGTH = 64
_REPOSITORY_CODE_SUFFIXES = frozenset(
    {".py", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".sh", ".bash"}
)
_REPOSITORY_CODE_NAMES = frozenset({"pyproject.toml", "setup.py", "requirements.txt"})
_INERT_QWEN_GUARD = "code/profile_guard.py"


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
    data_root: Path
    content_id: str
    repository_id: str
    revision: str
    files: tuple[ArtifactFile, ...]
    manifest: ArtifactManifestIdentity | None = None


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
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

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
    """Verify an immutable snapshot's own inventory and materialize data read-only."""

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
        manifest_path = model.profile.artifact.manifest_path
        manifest_source = self.fetcher.fetch(
            repository_id=model.repository_id,
            revision=model.catalog.revision,
            filename=manifest_path,
        )
        identity = _observed_manifest_identity(manifest_source, manifest_path)
        inventory = parse_artifact_manifest(
            manifest_source.read_bytes(), manifest_path=manifest_path
        )
        files = _select_artifact_files(model, inventory)
        receipt = _receipt(model, identity, files)
        content_id = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        destination = self.cache_root / "sha256" / content_id

        with _content_lock(self.cache_root / ".locks" / f"{content_id}.lock"):
            if destination.exists() or destination.is_symlink():
                _verify_materialization(destination, receipt, files)
            else:
                self._create_materialization(
                    model, destination, identity, receipt, files
                )
        return _verified_artifact(
            root=destination,
            content_id=content_id,
            model=model,
            files=files,
            manifest=identity,
        )

    def _create_materialization(
        self,
        model: ResolvedRuntimeModel,
        destination: Path,
        identity: ArtifactManifestIdentity,
        receipt: dict[str, Any],
        files: tuple[ArtifactFile, ...],
    ) -> None:
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent)
        )
        try:
            manifest_source = self.fetcher.fetch(
                repository_id=model.repository_id,
                revision=model.catalog.revision,
                filename=identity.path,
            )
            verify_file_identity(manifest_source, identity)
            _copy_verified(
                manifest_source,
                temporary / identity.path,
                sha256=identity.sha256,
                size_bytes=identity.size_bytes,
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


def open_verified_artifact(
    root: Path,
    model: ResolvedRuntimeModel,
    *,
    expected_content_id: str,
) -> VerifiedArtifact:
    """Re-open and fully verify one already-materialized artifact.

    Runtime containers receive a read-only content tree rather than Hub
    credentials.  This function repeats the catalog/profile, receipt, manifest,
    inventory, size, digest, and permission checks inside that trust boundary
    before a model loader sees any bytes.
    """

    if (
        not isinstance(expected_content_id, str)
        or len(expected_content_id) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_content_id)
    ):
        raise ArtifactError("expected artifact content ID is invalid")
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

    requested_root = Path(root)
    if requested_root.is_symlink():
        raise ArtifactIntegrityError("content-addressed artifact root is invalid")
    try:
        resolved_root = requested_root.resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError(
            "content-addressed artifact root is unavailable"
        ) from error

    receipt_path = resolved_root / _RECEIPT
    try:
        saved_receipt = json.loads(receipt_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            "artifact receipt is unavailable or invalid"
        ) from error
    identity = _manifest_identity_from_receipt(saved_receipt, model)
    manifest_path = resolved_root / identity.path
    verify_file_identity(manifest_path, identity)
    inventory = parse_artifact_manifest(
        manifest_path.read_bytes(), manifest_path=identity.path
    )
    files = _select_artifact_files(model, inventory)
    receipt = _receipt(model, identity, files)
    actual_content_id = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_content_id != expected_content_id:
        raise ArtifactIntegrityError(
            "materialized artifact content identity does not match the launch contract"
        )
    _verify_materialization(resolved_root, receipt, files)
    return _verified_artifact(
        root=resolved_root,
        content_id=actual_content_id,
        model=model,
        files=files,
        manifest=identity,
    )


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


def _observed_manifest_identity(
    path: Path, manifest_path: str
) -> ArtifactManifestIdentity:
    """Pin the manifest bytes from the selected immutable Hub commit."""

    validate_relative_artifact_path(manifest_path, field="artifact manifest path")
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise ArtifactIntegrityError("artifact manifest is unavailable") from error
    if not path.is_file() or not 0 < size_bytes <= 4 * 1024 * 1024:
        raise ArtifactManifestError("artifact manifest has an invalid size")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactIntegrityError("artifact manifest is unreadable") from error
    return ArtifactManifestIdentity(manifest_path, digest, size_bytes)


def _manifest_identity_from_receipt(
    value: object, model: ResolvedRuntimeModel
) -> ArtifactManifestIdentity:
    if not isinstance(value, dict) or value.get("schema_version") != _RECEIPT_SCHEMA:
        raise ArtifactIntegrityError("artifact receipt schema is unsupported")
    if (
        value.get("repository_id") != model.repository_id
        or value.get("revision") != model.catalog.revision
    ):
        raise ArtifactIntegrityError("artifact receipt has a different model revision")
    raw = value.get("manifest")
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
        raise ArtifactIntegrityError("artifact receipt manifest is invalid")
    if raw["path"] != model.profile.artifact.manifest_path:
        raise ArtifactIntegrityError("artifact manifest layout is unsupported")
    digest, size_bytes = raw["sha256"], raw["size_bytes"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or not 0 < size_bytes <= 4 * 1024 * 1024
    ):
        raise ArtifactIntegrityError("artifact receipt manifest identity is invalid")
    return ArtifactManifestIdentity(raw["path"], digest, size_bytes)


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
        or len(digest) != _SHA256_HEX_LENGTH
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
        _reject_repository_code(
            item.repository_path,
            inert_guard=(
                model.profile.family == "qwen3.5"
                and model.profile.artifact.manifest_path == "bundle-manifest.json"
                and path == _INERT_QWEN_GUARD
            ),
        )
        files.append(item)
    return tuple(files)


def _select_artifact_files(
    model: ResolvedRuntimeModel, inventory: dict[str, ArtifactFile]
) -> tuple[ArtifactFile, ...]:
    """Select data by family layout, independent of a packaged file list.

    Synthetic profiles with a different manifest layout retain their explicit
    selection seam. Production loaders only accept the three known layouts.
    """

    manifest_path = model.profile.artifact.manifest_path
    if manifest_path not in {
        "native/MANIFEST.json",
        "MODEL_MANIFEST.json",
        "bundle-manifest.json",
    }:
        if not model.profile.artifact.files:
            raise ArtifactManifestError(
                "Decision artifact manifest layout is unsupported"
            )
        return _select_profile_files(model, inventory)

    if model.profile.family == "vela":
        if manifest_path != "native/MANIFEST.json":
            raise ArtifactManifestError("Vela artifact manifest layout is unsupported")
        required = {
            "INVENTORY.json",
            "STATE_LAYOUT.json",
            "choice_encoder.safetensors",
            "decision_config.json",
            "decision_heads.safetensors",
            "encoder/config.json",
            "encoder/model.safetensors",
            "score_encoder.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        }
        optional = {"tokenizer/special_tokens_map.json"}
    elif model.profile.family == "qwen3.5":
        if manifest_path not in {"MODEL_MANIFEST.json", "bundle-manifest.json"}:
            raise ArtifactManifestError("Qwen artifact manifest layout is unsupported")
        required = {
            "backbone/config.json",
            "decision_config.json",
            "decision_head.safetensors",
            "runtime.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        optional = {
            "chat_template.jinja",
            "temperature.json",
            "runtime-profile/profile.json",
            "runtime-profile/l2norm_fwd_kernel.json",
            _INERT_QWEN_GUARD,
        }
        try:
            required.update(select_qwen_weight_files(inventory))
        except ReleaseArtifactError as error:
            raise ArtifactManifestError(str(error)) from error
    else:  # pragma: no cover - family parser closes this path
        raise ArtifactManifestError("Decision artifact family is unsupported")

    missing = required - inventory.keys()
    if missing:
        raise ArtifactManifestError(
            f"Decision artifact manifest is missing {sorted(missing)[0]}"
        )
    selected = required | (optional & inventory.keys())
    files = tuple(inventory[path] for path in sorted(selected))
    for item in files:
        _reject_repository_code(
            item.repository_path,
            inert_guard=(
                model.profile.family == "qwen3.5"
                and item.manifest_path == _INERT_QWEN_GUARD
                and manifest_path == "bundle-manifest.json"
            ),
        )
    return files


def _verified_artifact(
    *,
    root: Path,
    content_id: str,
    model: ResolvedRuntimeModel,
    files: tuple[ArtifactFile, ...],
    manifest: ArtifactManifestIdentity,
) -> VerifiedArtifact:
    manifest_parent = PurePosixPath(manifest.path).parent
    data_root = root
    if manifest_parent != PurePosixPath("."):
        data_root = root.joinpath(*manifest_parent.parts)
    if data_root.is_symlink() or not data_root.is_dir():
        raise ArtifactIntegrityError("artifact data root is invalid")
    return VerifiedArtifact(
        root=root,
        data_root=data_root,
        content_id=content_id,
        repository_id=model.repository_id,
        revision=model.catalog.revision,
        files=files,
        manifest=manifest,
    )


def _reject_repository_code(path: str, *, inert_guard: bool = False) -> None:
    if inert_guard and path == _INERT_QWEN_GUARD:
        return
    name = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _REPOSITORY_CODE_SUFFIXES or name.lower() in _REPOSITORY_CODE_NAMES:
        raise ArtifactManifestError(
            f"runtime profiles may not select repository code: {path}"
        )


def _receipt(
    model: ResolvedRuntimeModel,
    manifest: ArtifactManifestIdentity,
    files: tuple[ArtifactFile, ...],
) -> dict[str, Any]:
    return {
        "schema_version": _RECEIPT_SCHEMA,
        "repository_id": model.repository_id,
        "revision": model.catalog.revision,
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
            import fcntl  # noqa: PLC0415 - preserve the POSIX availability check
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
