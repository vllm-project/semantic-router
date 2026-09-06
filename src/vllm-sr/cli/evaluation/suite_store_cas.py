"""Private filesystem and content-addressed storage for normalized suites."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from cli.evaluation.canonical import sha256_digest
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    SuiteArtifactInstall,
    SuiteArtifactRole,
)

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_CHUNK_BYTES = 1024 * 1024
_PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

ArtifactValidator = Callable[[int, SuiteArtifactRole, int], int]


class SuiteCAS:
    """Own path confinement, permissions, immutable writes, and CAS publication."""

    def __init__(self, root: str | Path, *, create: bool):
        expanded = Path(root).expanduser()
        if expanded.is_symlink():
            raise SuiteStoreError("suite store root must not be a symlink")
        self.root = expanded.absolute()
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests" / "sha256"
        self.index = self.root / "index"
        directories = [self.root, self.objects]
        for domain in ("visible", "grading", "metadata"):
            directories.extend(
                (self.objects / domain, self.objects / domain / "sha256")
            )
        directories.extend((self.root / "manifests", self.manifests, self.index))
        for directory in directories:
            if create:
                self.ensure_private_dir(directory)
            else:
                self.require_private_dir(directory)
        self.read_only = not create

    @staticmethod
    def require_private_dir(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            raise SuiteStoreError("read-only suite store is incomplete")
        SuiteCAS.ensure_private_dir(path)

    @staticmethod
    def ensure_private_dir(path: Path) -> None:
        if path.exists() or path.is_symlink():
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise SuiteStoreError("suite store directory is not readable") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise SuiteStoreError("symlink is not allowed in suite store")
            if not stat.S_ISDIR(metadata.st_mode):
                raise SuiteStoreError("suite store path is not a directory")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode != _PRIVATE_DIR_MODE:
                raise SuiteStoreError(
                    f"suite store directory must have mode 0700, got {mode:04o}"
                )
            return
        try:
            path.mkdir(parents=True, mode=_PRIVATE_DIR_MODE)
            os.chmod(path, _PRIVATE_DIR_MODE)
        except OSError as exc:
            raise SuiteStoreError("could not create private suite store") from exc
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != _PRIVATE_DIR_MODE:
            raise SuiteStoreError("could not enforce suite directory mode 0700")

    @staticmethod
    def validate_private_file(path: Path) -> os.stat_result:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SuiteStoreError("suite store file is missing or unreadable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SuiteStoreError("symlink is not allowed in suite store")
        if not stat.S_ISREG(metadata.st_mode):
            raise SuiteStoreError("suite store object is not a regular file")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != _PRIVATE_FILE_MODE:
            raise SuiteStoreError(
                f"suite store file must have mode 0600, got {mode:04o}"
            )
        return metadata

    @staticmethod
    def portable_id(value: str) -> str:
        if not _PORTABLE_ID_RE.fullmatch(value):
            raise SuiteStoreError("invalid suite id")
        return value

    @staticmethod
    def digest_hex(digest: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise SuiteStoreError("invalid SHA256 digest")
        return digest.removeprefix("sha256:")

    def object_path(self, role: SuiteArtifactRole, digest: str) -> Path:
        _, _, domain = ARTIFACT_ROLE_LAYOUT[role]
        return self.objects / domain / "sha256" / self.digest_hex(digest)

    def index_path(self, suite_id: str) -> Path:
        return self.index / f"{self.portable_id(suite_id)}.json"

    def manifest_path(self, digest: str) -> Path:
        return self.manifests / self.digest_hex(digest)

    @staticmethod
    def open_readonly(path: Path) -> int:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise SuiteStoreError("could not safely open suite file") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise SuiteStoreError("suite input is not a regular file")
        return descriptor

    @staticmethod
    def safe_bundle_root(bundle_root: str | Path) -> Path:
        root = Path(bundle_root).expanduser()
        if root.is_symlink():
            raise SuiteStoreError("bundle root must not be a symlink")
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise SuiteStoreError("bundle root is missing") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise SuiteStoreError("bundle root is not a directory")
        return root.resolve(strict=True)

    @staticmethod
    def bundle_file(root: Path, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise SuiteStoreError("bundle artifact path is not a safe relative path")
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts[:-1]:
            current /= part
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise SuiteStoreError("bundle artifact parent is missing") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise SuiteStoreError("bundle artifact parent must not be a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise SuiteStoreError("bundle artifact parent is not a directory")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise SuiteStoreError(
                "bundle artifact is missing or escapes its root"
            ) from exc
        if candidate.is_symlink():
            raise SuiteStoreError("bundle artifact must not be a symlink")
        return candidate

    @staticmethod
    def stream_digest_file(path: Path) -> tuple[str, int]:
        descriptor = SuiteCAS.open_readonly(path)
        digest = hashlib.sha256()
        total = 0
        try:
            while True:
                chunk = os.read(descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        finally:
            os.close(descriptor)
        return "sha256:" + digest.hexdigest(), total

    @classmethod
    def verify_ref(cls, path: Path, ref: ArtifactRef) -> None:
        cls.validate_private_file(path)
        digest, size = cls.stream_digest_file(path)
        if digest != ref.digest or size != ref.size_bytes:
            raise SuiteStoreError(f"corrupt suite object {ref.digest}")

    def _publish_temp_object(
        self, temporary: Path, target: Path, ref: ArtifactRef
    ) -> None:
        if target.exists() or target.is_symlink():
            self.verify_ref(target, ref)
            temporary.unlink()
            return
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            self.verify_ref(target, ref)
        except OSError as exc:
            raise SuiteStoreError("could not publish suite CAS object") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        self.verify_ref(target, ref)

    def stage_artifact(
        self,
        root: Path,
        artifact: SuiteArtifactInstall,
        validate: ArtifactValidator,
    ) -> tuple[ArtifactRef, int]:
        source = self.bundle_file(root, artifact.relative_path)
        source_descriptor = self.open_readonly(source)
        initial = os.fstat(source_descriptor)
        if initial.st_size != artifact.size_bytes:
            os.close(source_descriptor)
            raise SuiteStoreError(
                f"bundle artifact {artifact.role!r} does not match its declared size"
            )
        _, expected_media_type, domain = ARTIFACT_ROLE_LAYOUT[artifact.role]
        object_directory = self.objects / domain / "sha256"
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".install-", dir=object_directory
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            os.fchmod(temporary_descriptor, _PRIVATE_FILE_MODE)
            while True:
                chunk = os.read(source_descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(temporary_descriptor, view)
                    if written <= 0:
                        raise SuiteStoreError(
                            "short write while staging suite artifact"
                        )
                    view = view[written:]
            os.fsync(temporary_descriptor)
            final = os.fstat(source_descriptor)
            if (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
            ) != (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
            ):
                raise SuiteStoreError("bundle artifact changed while it was installed")
            observed_digest = "sha256:" + digest.hexdigest()
            if observed_digest != artifact.digest or size != artifact.size_bytes:
                raise SuiteStoreError(
                    f"bundle artifact {artifact.role!r} does not match its digest and size"
                )
            ref = ArtifactRef(
                digest=observed_digest,
                size_bytes=size,
                media_type=expected_media_type,
            )
            record_count = validate(temporary_descriptor, artifact.role, size)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            os.close(source_descriptor)
            os.close(temporary_descriptor)
        target = self.object_path(artifact.role, ref.digest)
        self._publish_temp_object(temporary, target, ref)
        return ref, record_count

    @staticmethod
    def write_immutable(path: Path, data: bytes) -> None:
        if path.exists() or path.is_symlink():
            SuiteCAS.validate_private_file(path)
            digest, size = SuiteCAS.stream_digest_file(path)
            if digest != sha256_digest(data) or size != len(data):
                raise SuiteStoreError(
                    "immutable suite metadata already has other content"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise SuiteStoreError("short write while publishing suite metadata")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                SuiteCAS.validate_private_file(path)
                digest, size = SuiteCAS.stream_digest_file(path)
                if digest != sha256_digest(data) or size != len(data):
                    raise SuiteStoreError(
                        "immutable suite metadata already has other content"
                    ) from None
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary.exists():
                temporary.unlink()
