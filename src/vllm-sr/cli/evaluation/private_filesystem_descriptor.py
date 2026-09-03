"""Descriptor ownership and validated reads for private evaluation artifacts."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from typing_extensions import Self

from cli.evaluation.artifact_store_error import StoreError

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600

DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def require_private_directory_descriptor(
    descriptor: int,
    description: str,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISDIR(metadata.st_mode) or mode != PRIVATE_DIRECTORY_MODE:
        raise StoreError(f"{description} must be a directory with mode 0700")
    return metadata


def require_private_file_descriptor(
    descriptor: int,
    description: str,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or mode != PRIVATE_FILE_MODE
        or metadata.st_nlink != 1
    ):
        raise StoreError(
            f"{description} must be a single-link regular file with mode 0600"
        )
    return metadata


def stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_descriptor(descriptor: int, description: str) -> tuple[bytes, os.stat_result]:
    before = require_private_file_descriptor(descriptor, description)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    after = require_private_file_descriptor(descriptor, description)
    if (
        stable_file_identity(before) != stable_file_identity(after)
        or len(data) != before.st_size
    ):
        raise StoreError(f"{description} changed while it was being read")
    return data, after


class PrivateFilesystemReadPrimitives:
    """Own the anchored root descriptor and all validated read operations."""

    def __init__(self, root: Path, root_descriptor: int) -> None:
        self.root = root
        self._root_descriptor = root_descriptor

    @classmethod
    def create(cls, root: str | Path) -> Self:
        expanded = Path(root).expanduser().absolute()
        if expanded.is_symlink():
            raise StoreError("artifact store root must not be a symlink")
        try:
            expanded.mkdir(
                parents=True,
                mode=PRIVATE_DIRECTORY_MODE,
                exist_ok=True,
            )
            descriptor = os.open(expanded, DIRECTORY_OPEN_FLAGS)
        except OSError as exc:
            raise StoreError("artifact store root is unsafe or unavailable") from exc
        try:
            opened = require_private_directory_descriptor(
                descriptor,
                "artifact store root",
            )
            resolved = expanded.resolve(strict=True)
            verification = os.open(resolved, DIRECTORY_OPEN_FLAGS)
            try:
                if not same_inode(opened, os.fstat(verification)):
                    raise StoreError(
                        "artifact store root changed during initialization"
                    )
            finally:
                os.close(verification)
        except BaseException:
            os.close(descriptor)
            raise
        return cls(resolved, descriptor)

    def __del__(self) -> None:
        with suppress(OSError):
            os.close(self._root_descriptor)

    def within_root(self, path: str | Path) -> Path:
        candidate = Path(os.path.abspath(os.fspath(path)))
        if not candidate.is_relative_to(self.root):
            raise StoreError("artifact path escapes store root")
        return candidate

    def _parts(self, path: str | Path) -> tuple[str, ...]:
        candidate = self.within_root(path)
        parts = candidate.relative_to(self.root).parts
        if any(part in {"", ".", ".."} for part in parts):
            raise StoreError("artifact path contains an unsafe component")
        return parts

    @staticmethod
    def _name(path: Path) -> str:
        name = path.name
        if not name or name in {".", ".."} or Path(name).name != name:
            raise StoreError("artifact filename is unsafe")
        return name

    def _duplicate_root(self) -> int:
        descriptor = os.dup(self._root_descriptor)
        try:
            require_private_directory_descriptor(descriptor, "artifact store root")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager
    def _directory_descriptor(self, path: str | Path) -> Iterator[int]:
        candidate = self.within_root(path)
        descriptor = self._duplicate_root()
        try:
            for part in self._parts(candidate):
                try:
                    child = os.open(part, DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    raise
                except OSError as exc:
                    raise StoreError(
                        f"private directory is a symlink or unsafe: {candidate}"
                    ) from exc
                os.close(descriptor)
                descriptor = child
                require_private_directory_descriptor(
                    descriptor,
                    f"private directory {candidate}",
                )
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_file_at(
        directory_descriptor: int,
        name: str,
        *,
        flags: int = _FILE_READ_FLAGS,
        mode: int = PRIVATE_FILE_MODE,
        description: str,
    ) -> int:
        try:
            descriptor = os.open(
                name,
                flags | os.O_NONBLOCK,
                mode,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            raise
        except FileExistsError:
            raise
        except OSError as exc:
            raise StoreError(f"{description} is a symlink or unsafe") from exc
        try:
            require_private_file_descriptor(descriptor, description)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def directory_exists(self, path: str | Path) -> bool:
        try:
            with self._directory_descriptor(path):
                return True
        except FileNotFoundError:
            return False

    def read_private_file(
        self,
        path: str | Path,
        *,
        missing_ok: bool = False,
    ) -> bytes | None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        try:
            with self._directory_descriptor(candidate.parent) as parent:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    description=f"private file {candidate}",
                )
                try:
                    data, _ = read_descriptor(
                        descriptor,
                        f"private file {candidate}",
                    )
                    return data
                finally:
                    os.close(descriptor)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise StoreError(f"private file is unavailable: {candidate}") from None

    def read_private_file_tail(
        self,
        path: str | Path,
        maximum_bytes: int,
        *,
        missing_ok: bool = False,
    ) -> tuple[int, bytes] | None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            try:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    description=f"private tail file {candidate}",
                )
            except FileNotFoundError as exc:
                if missing_ok:
                    return None
                raise StoreError(f"private file is unavailable: {candidate}") from exc
            try:
                before = require_private_file_descriptor(
                    descriptor,
                    f"private tail file {candidate}",
                )
                size = before.st_size
                read_size = min(size, maximum_bytes)
                os.lseek(descriptor, size - read_size, os.SEEK_SET)
                chunks: list[bytes] = []
                remaining = read_size
                while remaining:
                    chunk = os.read(descriptor, remaining)
                    if not chunk:
                        raise StoreError("private file changed while reading its tail")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                after = require_private_file_descriptor(
                    descriptor,
                    f"private tail file {candidate}",
                )
                if stable_file_identity(before) != stable_file_identity(after):
                    raise StoreError("private file changed while reading its tail")
                return size, b"".join(chunks)
            finally:
                os.close(descriptor)

    def directory_entries(self, path: str | Path) -> tuple[str, ...]:
        try:
            with self._directory_descriptor(path) as descriptor:
                return tuple(sorted(os.listdir(descriptor)))
        except FileNotFoundError as exc:
            raise StoreError("private directory is unavailable") from exc
