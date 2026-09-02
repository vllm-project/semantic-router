"""Durable directory and file mutations for private evaluation artifacts."""

from __future__ import annotations

import fcntl
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.private_filesystem_descriptor import (
    DIRECTORY_OPEN_FLAGS,
    PRIVATE_DIRECTORY_MODE,
    PrivateFilesystemReadPrimitives,
    read_descriptor,
    require_private_directory_descriptor,
    same_inode,
)


def _write_all(descriptor: int, data: bytes, description: str) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise StoreError(f"short write while writing {description}")
        remaining = remaining[written:]


class PrivateFilesystemMutationPrimitives(PrivateFilesystemReadPrimitives):
    """Apply descriptor-anchored mutations without owning publication policy."""

    @staticmethod
    def _open_or_create_directory_at(
        parent: int,
        name: str,
        candidate: Path,
        *,
        allow_create: bool,
    ) -> int:
        try:
            return os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            if not allow_create:
                raise StoreError(
                    f"parent directory is unavailable for {candidate}"
                ) from None
            try:
                os.mkdir(name, PRIVATE_DIRECTORY_MODE, dir_fd=parent)
            except FileExistsError:
                pass
            except OSError as exc:
                raise StoreError(
                    f"cannot create private directory: {candidate}"
                ) from exc
            os.fsync(parent)
            try:
                return os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=parent)
            except OSError as exc:
                raise StoreError(
                    f"created private directory is unsafe: {candidate}"
                ) from exc
        except OSError as exc:
            raise StoreError(
                f"private directory is a symlink or unsafe: {candidate}"
            ) from exc

    def ensure_private_directory(
        self,
        path: str | Path,
        *,
        parents: bool,
    ) -> Path:
        candidate = self.within_root(path)
        descriptor = self._duplicate_root()
        try:
            parts = self._parts(candidate)
            for index, part in enumerate(parts):
                child = self._open_or_create_directory_at(
                    descriptor,
                    part,
                    candidate,
                    allow_create=parents or index == len(parts) - 1,
                )
                os.close(descriptor)
                descriptor = child
                require_private_directory_descriptor(
                    descriptor,
                    f"private directory {candidate}",
                )
        finally:
            os.close(descriptor)
        return candidate

    def sync_directory(self, path: str | Path) -> None:
        try:
            with self._directory_descriptor(path) as descriptor:
                os.fsync(descriptor)
        except FileNotFoundError as exc:
            raise StoreError("private directory is unavailable for sync") from exc

    def atomic_write(self, path: str | Path, data: bytes) -> None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        temporary_name = ""
        with self._directory_descriptor(candidate.parent) as parent:
            try:
                existing = self._open_file_at(
                    parent,
                    name,
                    description=f"private file {candidate}",
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                os.close(existing)
            try:
                for _ in range(16):
                    candidate_name = f".{name}.{secrets.token_hex(4)}"
                    try:
                        descriptor = self._open_file_at(
                            parent,
                            candidate_name,
                            flags=(
                                os.O_WRONLY
                                | os.O_CREAT
                                | os.O_EXCL
                                | os.O_CLOEXEC
                                | os.O_NOFOLLOW
                            ),
                            description="artifact temporary file",
                        )
                        temporary_name = candidate_name
                        break
                    except FileExistsError:
                        continue
                else:
                    raise StoreError("cannot allocate an artifact temporary file")
                try:
                    _write_all(descriptor, data, "artifact temporary file")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.replace(
                    temporary_name,
                    name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                published = self._open_file_at(
                    parent,
                    name,
                    description=f"published private file {candidate}",
                )
                try:
                    published_data, _ = read_descriptor(
                        published,
                        f"published private file {candidate}",
                    )
                    if published_data != data:
                        raise StoreError(
                            "atomic artifact publication changed its content"
                        )
                finally:
                    os.close(published)
                os.fsync(parent)
            finally:
                if temporary_name:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=parent)

    def append_private_file(self, path: str | Path, data: bytes) -> None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            created = False
            try:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    flags=os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
                    description=f"private append file {candidate}",
                )
            except FileNotFoundError:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    flags=(
                        os.O_WRONLY
                        | os.O_APPEND
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW
                    ),
                    description=f"private append file {candidate}",
                )
                created = True
            try:
                _write_all(descriptor, data, f"private append file {candidate}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if created:
                os.fsync(parent)

    @contextmanager
    def exclusive_lock(self, path: str | Path, description: str) -> Iterator[None]:
        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            created = False
            try:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    flags=os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                    description=description,
                )
            except FileNotFoundError:
                try:
                    descriptor = self._open_file_at(
                        parent,
                        name,
                        flags=(
                            os.O_RDWR
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_CLOEXEC
                            | os.O_NOFOLLOW
                        ),
                        description=description,
                    )
                    created = True
                except FileExistsError:
                    descriptor = self._open_file_at(
                        parent,
                        name,
                        flags=os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                        description=description,
                    )
            if created:
                os.fsync(parent)
            locked = False
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                yield
            finally:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    @contextmanager
    def exclusive_existing_lock(
        self,
        path: str | Path,
        description: str,
    ) -> Iterator[None]:
        """Acquire an existing lock file without creating filesystem state."""

        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            try:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    flags=os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                    description=description,
                )
            except FileNotFoundError:
                yield
                return
            locked = False
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                yield
            finally:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def unlink_private_file(
        self,
        path: str | Path,
        *,
        expected_data: bytes | None = None,
    ) -> None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            try:
                descriptor = self._open_file_at(
                    parent,
                    name,
                    description=f"private file {candidate}",
                )
            except FileNotFoundError as exc:
                raise StoreError(f"private file is unavailable: {candidate}") from exc
            try:
                data, opened = read_descriptor(
                    descriptor,
                    f"private file {candidate}",
                )
                if expected_data is not None and data != expected_data:
                    raise StoreError("private file changed before unlink")
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not same_inode(opened, current):
                    raise StoreError("private file changed before unlink")
                os.unlink(name, dir_fd=parent)
            finally:
                os.close(descriptor)

    def remove_private_directory(self, path: str | Path) -> None:
        candidate = self.within_root(path)
        name = self._name(candidate)
        with self._directory_descriptor(candidate.parent) as parent:
            try:
                child = os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=parent)
            except OSError as exc:
                raise StoreError("private directory is unsafe or unavailable") from exc
            try:
                require_private_directory_descriptor(
                    child,
                    f"private directory {candidate}",
                )
                if os.listdir(child):
                    raise StoreError("private directory is not empty")
                opened = os.fstat(child)
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not same_inode(opened, current):
                    raise StoreError("private directory changed before removal")
                os.rmdir(name, dir_fd=parent)
            finally:
                os.close(child)

    def rename_private_directory(self, source: Path, target: Path) -> None:
        source = self.within_root(source)
        target = self.within_root(target)
        if source.parent != target.parent:
            raise StoreError("private directory rename must retain its parent")
        source_name = self._name(source)
        target_name = self._name(target)
        with self._directory_descriptor(source.parent) as parent:
            try:
                source_descriptor = os.open(
                    source_name,
                    DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent,
                )
            except OSError as exc:
                raise StoreError("private source directory is unsafe") from exc
            try:
                source_metadata = require_private_directory_descriptor(
                    source_descriptor,
                    f"private directory {source}",
                )
                try:
                    os.stat(target_name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise StoreError("private destination directory already exists")
                os.rename(
                    source_name,
                    target_name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                target_descriptor = os.open(
                    target_name,
                    DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent,
                )
                try:
                    target_metadata = require_private_directory_descriptor(
                        target_descriptor,
                        f"private directory {target}",
                    )
                    if not same_inode(source_metadata, target_metadata):
                        raise StoreError(
                            "private directory rename changed its identity"
                        )
                finally:
                    os.close(target_descriptor)
            finally:
                os.close(source_descriptor)
