"""Bounded file I/O for immutable prepared dataset artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from contextlib import contextmanager

MAX_ROW_BYTES = 128 * 1024 * 1024
MAX_SCAN_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROWS = 50000
CHUNK_BYTES = 1024 * 1024


class DatasetSizeLimitError(ValueError):
    """A known input size cannot fit within this bounded dataset read."""

    def __init__(self, path, size, maximum, reason_code="source_size_limit"):
        super().__init__("Dataset input exceeds the supported size limit")
        self.path = path
        self.size = size
        self.maximum = maximum
        self.reason_code = reason_code


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def identity(path, *, dir_fd=None):
    value = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Dataset input must be a regular file")
    return _stat_identity(value)


@contextmanager
def verified_file(path, maximum, reason_code="source_size_limit", *, dir_fd=None):
    """Bind validation to the opened inode, rejecting links and concurrent changes."""
    before = identity(path, dir_fd=dir_fd)
    if before[2] > maximum:
        raise DatasetSizeLimitError(path, before[2], maximum, reason_code)
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd
        )
    except OSError as error:
        raise ValueError("Dataset input must remain a regular file") from error
    with os.fdopen(descriptor, "rb") as source:
        if _stat_identity(os.fstat(source.fileno())) != before:
            raise ValueError("Dataset input changed while being read")
        yield source
        if (
            _stat_identity(os.fstat(source.fileno())) != before
            or identity(path, dir_fd=dir_fd) != before
        ):
            raise ValueError("Dataset input changed while being read")


def read_small(path, maximum, *, dir_fd=None):
    with verified_file(path, maximum, dir_fd=dir_fd) as source:
        content = source.read(maximum + 1)
        if len(content) > maximum:
            raise DatasetSizeLimitError(path, len(content), maximum)
    return content


def file_digest(path, maximum=MAX_SCAN_BYTES, *, dir_fd=None):
    checksum, size = hashlib.sha256(), 0
    with verified_file(path, maximum, "scan_budget_exhausted", dir_fd=dir_fd) as source:
        while chunk := source.read(CHUNK_BYTES):
            size += len(chunk)
            if size > maximum:
                raise DatasetSizeLimitError(
                    path, size, maximum, "scan_budget_exhausted"
                )
            checksum.update(chunk)
    return checksum.hexdigest()


def bounded_lines(path, *, maximum=MAX_SCAN_BYTES, row_bytes=MAX_ROW_BYTES):
    """Read bounded raw lines from one unchanged inode, including whitespace."""
    size = 0
    with verified_file(path, maximum, "scan_budget_exhausted") as source:
        while line := source.readline(row_bytes + 1):
            if len(line) > row_bytes:
                raise DatasetSizeLimitError(path, len(line), row_bytes)
            size += len(line)
            if size > maximum:
                raise DatasetSizeLimitError(
                    path, size, maximum, "scan_budget_exhausted"
                )
            yield line


def verified_lines(
    path, expected_sha, *, maximum=MAX_SCAN_BYTES, row_bytes=MAX_ROW_BYTES
):
    """Yield bounded lines; callers must exhaust the scan before publishing results."""
    checksum = hashlib.sha256()
    for line in bounded_lines(path, maximum=maximum, row_bytes=row_bytes):
        checksum.update(line)
        if line.strip():
            yield line
    if checksum.hexdigest() != expected_sha:
        raise ValueError("Prepared dataset identity or content digest does not match")


def publish_dataset(directory, staged_data, sha, staged_manifest, rendered):
    """Publish complete files without following or replacing existing inodes."""
    before = directory.lstat()
    if not stat.S_ISDIR(before.st_mode) or directory.resolve() != directory:
        raise ValueError("Prepared dataset must not be a symlink")
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Prepared dataset directory changed")
        # Inspect existing metadata before adding any data to an existing folder.
        try:
            existing = read_small("manifest.json", 2 * 1024 * 1024, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        else:
            if existing != rendered:
                raise ValueError("Existing immutable dataset manifest changed")
        for stage, name in (
            (staged_data, "cases.jsonl"),
            (staged_manifest, "manifest.json"),
        ):
            stage.chmod(0o600)
            try:
                os.link(stage, name, dst_dir_fd=descriptor, follow_symlinks=False)
                # Remove the temporary hard link while holding the directory lock,
                # so another writer never mistakes link-count ctime changes for edits.
                stage.unlink()
            except FileExistsError:
                if name == "cases.jsonl":
                    if file_digest(name, dir_fd=descriptor) != sha:
                        raise ValueError(
                            "Existing immutable dataset content changed"
                        ) from None
                elif read_small(name, 2 * 1024 * 1024, dir_fd=descriptor) != rendered:
                    raise ValueError(
                        "Existing immutable dataset manifest changed"
                    ) from None
        after = directory.lstat()
        if not stat.S_ISDIR(after.st_mode) or (after.st_dev, after.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError("Prepared dataset directory changed")
    finally:
        os.close(descriptor)
