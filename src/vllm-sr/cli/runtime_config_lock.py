"""Cross-process coordination for local compiled-bootstrap deployment."""

from __future__ import annotations

import os
import stat
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - managed packages are Linux-only today.
    fcntl = None  # type: ignore[assignment]


LOCK_FILENAME = "compiled-bootstrap.lock"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
COORDINATION_PARENT_TRAVERSE_BITS = stat.S_IXGRP | stat.S_IXOTH
LOCK_FILE_OTHER_ACCESS_BITS = stat.S_IRWXO


class CompiledBootstrapLockError(RuntimeError):
    """Raised when the local bootstrap deployment lock cannot be acquired safely."""


@dataclass
class CompiledBootstrapLock:
    """Owned file-lock token passed through the complete Docker deployment."""

    compiled_bootstrap_path: Path
    coordination_dir: Path
    stack_name: str
    _directory_fd: int
    _lock_fd: int
    _closed: bool = False

    def __enter__(self) -> CompiledBootstrapLock:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._lock_fd < 0:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._lock_fd)
            finally:
                os.close(self._directory_fd)

    def assert_matches(
        self,
        *,
        compiled_bootstrap_path: str | Path,
        state_root_dir: str | Path,
        stack_name: str,
    ) -> None:
        expected_bootstrap = Path(compiled_bootstrap_path).expanduser().absolute()
        expected_directory = _runtime_coordination_dir(state_root_dir, stack_name)
        if self._closed or self.compiled_bootstrap_path != expected_bootstrap:
            raise CompiledBootstrapLockError(
                "The compiled-bootstrap lock does not match this deployment."
            )
        if self.coordination_dir != expected_directory or self.stack_name != stack_name:
            raise CompiledBootstrapLockError(
                "The compiled-bootstrap lock does not match this stack."
            )


def acquire_compiled_bootstrap_lock(
    *,
    compiled_bootstrap_path: str | Path,
    state_root_dir: str | Path,
    stack_name: str,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> CompiledBootstrapLock:
    """Acquire the stack-scoped lock used by local bootstrap compilation."""

    if timeout_seconds < 0:
        raise ValueError("Compiled-bootstrap lock timeout must not be negative")

    bootstrap_path = Path(compiled_bootstrap_path).expanduser().absolute()
    if fcntl is None:
        return CompiledBootstrapLock(
            compiled_bootstrap_path=bootstrap_path,
            coordination_dir=_runtime_coordination_dir(state_root_dir, stack_name),
            stack_name=stack_name,
            _directory_fd=-1,
            _lock_fd=-1,
        )
    coordination_dir, directory_fd = _open_runtime_coordination_dir(
        state_root_dir, stack_name
    )
    lock_fd = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(LOCK_FILENAME, flags, 0o600, dir_fd=directory_fd)
        except OSError as error:
            raise CompiledBootstrapLockError(
                "The compiled-bootstrap lock cannot be opened safely."
            ) from error
        info = os.fstat(lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CompiledBootstrapLockError(
                "The compiled-bootstrap lock must be a private regular file."
            )
        if stat.S_IMODE(info.st_mode) & LOCK_FILE_OTHER_ACCESS_BITS:
            raise CompiledBootstrapLockError(
                "The compiled-bootstrap lock must not grant access to other users."
            )
        os.set_inheritable(lock_fd, False)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise CompiledBootstrapLockError(
                        "Another local compiled-bootstrap operation is in progress."
                    ) from error
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return CompiledBootstrapLock(
            compiled_bootstrap_path=bootstrap_path,
            coordination_dir=coordination_dir,
            stack_name=stack_name,
            _directory_fd=directory_fd,
            _lock_fd=lock_fd,
        )
    except Exception:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(directory_fd)
        raise


def _runtime_coordination_dir(state_root_dir: str | Path, stack_name: str) -> Path:
    return (
        Path(state_root_dir).expanduser().absolute()
        / ".vllm-sr"
        / "runtime-locks"
        / stack_name
    )


def _open_runtime_coordination_dir(
    state_root_dir: str | Path, stack_name: str
) -> tuple[Path, int]:
    if (
        not stack_name
        or stack_name in {".", ".."}
        or "/" in stack_name
        or "\\" in stack_name
    ):
        raise CompiledBootstrapLockError("The runtime stack identity is invalid.")
    state_root = Path(state_root_dir).expanduser().absolute()
    descriptor = _open_real_directory(state_root)
    current_path = state_root
    try:
        components = (
            (".vllm-sr", 0),
            ("runtime-locks", COORDINATION_PARENT_TRAVERSE_BITS),
            (stack_name, 0),
        )
        for component, required_mode_bits in components:
            next_descriptor = _open_or_create_child_directory(
                descriptor,
                component,
                required_mode_bits=required_mode_bits,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            current_path /= component
        return current_path, descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_or_create_child_directory(
    parent_fd: int,
    name: str,
    *,
    required_mode_bits: int = 0,
) -> int:
    with suppress(FileExistsError):
        os.mkdir(name, 0o700 | required_mode_bits, dir_fd=parent_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise CompiledBootstrapLockError(
            "The local runtime coordination path must be a real directory."
        ) from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise CompiledBootstrapLockError(
            "The local runtime coordination path must be a real directory."
        )
    current_mode = stat.S_IMODE(info.st_mode)
    if current_mode & required_mode_bits != required_mode_bits:
        try:
            os.fchmod(descriptor, current_mode | required_mode_bits)
        except OSError as error:
            os.close(descriptor)
            raise CompiledBootstrapLockError(
                "The local runtime coordination path permissions cannot be "
                "prepared safely."
            ) from error
    os.set_inheritable(descriptor, False)
    return descriptor


def _open_real_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CompiledBootstrapLockError(
            "The local runtime coordination directory cannot be opened safely."
        ) from error
    os.set_inheritable(descriptor, False)
    return descriptor
