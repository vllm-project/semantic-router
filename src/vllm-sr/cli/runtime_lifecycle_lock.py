"""Cross-process coordination for local container stack lifecycle operations."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - native Windows local serve is unsupported.
    fcntl = None  # type: ignore[assignment]

from cli.consts import SUPPORTED_CONTAINER_RUNTIMES
from cli.runtime_stack import normalize_stack_name

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


class RuntimeLifecycleLockError(RuntimeError):
    """Raised when a local stack lifecycle operation cannot be serialized."""


@dataclass
class RuntimeLifecycleLock:
    """Owned file-lock token for one container runtime and stack identity."""

    runtime: str
    stack_name: str
    lock_path: Path
    _directory_fd: int
    _lock_fd: int
    _closed: bool = False

    def __enter__(self) -> RuntimeLifecycleLock:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if fcntl is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._lock_fd)
            finally:
                os.close(self._directory_fd)


def acquire_runtime_lifecycle_lock(
    *,
    runtime: str,
    stack_name: str,
    timeout_seconds: float = 0.0,
    lock_root: str | Path | None = None,
) -> RuntimeLifecycleLock:
    """Acquire the host-wide mutation lock for one local runtime stack."""

    if fcntl is None:
        raise RuntimeLifecycleLockError(
            "Local runtime lifecycle coordination requires a POSIX host."
        )
    if timeout_seconds < 0:
        raise ValueError("Runtime lifecycle lock timeout must not be negative")

    normalized_runtime = Path(str(runtime).strip()).name.lower()
    if normalized_runtime not in SUPPORTED_CONTAINER_RUNTIMES:
        raise ValueError(f"Unsupported container runtime: {runtime}")
    normalized_stack = normalize_stack_name(stack_name)
    lock_directory = (
        Path(lock_root).expanduser()
        if lock_root is not None
        else _default_lock_directory()
    )
    if not lock_directory.is_absolute():
        raise RuntimeLifecycleLockError(
            "The runtime lifecycle lock directory must be absolute."
        )

    directory_fd = _open_private_directory(lock_directory)
    lock_fd = -1
    digest = hashlib.sha256(
        f"{normalized_runtime}\0{normalized_stack}".encode()
    ).hexdigest()
    lock_name = f"{digest}.lock"
    lock_path = lock_directory / lock_name
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(lock_name, flags, _PRIVATE_FILE_MODE, dir_fd=directory_fd)
        except OSError as error:
            raise RuntimeLifecycleLockError(
                "The runtime lifecycle lock cannot be opened safely."
            ) from error
        info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
        ):
            raise RuntimeLifecycleLockError(
                "The runtime lifecycle lock must be a private regular file."
            )
        os.fchmod(lock_fd, _PRIVATE_FILE_MODE)
        os.set_inheritable(lock_fd, False)
        _acquire_file_lock(
            lock_fd,
            timeout_seconds=timeout_seconds,
            runtime=normalized_runtime,
            stack_name=normalized_stack,
        )
        return RuntimeLifecycleLock(
            runtime=normalized_runtime,
            stack_name=normalized_stack,
            lock_path=lock_path,
            _directory_fd=directory_fd,
            _lock_fd=lock_fd,
        )
    except Exception:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(directory_fd)
        raise


def _acquire_file_lock(
    lock_fd: int,
    *,
    timeout_seconds: float,
    runtime: str,
    stack_name: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            if time.monotonic() >= deadline:
                raise RuntimeLifecycleLockError(
                    f"Stack {stack_name!r} on {runtime} has another lifecycle "
                    "operation in progress; wait for it to finish or interrupt "
                    "the active serve command, then retry."
                ) from error
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _default_lock_directory() -> Path:
    user_id = os.geteuid()
    linux_runtime = Path(f"/run/user/{user_id}")
    if _is_safe_owned_directory(linux_runtime):
        return linux_runtime / "vllm-sr" / "locks"

    xdg_runtime = os.getenv("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        runtime_path = Path(xdg_runtime).expanduser()
        if not runtime_path.is_absolute() or not _is_safe_owned_directory(runtime_path):
            raise RuntimeLifecycleLockError(
                "XDG_RUNTIME_DIR is not a safe current-user directory."
            )
        return runtime_path / "vllm-sr" / "locks"

    xdg_state = os.getenv("XDG_STATE_HOME", "").strip()
    state_path = (
        Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local" / "state"
    )
    if not state_path.is_absolute():
        raise RuntimeLifecycleLockError("XDG_STATE_HOME must be an absolute path.")
    return state_path / "vllm-sr" / "locks"


def _is_safe_owned_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.geteuid()
        and not (stat.S_IMODE(info.st_mode) & 0o022)
    )


def _open_private_directory(path: Path) -> int:
    """Open an absolute directory without following any path-component symlink."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path.anchor, flags)
    components = path.parts[1:]
    try:
        for index, component in enumerate(components):
            is_private_tail = index >= max(0, len(components) - 2)
            with suppress(FileExistsError):
                os.mkdir(component, _PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise RuntimeLifecycleLockError(
                    "The runtime lifecycle lock path must contain only directories."
                )
            if not _is_safe_lock_path_component(info):
                os.close(next_descriptor)
                raise RuntimeLifecycleLockError(
                    "The runtime lifecycle lock path is not safely owned."
                )
            if is_private_tail:
                if info.st_uid != os.geteuid():
                    os.close(next_descriptor)
                    raise RuntimeLifecycleLockError(
                        "The runtime lifecycle lock directory must be user-owned."
                    )
                os.fchmod(next_descriptor, _PRIVATE_DIRECTORY_MODE)
            os.close(descriptor)
            descriptor = next_descriptor
        os.set_inheritable(descriptor, False)
        return descriptor
    except RuntimeLifecycleLockError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise RuntimeLifecycleLockError(
            "The runtime lifecycle lock path cannot be opened safely."
        ) from error


def _is_safe_lock_path_component(info: os.stat_result) -> bool:
    """Accept owned private ancestors and conventional root-owned temp roots.

    Sticky root-owned temporary directories such as ``/tmp`` are safe traversal
    ancestors here: every component is opened relative to an already verified
    descriptor without following symlinks, while the final two directories must
    still be current-user-owned and are forced to mode 0700.
    """

    if info.st_uid not in {0, os.geteuid()}:
        return False
    mode = stat.S_IMODE(info.st_mode)
    if not mode & 0o022:
        return True
    return info.st_uid == 0 and bool(mode & stat.S_ISVTX)
