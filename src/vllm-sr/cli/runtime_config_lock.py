"""Cross-process coordination for local runtime configuration changes."""

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


LOCK_FILENAME = "runtime-config.lock"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0


class RuntimeConfigLockError(RuntimeError):
    """Raised when the local runtime mutation lock cannot be acquired safely."""


@dataclass
class RuntimeConfigLock:
    """Owned file-lock token passed through the complete Docker deployment."""

    runtime_config_path: Path
    store_dir: Path
    stack_name: str
    _directory_fd: int
    _lock_fd: int
    _closed: bool = False

    def __enter__(self) -> RuntimeConfigLock:
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
        runtime_config_path: str | Path,
        state_root_dir: str | Path,
        stack_name: str,
    ) -> None:
        expected_runtime = Path(runtime_config_path).expanduser().absolute()
        expected_store = _recipe_store_dir(state_root_dir, stack_name)
        if self._closed or self.runtime_config_path != expected_runtime:
            raise RuntimeConfigLockError(
                "The runtime configuration lock does not match this deployment."
            )
        if self.store_dir != expected_store or self.stack_name != stack_name:
            raise RuntimeConfigLockError(
                "The runtime configuration lock does not match this stack."
            )


def acquire_runtime_config_lock(
    *,
    runtime_config_path: str | Path,
    state_root_dir: str | Path,
    stack_name: str,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> RuntimeConfigLock:
    """Acquire the same store lock used by Dashboard and Router mutations."""

    if timeout_seconds < 0:
        raise ValueError("Runtime config lock timeout must not be negative")

    runtime_path = Path(runtime_config_path).expanduser().absolute()
    if fcntl is None:
        store_dir = _recipe_store_dir(state_root_dir, stack_name)
        if any(
            (store_dir / marker).exists()
            for marker in ("active.json", "activation-pending.json")
        ):
            raise RuntimeConfigLockError(
                "Managed Recipe runtime coordination requires a POSIX host."
            )
        return RuntimeConfigLock(
            runtime_config_path=runtime_path,
            store_dir=store_dir,
            stack_name=stack_name,
            _directory_fd=-1,
            _lock_fd=-1,
        )
    store_dir, directory_fd = _open_recipe_store_dir(state_root_dir, stack_name)
    lock_fd = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(LOCK_FILENAME, flags, 0o600, dir_fd=directory_fd)
        except OSError as error:
            raise RuntimeConfigLockError(
                "The runtime configuration lock cannot be opened safely."
            ) from error
        info = os.fstat(lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeConfigLockError(
                "The runtime configuration lock must be a private regular file."
            )
        os.fchmod(lock_fd, 0o600)
        os.set_inheritable(lock_fd, False)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeConfigLockError(
                        "Another local runtime configuration operation is in progress."
                    ) from error
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return RuntimeConfigLock(
            runtime_config_path=runtime_path,
            store_dir=store_dir,
            stack_name=stack_name,
            _directory_fd=directory_fd,
            _lock_fd=lock_fd,
        )
    except Exception:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(directory_fd)
        raise


def _recipe_store_dir(state_root_dir: str | Path, stack_name: str) -> Path:
    return (
        Path(state_root_dir).expanduser().absolute()
        / ".vllm-sr"
        / "recipe-store"
        / stack_name
    )


def _open_recipe_store_dir(
    state_root_dir: str | Path, stack_name: str
) -> tuple[Path, int]:
    if (
        not stack_name
        or stack_name in {".", ".."}
        or "/" in stack_name
        or "\\" in stack_name
    ):
        raise RuntimeConfigLockError("The runtime stack identity is invalid.")
    state_root = Path(state_root_dir).expanduser().absolute()
    descriptor = _open_real_directory(state_root)
    current_path = state_root
    try:
        for component in (".vllm-sr", "recipe-store", stack_name):
            next_descriptor = _open_or_create_child_directory(descriptor, component)
            os.close(descriptor)
            descriptor = next_descriptor
            current_path /= component
        return current_path, descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_or_create_child_directory(parent_fd: int, name: str) -> int:
    with suppress(FileExistsError):
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise RuntimeConfigLockError(
            "The local runtime coordination path must be a real directory."
        ) from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise RuntimeConfigLockError(
            "The local runtime coordination path must be a real directory."
        )
    os.set_inheritable(descriptor, False)
    return descriptor


def _open_real_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeConfigLockError(
            "The local runtime coordination directory cannot be opened safely."
        ) from error
    os.set_inheritable(descriptor, False)
    return descriptor
