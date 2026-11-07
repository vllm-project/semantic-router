"""Per-stack Grafana admin password for the local observability stack.

``GF_SECURITY_ADMIN_PASSWORD`` overwrites the file when it is set. Otherwise a
generated value is persisted and reused. The value reaches the container only
through a bind-mounted file (``GF_SECURITY_ADMIN_PASSWORD__FILE``).
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from secrets import token_urlsafe

from cli.commands.runtime_paths import (
    CONTAINER_READABLE_STATE_FILE_MODE,
    private_runtime_state_subdirectory,
    write_private_state_bytes,
)
from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack

GRAFANA_ADMIN_USER = "admin"
GRAFANA_ADMIN_PASSWORD_ENV = "GF_SECURITY_ADMIN_PASSWORD"
GRAFANA_ADMIN_PASSWORD_FILE_ENV = "GF_SECURITY_ADMIN_PASSWORD__FILE"

SECRET_TOKEN_BYTES = 32
MAX_PASSWORD_BYTES = 4096

CONTAINER_GRAFANA_PASSWORD_PATH = "/run/secrets/grafana-admin-password"
GRAFANA_CREDENTIALS_DIRECTORY = "grafana-credentials"


def _stack_filename(base: str, stack_layout: RuntimeStackLayout) -> str:
    if stack_layout.stack_name == DEFAULT_STACK_NAME:
        return base
    return f"{base}.{stack_layout.stack_name}"


def grafana_password_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Host path of the password file the Grafana container is mounted from."""

    layout = stack_layout or resolve_runtime_stack()
    return private_runtime_state_subdirectory(
        state_root_dir, GRAFANA_CREDENTIALS_DIRECTORY
    ) / _stack_filename("admin-password", layout)


def _read_existing_password(path: Path) -> bytes | None:
    """Read a container-readable password file without tightening its mode.

    ``read_private_state_bytes`` hardens files back to owner-only. Grafana's
    unprivileged uid needs the relaxed mode, and the private parent directory
    is what keeps the file away from other host users.
    """

    path = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(
                f"Grafana password file must not be a symbolic link: {path}"
            ) from error
        raise ValueError(f"Grafana password file cannot be opened: {path}") from error

    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Grafana password path must be a regular file: {path}")
        if info.st_uid != os.getuid():
            raise ValueError(f"Grafana password file has an unexpected owner: {path}")
        if info.st_size > MAX_PASSWORD_BYTES:
            raise ValueError(f"Grafana password file is too large: {path}")
        stored = os.read(fd, MAX_PASSWORD_BYTES)
    finally:
        os.close(fd)
    if not stored or b"\n" in stored or b"\x00" in stored:
        return None
    return stored


def ensure_grafana_admin_password_file(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Materialize the admin password into a container-readable file to mount.

    Grafana runs as an unprivileged uid and cannot read a bind-mounted 0600
    file. Privacy comes from the enclosing owner-only directory.
    """

    layout = stack_layout or resolve_runtime_stack()
    path = grafana_password_path(state_root_dir, stack_layout=layout)
    explicit = os.getenv(GRAFANA_ADMIN_PASSWORD_ENV)
    if explicit is not None:
        if explicit == "" or any(char in explicit for char in "\r\n\x00"):
            raise ValueError(
                f"{GRAFANA_ADMIN_PASSWORD_ENV} must be a single-line password"
            )
        stored = explicit.encode("utf-8")
    else:
        stored = _read_existing_password(path)
        if stored is None:
            stored = token_urlsafe(SECRET_TOKEN_BYTES).encode("utf-8")
    write_private_state_bytes(path, stored, mode=CONTAINER_READABLE_STATE_FILE_MODE)
    return path
