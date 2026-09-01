"""Per-stack Grafana admin credentials for the local observability stack.

An explicit host ``GF_SECURITY_ADMIN_PASSWORD`` wins verbatim; otherwise a value
is generated and persisted per stack. The value only ever reaches the container
through a bind-mounted secret file (``GF_SECURITY_ADMIN_PASSWORD__FILE``), never
through argv, ``docker inspect .Config.Env``, or a log record.

Restart / removal: restarting keeps the same credential -- the explicit file is
rewritten idempotently, otherwise the persisted generated value is reused.
Unsetting the env stops the explicit file from being consulted; the next serve
falls back to the persisted (auto-generated) file, leaving the stale explicit
file ignored rather than deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
from secrets import token_urlsafe

from cli.commands.runtime_paths import (
    private_runtime_state_subdirectory,
    read_private_state_bytes,
    write_private_state_bytes,
)
from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack

GRAFANA_ADMIN_USER = "admin"
GRAFANA_ADMIN_PASSWORD_ENV = "GF_SECURITY_ADMIN_PASSWORD"
GRAFANA_ADMIN_PASSWORD_FILE_ENV = "GF_SECURITY_ADMIN_PASSWORD__FILE"

SECRET_TOKEN_BYTES = 32

CONTAINER_GRAFANA_PASSWORD_PATH = "/run/secrets/grafana-admin-password"

GRAFANA_CREDENTIALS_DIRECTORY = "grafana-credentials"


def grafana_admin_username() -> str:
    return GRAFANA_ADMIN_USER


def _stack_filename(base: str, stack_layout: RuntimeStackLayout) -> str:
    """Suffix the private filename with the stack, like the runtime config."""
    if stack_layout.stack_name == DEFAULT_STACK_NAME:
        return base
    return f"{base}.{stack_layout.stack_name}"


def _credentials_directory(
    state_root_dir: str | Path, stack_layout: RuntimeStackLayout
) -> Path:
    return private_runtime_state_subdirectory(
        state_root_dir, GRAFANA_CREDENTIALS_DIRECTORY
    )


def grafana_password_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Path of the persisted auto-generated password file (reused on restart)."""
    layout = stack_layout or resolve_runtime_stack()
    return _credentials_directory(state_root_dir, layout) / _stack_filename(
        "admin-password", layout
    )


def grafana_explicit_password_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Path of the file carrying the explicit env password, rewritten per serve."""
    layout = stack_layout or resolve_runtime_stack()
    return _credentials_directory(state_root_dir, layout) / _stack_filename(
        "admin-password.explicit", layout
    )


def ensure_grafana_admin_password_file(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Resolve the admin password and return an existing file ready to mount."""
    layout = stack_layout or resolve_runtime_stack()
    explicit = os.getenv(GRAFANA_ADMIN_PASSWORD_ENV)
    if explicit:
        path = grafana_explicit_password_path(state_root_dir, stack_layout=layout)
        # No trailing newline: Grafana's ``__FILE`` reader strips surrounding
        # whitespace, so the value is written exactly as the password.
        write_private_state_bytes(path, explicit.encode("utf-8"))
        return path

    path = grafana_password_path(state_root_dir, stack_layout=layout)
    stored = read_private_state_bytes(path)
    if stored is None:
        password = token_urlsafe(SECRET_TOKEN_BYTES)
        write_private_state_bytes(path, password.encode("utf-8"))
    return path


def resolve_grafana_admin_password(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> str:
    """Return the password ``ensure_..._file`` materialized, read back from disk."""
    path = ensure_grafana_admin_password_file(state_root_dir, stack_layout=stack_layout)
    stored = read_private_state_bytes(path)
    assert stored is not None, "ensure_grafana_admin_password_file leaves no file"
    return stored.decode("utf-8")
