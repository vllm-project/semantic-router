"""Per-stack Grafana admin password for the local observability stack.

Host ``GF_SECURITY_ADMIN_PASSWORD`` overwrites the file; otherwise a generated
value is persisted and reused. The value reaches the container only through a
bind-mounted file (``GF_SECURITY_ADMIN_PASSWORD__FILE``).
"""

from __future__ import annotations

import os
from pathlib import Path
from secrets import token_urlsafe

from cli.commands.runtime_paths import (
    CONTAINER_READABLE_STATE_FILE_MODE,
    private_runtime_state_subdirectory,
    read_container_readable_state_bytes,
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


def ensure_grafana_admin_password_file(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Materialize the admin password into a container-readable file to mount.

    Grafana runs as an unprivileged uid and cannot read a bind-mounted 0600
    file; privacy comes from the enclosing owner-only directory.
    """

    layout = stack_layout or resolve_runtime_stack()
    path = grafana_password_path(state_root_dir, stack_layout=layout)
    explicit = os.getenv(GRAFANA_ADMIN_PASSWORD_ENV)
    if explicit:
        stored = explicit.encode("utf-8")
    else:
        stored = read_container_readable_state_bytes(path)
        if stored is None:
            stored = token_urlsafe(SECRET_TOKEN_BYTES).encode("utf-8")
    # Rewriting every serve restores the value and its container-readable mode.
    write_private_state_bytes(path, stored, mode=CONTAINER_READABLE_STATE_FILE_MODE)
    return path
