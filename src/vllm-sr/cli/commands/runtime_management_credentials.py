"""Private management credentials for catalog-selected runtime sources."""

from __future__ import annotations

import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from cli.commands.runtime_paths import (
    private_runtime_state_subdirectory,
    write_runtime_config_bytes,
)
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.runtime_env_names import runtime_env_name_is_allowed

_MANAGED_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_MAX_CREDENTIAL_BYTES = 256


def management_credential_env_names(config_path: str | Path | None) -> set[str]:
    """Return exact bearer-token env references from the management API schema."""

    if config_path is None:
        return set()
    try:
        document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return set()
    if not isinstance(document, dict):
        return set()

    node: object = document
    for field in ("global", "services", "management_api", "auth"):
        if not isinstance(node, dict):
            return set()
        node = node.get(field)
    if not isinstance(node, dict):
        return set()
    tokens = node.get("tokens")
    if tokens is None:
        return set()
    if not isinstance(tokens, list):
        raise ValueError("management API auth tokens must be a list")
    names: set[str] = set()
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("env"), str):
            raise ValueError("management API auth token env name is invalid")
        name = token["env"]
        if not runtime_env_name_is_allowed(name):
            raise ValueError("management API auth token env name is invalid")
        names.add(name)
    return names


@contextmanager
def catalog_management_credential_environment(
    config_path: Path,
    *,
    state_root: Path,
    stack_name: str,
) -> Iterator[dict[str, str]]:
    """Bind catalog management credentials without exposing values in argv/logs."""

    names = management_credential_env_names(config_path)
    original = {name: os.environ.get(name) for name in names}
    bindings: dict[str, str] = {}
    try:
        for name in sorted(names):
            value = os.environ.get(name, "").strip()
            if name == MANAGEMENT_CREDENTIAL_ENV:
                value = _standard_management_credential(
                    value, state_root=state_root, stack_name=stack_name
                )
            elif not value:
                raise ValueError(
                    f"Catalog management credential environment variable is unset: {name}"
                )
            bindings[name] = value
            # Docker's secret-safe ``-e NAME`` form inherits from this process.
            os.environ[name] = value
        yield bindings
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _standard_management_credential(
    configured: str, *, state_root: Path, stack_name: str
) -> str:
    if configured:
        token = _validated_managed_token(configured)
    else:
        credential_dir = private_runtime_state_subdirectory(
            state_root, "catalog-credentials"
        )
        credential_path = credential_dir / f"{stack_name}.token"
        token = _read_private_token(credential_path)
        if token is None:
            token = secrets.token_hex(32)
            write_runtime_config_bytes(credential_path, (token + "\n").encode())
    return token


def _read_private_token(path: Path) -> str | None:
    if not path.exists():
        return None
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size > _MAX_CREDENTIAL_BYTES
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("Catalog management credential file is not private")
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("Catalog management credential file is invalid") from error
    return _validated_managed_token(value)


def _validated_managed_token(value: str) -> str:
    if not _MANAGED_TOKEN.fullmatch(value):
        raise ValueError(
            f"{MANAGEMENT_CREDENTIAL_ENV} must contain exactly 64 lowercase "
            "hexadecimal characters"
        )
    return value
