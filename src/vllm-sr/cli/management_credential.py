"""The Router management credential of a local stack, owned by the CLI.

The Dashboard calls the Router's management API with one bearer token. A
Recipe that turns on bearer authentication binds that token, by name, as the
Router's ``dashboard_control_plane`` credential, so both containers need the
same value. ``vllm-sr serve`` creates both of them, so it owns the value: the
operator's own ``VLLM_SR_DASHBOARD_RECIPE_TOKEN`` wins, and otherwise each
stack generates one on its first start and keeps it in owner-only state below
the runtime-state directory, beside the storage credentials.

The Dashboard always receives it, the Router whenever its runtime config binds
it, both as an inherited environment name. The value never enters a
``docker`` argv list, a generated config, the Recipe store or a log record, and
the Dashboard keeps no copy on disk.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path

from cli.commands.runtime_paths import (
    private_runtime_state_subdirectory,
    read_private_state_bytes,
    write_private_state_bytes,
)
from cli.consts import DEFAULT_STACK_NAME
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)

MANAGEMENT_CREDENTIAL_SCHEMA = "vllm-sr/management-credential/v1"
MANAGEMENT_CREDENTIAL_DIRECTORY = "management-credential"
# The only shape the Dashboard accepts (dashboard/backend/recipe).
_TOKEN = re.compile(r"[0-9a-f]{64}")
TOKEN_BYTES = 32

RECOVERY_HINT = (
    "Delete the file and rerun `vllm-sr serve` to generate a new credential; "
    "serve hands it to the Router and the Dashboard it recreates."
)


class ManagementCredentialError(ValueError):
    """An operator value or a stored state the CLI will not use."""


def management_credential_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Return this stack's credential state file, in an owner-only directory."""

    layout = stack_layout or resolve_runtime_stack()
    directory = private_runtime_state_subdirectory(
        state_root_dir, MANAGEMENT_CREDENTIAL_DIRECTORY
    )
    if layout.stack_name == DEFAULT_STACK_NAME:
        return directory / "dashboard.json"
    return directory / f"dashboard.{layout.stack_name}.json"


def stack_management_credential(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> str:
    """Return the stack's management credential, generating it on first use."""

    operator_value = os.environ.get(MANAGEMENT_CREDENTIAL_ENV, "")
    if operator_value:
        if not _TOKEN.fullmatch(operator_value):
            raise ManagementCredentialError(
                f"{MANAGEMENT_CREDENTIAL_ENV} must be 64 lowercase hexadecimal "
                "characters, such as the output of `openssl rand -hex 32`"
            )
        return operator_value

    layout = stack_layout or resolve_runtime_stack()
    path = management_credential_path(state_root_dir, stack_layout=layout)
    try:
        data = read_private_state_bytes(path)
    except ValueError as error:
        raise ManagementCredentialError(f"{error}. {RECOVERY_HINT}") from error
    if data is not None:
        return _decode(data, stack_name=layout.stack_name, path=path)

    token = secrets.token_hex(TOKEN_BYTES)
    document = {
        "schema": MANAGEMENT_CREDENTIAL_SCHEMA,
        "stack": layout.stack_name,
        "token": token,
    }
    write_private_state_bytes(
        path, (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    )
    log.info(
        f"Generated the Router management credential for stack {layout.stack_name}"
    )
    return token


def _decode(data: bytes, *, stack_name: str, path: Path) -> str:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        document = None
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "stack", "token"}
        or document["schema"] != MANAGEMENT_CREDENTIAL_SCHEMA
    ):
        raise ManagementCredentialError(
            f"The management credential state is invalid: {path}. {RECOVERY_HINT}"
        )
    if document["stack"] != stack_name:
        raise ManagementCredentialError(
            f"The management credential state belongs to stack "
            f"{document['stack']}, not {stack_name}: {path}. {RECOVERY_HINT}"
        )
    token = document["token"]
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ManagementCredentialError(
            f"The stored management credential is invalid: {path}. {RECOVERY_HINT}"
        )
    return token
