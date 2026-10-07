"""Discover management API credential references in Router configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.runtime_env_names import runtime_env_name_is_allowed

# The Dashboard keeps its management credential here, in the stack's Recipe
# store (dashboard/backend/recipe/activation_topology_store.go).
RECIPE_MANAGEMENT_CREDENTIAL_PATH = Path("credentials") / "router-management.token"
_MANAGEMENT_CREDENTIAL = re.compile(r"[0-9a-f]{64}")


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


def recipe_management_credential_env(
    runtime_config_path: str | Path, recipe_store_dir: str | Path
) -> dict[str, str]:
    """The Dashboard's management credential, for a Router that requires it.

    A Recipe that turns on bearer auth for the Router's management API binds
    the Dashboard's token to MANAGEMENT_CREDENTIAL_ENV in the runtime config,
    and the Dashboard keeps the token in the stack's Recipe store. A Router
    `vllm-sr serve` creates for that config gets the same token. An operator's
    own MANAGEMENT_CREDENTIAL_ENV is left to the source config's passthrough.
    """

    if MANAGEMENT_CREDENTIAL_ENV in os.environ:
        return {}
    if MANAGEMENT_CREDENTIAL_ENV not in management_credential_env_names(
        runtime_config_path
    ):
        return {}
    path = Path(recipe_store_dir) / RECIPE_MANAGEMENT_CREDENTIAL_PATH
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError(
            f"The runtime config requires the Dashboard's management credential "
            f"({MANAGEMENT_CREDENTIAL_ENV}), but the Recipe store has none: {path}"
        ) from error
    except PermissionError as error:
        raise ValueError(
            f"The Router needs the Dashboard's management credential, and this "
            f"user cannot read it: {path}. Run `vllm-sr serve` as root."
        ) from error
    if not _MANAGEMENT_CREDENTIAL.fullmatch(token):
        raise ValueError(f"The Dashboard's management credential is invalid: {path}")
    return {MANAGEMENT_CREDENTIAL_ENV: token}
