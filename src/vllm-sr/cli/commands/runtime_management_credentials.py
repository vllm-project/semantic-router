"""Management credential discovery for runtime bootstrap configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from cli.yaml_contract import load_yaml

from cli.runtime_env_names import runtime_env_name_is_allowed


def management_credential_env_names(config_path: str | Path | None) -> set[str]:
    """Return exact bearer-token env references from the management API schema."""

    if config_path is None:
        return set()
    try:
        document = load_yaml(Path(config_path).read_text(encoding="utf-8"))
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
