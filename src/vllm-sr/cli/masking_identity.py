"""The masking identity namespace a local `vllm-sr serve` provisions.

Masking plugins in registry mode keep one placeholder per value for good. The
Router keeps the registry itself; the CLI creates the namespace key once, in
the persistent state directory, and names the namespace in the runtime config
for masking plugins that do not choose how they number placeholders.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

LOCAL_NAMESPACE = "local"
CONTAINER_STATE_DIR = "/app/.vllm-sr"
KEY_BYTES = 32


def _masking_dir(state_dir: Path | str, stack_name: str) -> Path:
    return Path(state_dir) / "masking" / stack_name


def provision_local_masking(state_dir: Path | str, stack_name: str) -> Path:
    """Create the namespace key once and return it.

    A registry without its key means the key was lost. A new key would let new
    values take indexes that earlier files and sessions already use, so that
    is refused until the key is restored or the directory is moved aside.
    """
    directory = _masking_dir(state_dir, stack_name)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = directory / "key"
    if key.is_file():
        if key.stat().st_size != KEY_BYTES:
            raise RuntimeError(f"{key} must hold {KEY_BYTES} bytes")
        return key
    if any(directory.glob("registry.log*")):
        raise RuntimeError(
            f"{key} is missing but its masking registry is not. Restore the key "
            f"from a backup, or move {directory} aside to start a new namespace "
            "(and change the placeholder templates, so old placeholders cannot "
            "be read as new values)."
        )
    temporary = directory / "key.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(secrets.token_bytes(KEY_BYTES))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, key)
    return key


def _masking_plugins(node):
    if isinstance(node, dict):
        if node.get("type") == "masking" and isinstance(
            node.get("configuration"), dict
        ):
            yield node["configuration"]
        for value in node.values():
            yield from _masking_plugins(value)
    elif isinstance(node, list):
        for value in node:
            yield from _masking_plugins(value)


def inject_local_masking_identity(config: dict, stack_name: str) -> bool:
    """Name the local namespace for masking plugins that set neither index nor
    identity_namespace, adding the namespace when the config lacks it."""
    plugins = [
        plugin
        for plugin in _masking_plugins(config)
        if not plugin.get("index") and not plugin.get("identity_namespace")
    ]
    if not plugins:
        return False
    for plugin in plugins:
        plugin["identity_namespace"] = LOCAL_NAMESPACE
    services = config.setdefault("global", {}).setdefault("services", {})
    identity = services.setdefault("masking_identity", {})
    namespaces = identity.setdefault("namespaces", {})
    directory = f"{CONTAINER_STATE_DIR}/masking/{stack_name}"
    namespaces.setdefault(
        LOCAL_NAMESPACE,
        {
            "key_file": f"{directory}/key",
            "scope": "principal_or_installation",
            "registry": {"backend": "file", "path": f"{directory}/registry.log"},
        },
    )
    return True
