"""Host-only CLI state: controller directories, the single socket directory
Dashboard sees, and Engine mode's own workspace."""

import hashlib
import os
import re
from pathlib import Path

from cli.commands.runtime_paths import (
    _create_or_harden_private_directory,
    cli_user_share_gid,
)


def cli_state_directory() -> Path:
    """``$XDG_STATE_HOME/vllm-sr``, by default ``~/.local/state/vllm-sr``."""
    configured = os.environ.get("XDG_STATE_HOME", "")
    base = Path(configured) if configured else Path.home() / ".local" / "state"
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "vllm-sr"


def instance_control_directory(state_root: str, stack_name: str) -> Path:
    identity = f"{Path(state_root).resolve()}\0{stack_name}"
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return cli_state_directory() / "instances" / key


def prepare_instance_control_directory(state_root: str, stack_name: str) -> Path:
    directory = instance_control_directory(state_root, stack_name)
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _create_or_harden_private_directory(directory)
    socket = _create_or_harden_private_directory(directory / "socket")
    os.chown(socket, -1, cli_user_share_gid())
    os.chmod(socket, 0o750)
    return directory


def engine_workspace_directory(model: str | None = None) -> Path:
    """Where Engine mode keeps a model's config, runtime state and model cache when no --config names one.

    Each model has its own, as a directory of its own did before: a config
    keeps publishing the model it was created for, so another model needs
    another config.
    """
    name = re.sub(r"[^A-Za-z0-9._-]+", "--", model.strip("/")) if model else "default"
    directory = cli_state_directory() / "engine" / name
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return _create_or_harden_private_directory(directory)
