"""Host-only controller state and the single socket directory Dashboard sees."""

import hashlib
import os
from pathlib import Path

from cli.commands.runtime_paths import (
    _create_or_harden_private_directory,
    cli_user_share_gid,
)


def instance_control_directory(state_root: str, stack_name: str) -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "")
    base = Path(configured) if configured else Path.home() / ".local" / "state"
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    identity = f"{Path(state_root).resolve()}\0{stack_name}"
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return base / "vllm-sr" / "instances" / key


def prepare_instance_control_directory(state_root: str, stack_name: str) -> Path:
    directory = instance_control_directory(state_root, stack_name)
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _create_or_harden_private_directory(directory)
    socket = _create_or_harden_private_directory(directory / "socket")
    os.chown(socket, -1, cli_user_share_gid())
    os.chmod(socket, 0o750)
    return directory
