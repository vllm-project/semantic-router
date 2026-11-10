"""Host-only CLI state: controller directories, the single socket directory
Dashboard sees, and Engine mode's own workspace."""

import hashlib
import os
import re
import stat
from pathlib import Path

from cli.commands.runtime_paths import (
    _create_or_harden_private_directory,
    cli_user_share_gid,
)

# A unix socket address holds 104 bytes on macOS and 108 on Linux, NUL included.
MAX_UNIX_SOCKET_PATH_BYTES = 103
_SOCKET_NAME = "control.sock"
# Not $TMPDIR, which is itself long on macOS and on many CI hosts.
_SHORT_TEMPORARY_ROOT = Path("/tmp")


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


def instance_socket_address(directory: Path) -> str:
    """The address the controller of *directory* listens on.

    The socket lies in *directory*/socket, which the Dashboard mounts and which
    outlives a reboot. When that path is too long for a socket address, as it
    is under a long home path, the address reaches the same directory through a
    short private link in $XDG_RUNTIME_DIR or /tmp/vllm-sr-<uid>.
    """

    socket_directory = os.path.abspath(directory / "socket")
    path = Path(socket_directory) / _SOCKET_NAME
    if _fits(path):
        return str(path)
    name = hashlib.sha256(os.fsencode(socket_directory)).hexdigest()[:24]
    link = _socket_link_directory(name) / name
    address = link / _SOCKET_NAME
    if not _fits(address):
        raise ValueError(
            f"Instance socket address {address} is longer than the "
            f"{MAX_UNIX_SOCKET_PATH_BYTES} bytes a unix socket allows; set "
            "XDG_RUNTIME_DIR to a shorter private directory"
        )
    _point_link(link, socket_directory)
    return str(address)


def _fits(path: Path) -> bool:
    return len(os.fsencode(path)) <= MAX_UNIX_SOCKET_PATH_BYTES


def _socket_link_directory(name: str) -> Path:
    runtime = _private_runtime_directory()
    if runtime is not None and _fits(runtime / "vllm-sr" / name / _SOCKET_NAME):
        return _create_or_harden_private_directory(runtime / "vllm-sr")
    fallback = _SHORT_TEMPORARY_ROOT / f"vllm-sr-{os.geteuid()}"
    try:
        return _create_or_harden_private_directory(fallback)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{error}; remove {fallback} or set XDG_RUNTIME_DIR to a private "
            "directory for the instance socket"
        ) from error


def _private_runtime_directory() -> Path | None:
    """$XDG_RUNTIME_DIR when it is the private directory the spec requires."""
    configured = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not configured or not Path(configured).is_absolute():
        return None
    try:
        info = os.lstat(configured)
    except OSError:
        return None
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        return None
    return Path(configured)


def _point_link(link: Path, target: str) -> None:
    try:
        if os.readlink(link) == target:
            return
    except OSError:
        pass
    temporary = link.with_name(f".{link.name}.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.symlink(target, temporary)
    os.replace(temporary, link)


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
