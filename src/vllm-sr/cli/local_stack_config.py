"""The document `vllm-sr config apply` sends to the local stack's Router.

`vllm-sr serve` never hands the Router the source file as written: it adds the
stack's own wiring -- a management listener the published port reaches, the
endpoints of the stack's service and store containers, tracing, knowledge-base
paths -- and the platform and algorithm defaults the stack was started with.
`config plan` and `config apply` send a source file with that same wiring when
they talk to a local stack, so the Router keeps every connection `serve` made,
and a later `serve` of the same file finds the document it would have written.

A change the running Router can't take (it answers `RESTART_REQUIRED`) is
saved the way the Dashboard saves one: the stack's runtime config holds it and
a pending activation says that the next `vllm-sr serve` applies it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cli.bootstrap import is_setup_mode_config
from cli.commands.runtime_paths import (
    _runtime_config_filename,
    materialize_runtime_config,
    resolve_state_root_dir,
)
from cli.commands.runtime_support import (
    RUNTIME_ALGORITHM_OVERRIDE_ENV,
    build_effective_config_bytes,
)
from cli.container_mounts import (
    ContainerMountsUnavailableError,
    inspect_container_mounts,
)
from cli.container_runtime import get_container_runtime
from cli.pending_activation import (
    ORIGIN_CLI,
    REASON_RESTART,
    record_pending_activation,
)
from cli.runtime_config_lock import acquire_runtime_config_lock
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack

CONTAINER_STATE_DIR = "/app/.vllm-sr"
_INSPECT_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class LocalStack:
    """A local stack's Router container and how `serve` started it."""

    layout: RuntimeStackLayout
    state_dir: Path
    platform: str | None
    algorithm: str | None

    def owns(self, config_path: Path) -> bool:
        """Whether config_path is a source of this stack (its state lives here)."""

        state_root = Path(resolve_state_root_dir(str(config_path))).resolve()
        return (state_root / ".vllm-sr").resolve() == self.state_dir.resolve()

    def runtime_config(self) -> Path:
        return self.state_dir / _runtime_config_filename(self.layout.stack_name)


def local_stack() -> LocalStack | None:
    """The local stack whose Router container exists, or None."""

    layout = resolve_runtime_stack()
    try:
        mounts = inspect_container_mounts(layout.router_container_name)
    except (ContainerMountsUnavailableError, OSError, SystemExit):
        return None
    state_dir = next(
        (
            mount.get("Source")
            for mount in mounts
            if mount.get("Destination") == CONTAINER_STATE_DIR and mount.get("Source")
        ),
        None,
    )
    if not state_dir:
        return None
    env = _container_env(layout.router_container_name)
    return LocalStack(
        layout=layout,
        state_dir=Path(state_dir),
        platform=env.get("VLLM_SR_PLATFORM") or env.get("DASHBOARD_PLATFORM") or None,
        algorithm=env.get(RUNTIME_ALGORITHM_OVERRIDE_ENV) or None,
    )


def stack_document(stack: LocalStack, config_path: Path) -> str:
    """The source file as `vllm-sr serve` would hand it to this stack's Router."""

    return build_effective_config_bytes(
        config_path,
        stack.algorithm,
        is_setup_mode_config(config_path),
        stack.platform,
    ).decode("utf-8")


def save_pending_restart(stack: LocalStack, config_path: Path, detail: str) -> Path:
    """Save a change that needs a restart for the next `vllm-sr serve`.

    The runtime config holds the document `serve` would materialize from
    config_path, with the receipt `serve` writes, so `serve` of the same file
    applies it as its own.
    """

    state_root = Path(resolve_state_root_dir(str(config_path)))
    runtime_config = stack.runtime_config()
    lock = acquire_runtime_config_lock(
        runtime_config_path=runtime_config,
        state_root_dir=state_root,
        stack_name=stack.layout.stack_name,
        timeout_seconds=0,
    )
    try:
        document = build_effective_config_bytes(
            config_path,
            stack.algorithm,
            is_setup_mode_config(config_path),
            stack.platform,
        )
        materialize_runtime_config(
            config_path,
            document,
            state_root_dir=state_root,
            stack_name=stack.layout.stack_name,
            replace_active=True,
        )
        record_pending_activation(
            runtime_config, document, REASON_RESTART, detail, origin=ORIGIN_CLI
        )
    finally:
        lock.close()
    return runtime_config


def _container_env(container_name: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                get_container_runtime(),
                "inspect",
                "--format",
                "{{json .Config.Env}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_INSPECT_TIMEOUT_SECONDS,
        )
        entries = json.loads(result.stdout or "null") or []
    except (OSError, subprocess.SubprocessError, ValueError, SystemExit):
        return {}
    env = {}
    for entry in entries:
        if isinstance(entry, str) and "=" in entry:
            name, value = entry.split("=", 1)
            env[name] = value
    return env
