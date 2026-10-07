"""Pending activations: saved configs that need `vllm-sr serve` to apply.

`vllm-sr config apply` records one the same way when the Router answers
`RESTART_REQUIRED`, with origin "cli".

The Dashboard holds no container runtime. When a saved config needs the
Router (and Envoy with `--gateway extproc`) created anew -- first-run setup
activating a config, or a change the running containers cannot take, such as
one the Router answers `restart_required` -- the Dashboard writes and syncs
the runtime config, then records a pending activation beside it. A
`vllm-sr serve` attached to the stack keeps a heartbeat there that says what it
does: it starts the stack, or it waits to apply an activation (first-run
setup), so the Dashboard can tell a service that is starting from a stopped one
and say whether the CLI applies the change now; the waiting CLI recreates the
containers from the saved config. Without one, the next `vllm-sr serve`
applies it, since it serves the saved runtime config like any other start.

The Dashboard's half of both files is
`dashboard/backend/handlers/pending_activation.go`.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Beside the runtime config: runtime-config.yaml -> runtime-config.serve-heartbeat.json.
HEARTBEAT_SUFFIX = ".serve-heartbeat.json"
RECORD_SUFFIX = ".pending-activation.json"
HEARTBEAT_SECONDS = 2.0

# What the heartbeat says the CLI does.
SERVE_STARTING = "starting"
SERVE_WAITING = "waiting"

REASON_SETUP = "setup"
REASON_RESTART = "restart"

# Who saved the activation. The Dashboard's records carry no origin.
ORIGIN_DASHBOARD = "dashboard"
ORIGIN_CLI = "cli"


@dataclass(frozen=True)
class PendingActivation:
    reason: str
    detail: str = ""
    origin: str = ORIGIN_DASHBOARD

    def saved_by(self) -> str:
        """Who saved the change, as status and serve name it."""

        if self.origin == ORIGIN_CLI:
            return "with `vllm-sr config apply`"
        return "in the Dashboard"


class _WaitStoppedError(Exception):
    """SIGTERM while waiting for an activation."""


def heartbeat_file(runtime_config: str | Path) -> Path:
    return Path(runtime_config).with_suffix(HEARTBEAT_SUFFIX)


def record_file(runtime_config: str | Path) -> Path:
    return Path(runtime_config).with_suffix(RECORD_SUFFIX)


def read_pending_activation(runtime_config: str | Path) -> PendingActivation | None:
    """The activation the Dashboard recorded that no container serves yet."""

    try:
        raw = record_file(runtime_config).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        record = {}
    if not isinstance(record, dict):
        record = {}
    reason = record.get("reason")
    detail = record.get("detail")
    origin = record.get("origin")
    return PendingActivation(
        reason=reason if reason in (REASON_SETUP, REASON_RESTART) else REASON_SETUP,
        detail=detail if isinstance(detail, str) else "",
        origin=ORIGIN_CLI if origin == ORIGIN_CLI else ORIGIN_DASHBOARD,
    )


def record_pending_activation(
    runtime_config: str | Path,
    config: bytes,
    reason: str,
    detail: str = "",
    *,
    origin: str = ORIGIN_CLI,
) -> None:
    """Record that config, already written to runtime_config, waits for serve.

    The record has the Dashboard's format, so each side reads the other's.
    """

    record = {
        "reason": reason,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "origin": origin,
    }
    if detail:
        record["detail"] = detail
    target = record_file(runtime_config)
    staged = target.with_name(f".{target.name}.tmp")
    staged.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(staged, 0o644)
    os.replace(staged, target)


def clear_pending_activation(runtime_config: str | Path) -> None:
    record_file(runtime_config).unlink(missing_ok=True)


@contextmanager
def serve_heartbeat(
    runtime_config: str | Path,
    interval: float = HEARTBEAT_SECONDS,
    state: str = SERVE_STARTING,
) -> Iterator[Callable[[str], None]]:
    """Beat for the block's duration, so the Dashboard knows a CLI is attached.

    The block receives a callable that changes the state the heartbeat reports.
    """

    heartbeat = heartbeat_file(runtime_config)
    stopped = threading.Event()
    written = threading.Lock()
    current = [state]

    def set_state(value: str) -> None:
        with written:
            current[0] = value
            _beat(heartbeat, value)

    def beat() -> None:
        while not stopped.wait(interval):
            with written:
                _beat(heartbeat, current[0])

    set_state(state)
    thread = threading.Thread(target=beat, name="serve-heartbeat", daemon=True)
    thread.start()
    try:
        yield set_state
    finally:
        stopped.set()
        thread.join()
        heartbeat.unlink(missing_ok=True)


def wait_for_pending_activation(
    runtime_config: str | Path,
    *,
    dashboard_running: Callable[[], bool],
    interval: float = HEARTBEAT_SECONDS,
) -> PendingActivation | None:
    """Wait until the Dashboard records an activation.

    Returns it once it has, and None when Ctrl-C or SIGTERM ends the wait; the
    stack stays as it is then. Raises if the Dashboard exits.
    """

    def stop(_signum, _frame):
        raise _WaitStoppedError

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            pending = read_pending_activation(runtime_config)
            if pending is not None:
                return pending
            if not dashboard_running():
                raise RuntimeError(
                    "The Dashboard exited during setup; see `vllm-sr logs dashboard`"
                )
            time.sleep(interval)
    except (KeyboardInterrupt, _WaitStoppedError):
        return None
    finally:
        signal.signal(signal.SIGTERM, previous)


def _beat(path: Path, state: str) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(
        json.dumps({"pid": os.getpid(), "state": state}), encoding="utf-8"
    )
    os.replace(staged, path)
