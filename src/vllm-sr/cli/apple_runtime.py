"""Stack-scoped lifecycle for the native Apple model runtime bridge."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from cli.apple_runtime_environment import cache_root, prepare_environment
from cli.commands.runtime_paths import _create_or_harden_private_directory
from cli.runtime_stack import resolve_runtime_stack
from cli.terminal import echo

ENDPOINT_ENV = "VLLM_SR_HOST_RUNTIME_ENDPOINT"
TOKEN_ENV = "VLLM_SR_HOST_RUNTIME_TOKEN"


def state_path() -> Path:
    stack = resolve_runtime_stack().stack_name
    key = hashlib.sha256(stack.encode()).hexdigest()
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "vllm-sr"
        / "apple"
        / "stacks"
        / key
        / "state.json"
    )


def read_state() -> dict | None:
    path = state_path()
    if not path.exists():
        return None
    if (
        path.is_symlink()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("Apple runtime state must be a private, user-owned file")
    return json.loads(path.read_text())


def write_state(state: dict) -> None:
    path = state_path()
    descriptor, filename = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    temporary = Path(filename)
    with os.fdopen(descriptor, "w") as output:
        json.dump(state, output)
    temporary.replace(path)


def request(
    state: dict, path: str, *, method: str = "GET", body: dict | None = None
) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{state['port']}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + state["token"],
            "Content-Type": "application/json",
        },
    )
    # Local probes must never go through the user's HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=15
    ) as response:
        return json.loads(response.read(1 << 20))


def process_identity(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip()


def _alive(state: dict) -> bool:
    pid = state.get("pid")
    return bool(
        isinstance(pid, int)
        and pid > 1
        and state.get("process_identity")
        and process_identity(pid) == state["process_identity"]
    )


def _owned_workers(state: dict) -> bool:
    """Recover children of a crashed supervisor without trusting a reused PID."""
    if not state.get("directory") or not state.get("pid"):
        return False
    result = subprocess.run(
        ["ps", "-ww", "-ax", "-o", "pgid=", "-o", "command="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    prefix = "--models " + state["directory"] + "/"
    for line in result.stdout.splitlines():
        group, _, command = line.strip().partition(" ")
        if (
            group == str(state["pid"])
            and "-m vllm_srun serve" in command
            and prefix in command
        ):
            return True
    return False


def start_bridge(
    image: str,
    *,
    engine_port: int | None = None,
    engine_mode: bool = False,
    log_level: str = "info",
) -> dict:
    """Called under the existing Docker stack lifecycle lock."""
    python = prepare_environment(image)
    stop_bridge()
    stacks = _create_or_harden_private_directory(cache_root() / "stacks")
    stack_directory = _create_or_harden_private_directory(
        stacks / state_path().parent.name
    )
    identity = secrets.token_hex(16)
    directory = _create_or_harden_private_directory(stack_directory / identity)
    cache = _create_or_harden_private_directory(cache_root() / "models")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    image_id = json.loads((python.parents[2] / "ready.json").read_text())["image"]
    state = {
        "port": port,
        "token": secrets.token_urlsafe(32),
        "image": image_id,
        "identity": identity,
        "python": str(python),
        "directory": str(directory),
        "cache": str(cache),
        "log": str(stack_directory / "runtime.log"),
        "engine_port": engine_port,
        "engine_mode": engine_mode,
        "log_level": log_level,
    }
    write_state(state)
    child = None
    try:
        # Use the CLI's interpreter for the stdlib-only supervisor. Children use
        # the managed native interpreter, not the user's Python environment.
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cli.apple_runtime_server",
                "--state",
                str(state_path()),
            ],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            },
        )
        state["pid"] = child.pid
        state["process_identity"] = process_identity(child.pid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(
                    "Apple host supervisor exited; inspect vllm-sr logs model-runtime"
                )
            try:
                answer = request(state, "/status")
                if (
                    answer["identity"] == state["identity"]
                    and answer["pid"] == child.pid
                ):
                    owned = read_state()
                    if owned is None:
                        raise RuntimeError(
                            "Apple supervisor ownership state is missing"
                        )
                    return owned
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        raise RuntimeError("Apple host supervisor did not become reachable")
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        raise


def stop_bridge() -> None:
    state = read_state()
    if state is None:
        return
    # Identity and start time are checked before every signal. A stale PID or
    # occupied port is not authority to stop another user's process.
    if _alive(state):
        try:
            answer = request(state, "/status")
            if (
                answer.get("identity") == state["identity"]
                and answer.get("pid") == state["pid"]
            ):
                request(state, "/shutdown", method="POST", body={})
        except (OSError, urllib.error.URLError):
            pass
        deadline = time.monotonic() + 15
        while _alive(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _alive(state):
            os.killpg(state["pid"], signal.SIGTERM)
            time.sleep(0.2)
        if _alive(state):
            os.killpg(state["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _alive(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _alive(state):
            raise RuntimeError(
                "Apple host process has not stopped; ownership state retained"
            )
    if _owned_workers(state):
        os.killpg(state["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while _owned_workers(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _owned_workers(state):
            os.killpg(state["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _owned_workers(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _owned_workers(state):
            raise RuntimeError(
                "Apple model workers remain alive; ownership state retained"
            )
    state_path().unlink(missing_ok=True)


def bridge_status() -> None:
    state = read_state()
    if state is None:
        echo("Model runtime: no Apple host service")
        return
    try:
        answer = request(state, "/status")
        if answer.get("identity") != state["identity"]:
            raise ValueError("host service identity changed")
        echo(
            f"Model runtime: MPS host supervisor running, {answer['processes']} process(es)"
        )
    except (OSError, ValueError, urllib.error.URLError):
        echo("Model runtime: Apple host service unavailable; inspect its logs")


def bridge_logs(follow: bool = False) -> None:
    # Logs remain available after stop removes the live ownership file.
    path = state_path().parent / "runtime.log"
    if not path.is_file():
        echo("No Apple model runtime logs found")
        return
    with path.open() as source:
        for line in source.readlines()[-200:]:
            echo(line, nl=False)
        while follow:
            line = source.readline()
            if line:
                echo(line, nl=False)
            else:
                time.sleep(0.2)
