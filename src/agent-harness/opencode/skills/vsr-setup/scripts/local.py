#!/usr/bin/env python3
"""Install, upgrade and run vSR's local vLLM Semantic Router runtime, in a venv of its own.

Every command is safe to rerun: it reads what exists and does only what is missing. `--dry-run`
prints the commands without running them. Nothing here uses sudo. The paths and the service names
are the Claude Code plugin's, so either plugin finds the runtime the other installed.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
import detect
import runtime

HOME = detect.HOME
COMMIT = detect.COMMIT
REPO = "https://github.com/vllm-project/semantic-router.git"
# The multimodal extra adds Pillow, which the runtime needs to read the images redact and shield send.
PACKAGE = f"vllm-srun[multimodal] @ git+{REPO}@{COMMIT}#subdirectory=src/model-runtime"
# That commit's lock of every runtime dependency but torch, compiled for Python 3.12; its numpy
# needs 3.11 or later, so an older Python resolves the dependencies itself.
LOCK = (
    "https://raw.githubusercontent.com/vllm-project/semantic-router/"
    f"{COMMIT}/src/model-runtime/requirements-lock.txt"
)
LOCKED_PYTHON = (3, 12)
CPU_TORCH = "https://download.pytorch.org/whl/cpu"
LABEL = detect.LABEL
command_of = detect.command_of
alive = detect.alive


def started_pid() -> int | None:
    return detect.started_pid(HOME, command_of)


def serve_command(venv: Path, model: str, device: str, port: int) -> list[str]:
    # On a CPU the runtime computes in float32 by itself, so no --dtype is passed.
    return [
        str(venv / "bin/vllm-srun"),
        "serve",
        model,
        "--device",
        device,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def disk_short(venv: Path, device: str) -> str | None:
    """Why the disk is too full to install and download vela3-lite, or None when it fits."""
    cache = detect.model_cache()
    found = {
        "installed": (venv / "bin/vllm-srun").exists(),
        "modelCached": detect.model_cached(cache),
        "diskFreeGiB": min(detect.free_gib(venv), detect.free_gib(cache)),
    }
    return detect.disk_short(found, "cpu" if device == "cpu" else "gpu")


def _has(venv: Path, module: str) -> bool:
    python = venv / "bin/python"
    return (
        python.exists()
        and subprocess.run(
            [str(python), "-c", f"import {module}"],
            capture_output=True,
            timeout=60,
            check=False,
        ).returncode
        == 0
    )


def _at_least(python: str, version: tuple[int, int]) -> bool:
    try:
        done = subprocess.run(
            [python, "-c", f"import sys; sys.exit(sys.version_info < {version!r})"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def install_steps(venv: Path, device: str, python: str) -> list[list[str]]:
    """What is still missing for a venv that serves the pinned runtime on `device`."""
    pip = [str(venv / "bin/python"), "-m", "pip", "install", "--upgrade"]
    steps = []
    if not (venv / "bin/python").exists():
        steps.append([python, "-m", "venv", str(venv)])
        steps.append([*pip, "pip"])
    if not _has(venv, "torch"):
        if device == "cpu" and platform.system() == "Linux":
            # The CPU wheels save several GB of CUDA libraries.
            steps.append([*pip, "torch>=2.4", "--index-url", CPU_TORCH])
        else:
            steps.append([*pip, "torch>=2.4"])
    present = (venv / "bin/vllm-srun").exists()
    commit = detect.installed_commit(venv) if present else None
    if present and (
        commit is None or (commit != COMMIT and commit not in detect.OLDER)
    ):
        # A local build, or a commit setup does not know (the Claude Code plugin's, perhaps newer):
        # left as it is, never moved backwards.
        return steps
    if present and commit != COMMIT:
        # pip keeps an installed 0.2.0 that names another commit, so the new one is forced in first,
        # without its dependencies or the lock; the next step brings those.
        steps.append([*pip, "--force-reinstall", "--no-deps", PACKAGE])
    if not present or commit != COMMIT or not _has(venv, "PIL"):
        built_with = (
            str(venv / "bin/python") if (venv / "bin/python").exists() else python
        )
        locked = ["-c", LOCK] if _at_least(built_with, LOCKED_PYTHON) else []
        steps.append([*pip, *locked, PACKAGE])
    return steps


def run(steps: list[list[str]], dry_run: bool) -> int:
    for step in steps:
        print("$ " + shlex.join(step), flush=True)
        if not dry_run:
            done = subprocess.run(step, check=False)
            if done.returncode != 0:
                print(f"failed with exit code {done.returncode}", file=sys.stderr)
                return done.returncode
    return 0


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def url_of(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/decisions"


def stop(dry_run: bool, timeout: float = 30) -> int:
    """Stop the runtime started for this login: only a live vllm-srun the pid file names, so a pid a
    reboot handed to another process is never signalled."""
    pid_file = HOME / "runtime.pid"
    pid = started_pid()
    if pid is None:
        if pid_file.exists() and not dry_run:
            pid_file.unlink()
        print("no runtime started for this login is running")
        return 0
    print(f"$ kill {pid}")
    if dry_run:
        return 0
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(pid):
        print(
            f"pid {pid} is still running after {timeout:.0f} s; ask the user to stop it",
            file=sys.stderr,
        )
        return 1
    pid_file.unlink(missing_ok=True)
    print(f"stopped pid {pid}")
    return 0


def start(
    venv: Path, model: str, device: str, port: int, log: Path, dry_run: bool
) -> int:
    url = url_of(port)
    if runtime.health(url)["ready"]:
        print(f"already running: {url}")
        return 0
    pid = started_pid()
    if pid is not None:
        print(f"already starting: pid {pid}; wait for it: local.py wait --port {port}")
        return 0
    if port_open(port):
        print(
            f"port {port} is taken by something that is not a ready runtime (a runtime still loading?); "
            f"run `local.py wait --port {port}` or pick another --port",
            file=sys.stderr,
        )
        return 1
    command = serve_command(venv, model, device, port)
    print(f"$ {shlex.join(command)} >> {log} 2>&1 &")
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "runtime.pid").write_text(str(process.pid))
    print(
        f"started pid {process.pid}; log {log}. The first start downloads the model, so wait for it: "
        f"local.py wait --port {port}"
    )
    return 0


def exited_pid() -> int | None:
    """The pid in runtime.pid when that process is gone: a runtime started for this login that
    exited, read afresh each time so one that died before `wait` began counts too."""
    try:
        pid = int((HOME / "runtime.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return None if alive(pid) else pid


def _tail(log: Path) -> str:
    return (
        "\n".join(log.read_text(errors="replace").splitlines()[-15:])
        if log.exists()
        else ""
    )


def wait(port: int, timeout: float, log: Path) -> int:
    """Until the runtime answers, then how long a short question takes. A runtime started for this
    login that exits while loading is said at once, not after the timeout."""
    url = url_of(port)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.health(url, timeout=2)["ready"]:
            ms = runtime.time_question(url)
            print(
                f"ready: {url}" + (f" · {ms} ms a question" if ms is not None else "")
            )
            slow = runtime.slow_note(ms)
            if slow:
                print(f"warning: {slow}", file=sys.stderr)
            return 0
        exited = exited_pid()
        if exited is not None and not port_open(port):
            print(
                f"the runtime (pid {exited}) exited while loading. Last lines of {log}:\n{_tail(log)}",
                file=sys.stderr,
            )
            return 1
        time.sleep(3)
    print(
        f"not ready after {timeout:.0f} s. Last lines of {log}:\n{_tail(log)}",
        file=sys.stderr,
    )
    return 1


def unit_text(kind: str, command: list[str], log: Path) -> tuple[Path, str]:
    if kind == "systemd":
        path = Path.home() / f".config/systemd/user/{LABEL}.service"
        return path, (
            "[Unit]\nDescription=vLLM Semantic Router runtime for vSR\nAfter=network-online.target\n\n"
            f"[Service]\nExecStart={shlex.join(command)}\nRestart=on-failure\nRestartSec=5\n"
            f"StandardOutput=append:{log}\nStandardError=append:{log}\n\n[Install]\nWantedBy=default.target\n"
        )
    path = Path.home() / f"Library/LaunchAgents/local.{LABEL}.plist"
    arguments = "".join(f"    <string>{escape(part)}</string>\n" for part in command)
    return path, (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        f"  <key>Label</key><string>local.{LABEL}</string>\n"
        f"  <key>ProgramArguments</key>\n  <array>\n{arguments}  </array>\n"
        "  <key>RunAtLoad</key><true/>\n  <key>KeepAlive</key><true/>\n"
        f"  <key>StandardOutPath</key><string>{escape(str(log))}</string>\n"
        f"  <key>StandardErrorPath</key><string>{escape(str(log))}</string>\n"
        "</dict>\n</plist>\n"
    )


def enable_steps(kind: str, path: Path, changed: bool = True) -> list[list[str]]:
    """The commands that load the service. A changed systemd unit is restarted: `enable --now` only
    starts a service that is not running, so one already running would keep its old flags. launchd
    reloads it either way."""
    if kind == "systemd":
        unit = f"{LABEL}.service"
        return [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", unit],
            ["systemctl", "--user", "restart" if changed else "start", unit],
        ]
    domain = f"gui/{os.getuid()}"
    return [
        ["launchctl", "bootout", domain, str(path)],
        ["launchctl", "bootstrap", domain, str(path)],
    ]


def service(
    kind: str, venv: Path, model: str, device: str, port: int, log: Path, write: bool
) -> int:
    path, text = unit_text(kind, serve_command(venv, model, device, port), log)
    current = path.read_text() if path.exists() else None
    print(f"# {path}\n{text}")
    if not write:
        print("(not written; pass --write to install it)")
        return 0
    if current != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
    else:
        print(f"unchanged: {path}")
    for step in enable_steps(kind, path, changed=current != text):
        print("$ " + shlex.join(step), flush=True)
        done = subprocess.run(step, check=False)
        if done.returncode != 0 and step[1] != "bootout":
            return done.returncode
    return 0


def restart_plan(kept_by: str | None) -> list[list[str]] | str:
    """How to restart a runtime so it loads newly installed code: the service's own restart, or for
    one started for this login, `kill` and then `start` (the string)."""
    if kept_by == "systemd":
        return [["systemctl", "--user", "restart", f"{LABEL}.service"]]
    if kept_by == "launchd":
        return [["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/local.{LABEL}"]]
    if kept_by == "pid":
        return "kill"
    return "none"


def restart(
    venv: Path, model: str, device: str, port: int, log: Path, dry_run: bool
) -> int:
    plan = restart_plan(detect.kept_by())
    if plan == "none" and not runtime.health(url_of(port))["ready"]:
        # Nothing answers: a runtime started for this login that is gone (its pid file stale, say).
        if (HOME / "runtime.pid").exists() and not dry_run:
            (HOME / "runtime.pid").unlink()
        return start(venv, model, device, port, log, dry_run)
    if plan == "none":
        print(
            "no runtime that setup started was found (no vsr-runtime service and no pid file); "
            "ask the user to restart theirs",
            file=sys.stderr,
        )
        return 1
    if isinstance(plan, list):
        return run(plan, dry_run)
    stopped = stop(dry_run)
    if stopped != 0:
        return stopped
    if not dry_run and port_open(port):
        print(
            f"port {port} still answers after the runtime setup started stopped, so another runtime "
            "serves it; ask the user to restart theirs",
            file=sys.stderr,
        )
        return 1
    return start(venv, model, device, port, log, dry_run)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "start", "stop", "restart", "wait", "service"):
        one = sub.add_parser(name)
        one.add_argument("--venv", type=Path, default=HOME / "venv")
        one.add_argument("--model", default=detect.MODEL)
        one.add_argument(
            "--device",
            choices=("cpu", "cuda", "rocm"),
            required=name in ("start", "restart", "service"),
            default="cpu",
        )
        one.add_argument("--port", type=int, default=8790)
        one.add_argument("--log", type=Path, default=HOME / "runtime.log")
        one.add_argument(
            "--python",
            default=sys.executable,
            help="a Python 3.10+ to build the venv with",
        )
        one.add_argument("--timeout", type=float, default=900)
        one.add_argument(
            "--kind",
            choices=("systemd", "launchd"),
            default="launchd" if platform.system() == "Darwin" else "systemd",
        )
        one.add_argument("--dry-run", action="store_true")
        one.add_argument(
            "--skip-disk-check",
            action="store_true",
            help="install even when the free-space estimate says it will not fit",
        )
        one.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "install":
        steps = install_steps(args.venv, args.device, args.python)
        if not steps:
            commit = detect.installed_commit(args.venv)
            print(
                f"nothing to install: {args.venv} has torch and vllm-srun {COMMIT[:9]} with its multimodal extra"
                if commit == COMMIT
                else f"nothing to install: {args.venv} has a vllm-srun setup did not pin "
                f"({str(commit)[:9] if commit else 'a local build'}), left as it is"
            )
            return 0
        short = None if args.skip_disk_check else disk_short(args.venv, args.device)
        if short and not args.dry_run:
            print(
                f"not installing: {short} (--skip-disk-check installs anyway)",
                file=sys.stderr,
            )
            return 1
        if short:
            print(f"warning: {short}", file=sys.stderr)
        return run(steps, args.dry_run)
    if args.command == "wait":
        return wait(args.port, args.timeout, args.log)
    if args.command == "stop":
        return stop(args.dry_run)
    if args.command == "start":
        return start(
            args.venv, args.model, args.device, args.port, args.log, args.dry_run
        )
    if args.command == "restart":
        return restart(
            args.venv, args.model, args.device, args.port, args.log, args.dry_run
        )
    return service(
        args.kind, args.venv, args.model, args.device, args.port, args.log, args.write
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
