#!/usr/bin/env python3
"""Install and run a local vLLM Semantic Router runtime for vSR, in a venv of its own.

Every command is safe to rerun: it reads what exists and does only what is missing.
`--dry-run` prints the commands without running them. Nothing here uses sudo.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import detect
import runtime

HOME = Path.home() / ".local/share/vsr"
# Pinned so every install gets the runtime vSR was tested against; to bump, test a newer main commit and paste its full SHA.
COMMIT = "579e0ca2a0b1b03580eada4213da84fe68db57c2"
PACKAGE = (
    "vllm-srun @ git+https://github.com/vllm-project/semantic-router.git"
    f"@{COMMIT}#subdirectory=src/model-runtime"
)
# That commit's lock of every runtime dependency but torch, compiled for Python 3.12; its
# numpy needs 3.11 or later, so an older Python resolves the dependencies itself.
LOCK = (
    "https://raw.githubusercontent.com/vllm-project/semantic-router/"
    f"{COMMIT}/src/model-runtime/requirements-lock.txt"
)
LOCKED_PYTHON = (3, 12)
CPU_TORCH = "https://download.pytorch.org/whl/cpu"
LABEL = "vsr-runtime"


def serve_command(
    venv: Path, model: str, device: str, port: int, dtype: str | None = None
) -> list[str]:
    command = [
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
    return command + (["--dtype", dtype] if dtype else [])


def serve_dtype(device: str, asked: str, bf16: bool | None) -> str | None:
    """The --dtype to pass: as asked, or float32 on a CPU that would emulate bfloat16 (vela3's
    published precision); None leaves the runtime's own `auto`."""
    if asked != "auto":
        return asked
    return "float32" if device == "cpu" and bf16 is False else None


def disk_short(venv: Path) -> str | None:
    """Why the disk is too full to install and download vela3-lite, or None when it fits."""
    cache = detect.model_cache()
    found = {
        "vllmSrun": (venv / "bin/vllm-srun").exists(),
        "modelCached": detect.model_cached(cache),
        "diskFreeGiB": min(detect.free_gib(venv), detect.free_gib(cache)),
    }
    return detect.disk_short(found, "cpu")


def install_steps(venv: Path, device: str, python: str) -> list[list[str]]:
    """What is still missing for a venv that can serve on `device`."""
    pip = [str(venv / "bin/python"), "-m", "pip", "install", "--upgrade"]
    steps = []
    if not (venv / "bin/python").exists():
        steps.append([python, "-m", "venv", str(venv)])
        steps.append([*pip, "pip"])

    def has(module):
        return (venv / "bin/python").exists() and subprocess.run(
            [str(venv / "bin/python"), "-c", f"import {module}"],
            capture_output=True,
            check=False,
        ).returncode == 0

    if not has("torch"):
        if device == "cpu" and platform.system() == "Linux":
            steps.append([*pip, "torch>=2.4", "--index-url", CPU_TORCH])
        else:
            steps.append([*pip, "torch>=2.4"])
    if not (venv / "bin/vllm-srun").exists():
        built_with = (
            str(venv / "bin/python") if (venv / "bin/python").exists() else python
        )
        locked = ["-c", LOCK] if _at_least(built_with, LOCKED_PYTHON) else []
        steps.append([*pip, *locked, PACKAGE])
    return steps


def _at_least(python: str, version: tuple[int, int]) -> bool:
    done = subprocess.run(
        [python, "-c", f"import sys; sys.exit(sys.version_info < {version!r})"],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


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


def start(
    venv: Path,
    model: str,
    device: str,
    port: int,
    log: Path,
    dry_run: bool,
    dtype: str | None = None,
) -> int:
    url = f"http://127.0.0.1:{port}/v1/decisions"
    if runtime.health(url)["ready"]:
        print(f"already running: {url}")
        return 0
    if port_open(port):
        print(
            f"port {port} is taken by something that is not a ready runtime (a runtime still loading?); "
            f"run `local.py wait --port {port}` or pick another --port",
            file=sys.stderr,
        )
        return 1
    command = serve_command(venv, model, device, port, dtype)
    print(f"$ {shlex.join(command)} > {log} 2>&1 &")
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
        f"started pid {process.pid}; log {log}. The first start downloads the model, so wait for it: local.py wait --port {port}"
    )
    return 0


def wait(port: int, timeout: float, log: Path) -> int:
    url = f"http://127.0.0.1:{port}/v1/decisions"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.health(url, timeout=2)["ready"]:
            print(f"ready: {url}")
            return 0
        time.sleep(3)
    tail = log.read_text(errors="replace").splitlines()[-15:] if log.exists() else []
    print(
        f"not ready after {timeout:.0f} s. Last lines of {log}:\n" + "\n".join(tail),
        file=sys.stderr,
    )
    return 1


def unit_text(kind: str, command: list[str], log: Path) -> tuple[Path, str]:
    if kind == "systemd":
        path = Path.home() / f".config/systemd/user/{LABEL}.service"
        return path, (
            "[Unit]\nDescription=vLLM Semantic Router runtime for the vSR Claude Code plugin\nAfter=network-online.target\n\n"
            f"[Service]\nExecStart={shlex.join(command)}\nRestart=on-failure\nRestartSec=5\n"
            f"StandardOutput=append:{log}\nStandardError=append:{log}\n\n[Install]\nWantedBy=default.target\n"
        )
    path = Path.home() / f"Library/LaunchAgents/local.{LABEL}.plist"
    arguments = "".join(f"    <string>{part}</string>\n" for part in command)
    return path, (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        f"  <key>Label</key><string>local.{LABEL}</string>\n"
        f"  <key>ProgramArguments</key>\n  <array>\n{arguments}  </array>\n"
        "  <key>RunAtLoad</key><true/>\n  <key>KeepAlive</key><true/>\n"
        f"  <key>StandardOutPath</key><string>{log}</string>\n  <key>StandardErrorPath</key><string>{log}</string>\n"
        "</dict>\n</plist>\n"
    )


def enable_steps(kind: str, path: Path) -> list[list[str]]:
    if kind == "systemd":
        return [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", f"{LABEL}.service"],
        ]
    domain = f"gui/{os.getuid()}"
    return [
        ["launchctl", "bootout", domain, str(path)],
        ["launchctl", "bootstrap", domain, str(path)],
    ]


def service(
    kind: str,
    venv: Path,
    model: str,
    device: str,
    port: int,
    log: Path,
    write: bool,
    dtype: str | None = None,
) -> int:
    path, text = unit_text(kind, serve_command(venv, model, device, port, dtype), log)
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
    steps = enable_steps(kind, path)
    for step in steps:
        print("$ " + shlex.join(step), flush=True)
        done = subprocess.run(step, check=False)
        if done.returncode != 0 and step[1] != "bootout":
            return done.returncode
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "start", "wait", "service"):
        one = sub.add_parser(name)
        one.add_argument("--venv", type=Path, default=HOME / "venv")
        one.add_argument("--model", default="vllm-sr/vela3-lite")
        one.add_argument(
            "--device", default="cpu", choices=("cpu", "cuda", "rocm", "auto")
        )
        one.add_argument("--port", type=int, default=8790)
        one.add_argument(
            "--dtype",
            default="auto",
            choices=("auto", "float32", "bfloat16", "float16"),
            help="auto: float32 on a CPU without native bf16, else the runtime's default",
        )
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
        one.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "install":
        steps = install_steps(args.venv, args.device, args.python)
        if not steps:
            print(f"nothing to install: {args.venv} already has torch and vllm-srun")
        short = disk_short(args.venv) if steps and args.device == "cpu" else None
        if short and not args.dry_run:
            print(f"not installing: {short}", file=sys.stderr)
            return 1
        if short:
            print(f"warning: {short}", file=sys.stderr)
        return run(steps, args.dry_run)
    if args.command == "wait":
        return wait(args.port, args.timeout, args.log)
    bf16 = detect.cpu_bf16() if args.device == "cpu" else None
    dtype = serve_dtype(args.device, args.dtype, bf16)
    if dtype == "float32" and args.dtype == "auto":
        print("this CPU has no native bf16, so the runtime computes in float32")
    if args.command == "start":
        return start(
            args.venv, args.model, args.device, args.port, args.log, args.dry_run, dtype
        )
    return service(
        args.kind,
        args.venv,
        args.model,
        args.device,
        args.port,
        args.log,
        args.write,
        dtype,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
