#!/usr/bin/env python3
"""Hold process-scoped test resource locks while one command runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

MAX_TCP_PORT = 65535
LOCK_ENV = "VLLM_SR_TEST_LOCK_FDS"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Superset of the maintained local CLI and memory fixture host listeners.
STACK_PORTS = (
    3000,
    4318,
    5432,
    6379,
    8000,
    8080,
    8700,
    8888,
    9090,
    9091,
    9190,
    16686,
    19530,
    50051,
)


class ResourceLocks:
    """Locks survive nested make/shell invocations, without a persistent lease registry."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            root or Path(tempfile.gettempdir()) / f"vllm-sr-test-locks-{os.getuid()}"
        )
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise ValueError("test lock directory must be owned by the current user")
        self.fds: dict[str, int] = {}
        self.owned: set[str] = set()
        for name, descriptor in json.loads(os.getenv(LOCK_ENV, "{}")).items():
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
                continue
            try:
                stat = os.fstat(descriptor)
                path_stat = (self.root / name).stat()
                if (stat.st_dev, stat.st_ino) == (path_stat.st_dev, path_stat.st_ino):
                    self.fds[name] = descriptor
            except OSError:
                pass

    def acquire(self, name: str, timeout: float) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError(f"invalid resource name: {name}")
        if name in self.fds:
            return
        fd = os.open(self.root / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        deadline = time.monotonic() + timeout
        announced_wait = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(f"test resource is in use: {name}") from None
                if not announced_wait:
                    print(
                        f"Waiting for test resource: {name}",
                        file=sys.stderr,
                        flush=True,
                    )
                    announced_wait = True
                time.sleep(0.1)
        self.fds[name] = fd
        self.owned.add(name)

    def release(self, names: set[str]) -> None:
        for name in names & self.owned:
            os.close(self.fds.pop(name))
            self.owned.remove(name)

    def close(self) -> None:
        self.release(set(self.owned))

    def environment(self) -> dict[str, str]:
        return {**os.environ, LOCK_ENV: json.dumps(self.fds)}


def reserve_stack_ports(locks: ResourceLocks, explicit: str | None) -> int:
    candidates = (
        [int(explicit)] if explicit else random.sample(range(500, 14500, 100), 140)
    )
    for offset in candidates:
        if offset < 0 or max(STACK_PORTS) + offset > MAX_TCP_PORT:
            raise ValueError("port offset puts a stack listener outside 1..65535")
        before = set(locks.owned)
        try:
            for port in sorted(base + offset for base in STACK_PORTS):
                locks.acquire(f"tcp-{port}", 0)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
            return offset
        except (TimeoutError, OSError):
            locks.release(set(locks.owned) - before)
    raise RuntimeError("no available port set for this test stack")


def run_command(
    command: list[str],
    environment: dict[str, str],
    locks: ResourceLocks,
    log_path: Path | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        env=environment,
        pass_fds=tuple(locks.fds.values()),
        stdout=subprocess.PIPE if log_path else None,
        stderr=subprocess.STDOUT if log_path else None,
        text=True,
    )

    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if log_path and process.stdout:
            with log_path.open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
        code = process.wait()
        return 128 - code if code < 0 else code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def container_runtime() -> str:
    return os.getenv("CONTAINER_RUNTIME") or (
        "docker" if shutil.which("docker") else "podman"
    )


def require_empty_stack(stack: str) -> None:
    runtime = container_runtime()
    result = subprocess.run(
        [runtime, "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    prefix = "vllm-sr-" if stack == "vllm-sr" else f"{stack}-vllm-sr-"
    existing = [name for name in result.stdout.splitlines() if name.startswith(prefix)]
    if existing:
        raise RuntimeError(
            f"refusing to clean or replace an existing test stack: {', '.join(existing)}"
        )


def make_dry_run(flags: str) -> bool:
    """Accept exported options and make's normalized, dashless short options."""
    lexer = shlex.shlex(flags, posix=True)
    lexer.whitespace_split = True
    lexer.quotes = ""  # MAKEFLAGS escapes spaces; quotes are literal characters.
    lexer.commenters = ""
    skip_argument = False
    for index, option in enumerate(lexer):
        if skip_argument:
            skip_argument = False
            continue
        if option == "--":
            break  # The remaining words are make variable assignments.
        if option in {"--dry-run", "--just-print", "--recon"}:
            return True
        if option.startswith("--") or "=" in option:
            skip_argument = option in {
                "--directory",
                "--file",
                "--makefile",
                "--include-dir",
                "--old-file",
                "--assume-old",
                "--new-file",
                "--assume-new",
                "--what-if",
                "--eval",
            }
            continue
        if not option.startswith("-") and index != 0:
            continue
        short_options = option.removeprefix("-")
        for position, flag in enumerate(short_options):
            if flag == "n":
                return True
            if flag not in "bBdeikLmnpqrRsStvw":
                skip_argument = flag in "CEfIoW" and position == len(short_options) - 1
                break  # An option argument must not be interpreted as flags.
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--isolate-stack", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    # Recursive make recipes still execute under -n: preserve dry-run semantics.
    if make_dry_run(os.getenv("MAKEFLAGS", "")):
        return subprocess.run(command, check=False).returncode
    locks = ResourceLocks()
    log_path = None
    try:
        for resource in sorted(set(args.resource)):
            locks.acquire(resource, args.timeout)
        environment = locks.environment()
        if args.isolate_stack:
            run_id = environment.get("VLLM_SR_RUN_ID") or uuid.uuid4().hex[:16]
            stack = environment.get("VLLM_SR_STACK_NAME") or f"test-{run_id}"
            locks.acquire(f"stack-{stack}", args.timeout)
            require_empty_stack(stack)
            offset = reserve_stack_ports(locks, environment.get("VLLM_SR_PORT_OFFSET"))
            output = Path(
                environment.get("VLLM_SR_TEST_OUTPUT_DIR")
                or REPO_ROOT / ".agent-harness" / "runs" / run_id
            ).absolute()
            output.mkdir(parents=True, exist_ok=False)
            log_path = output / "command.log"
            environment.update(locks.environment())
            environment.update(
                {
                    "VLLM_SR_RUN_ID": run_id,
                    "VLLM_SR_TEST_ISOLATED": "1",
                    "VLLM_SR_TEST_OUTPUT_DIR": str(output),
                    "CONTAINER_RUNTIME": container_runtime(),
                    "VLLM_SR_STACK_NAME": stack,
                    "VLLM_SR_PORT_OFFSET": str(offset),
                    "VLLM_SR_STATE_ROOT_DIR": str(output / "state"),
                    "DOCKER_TAG": environment.get("DOCKER_TAG") or f"test-{run_id}",
                    "MEMORY_OUTPUT_DIR": environment.get("MEMORY_OUTPUT_DIR")
                    or str(output),
                }
            )
            print(
                f"Test stack {stack}, port offset {offset}, artifacts {output}",
                flush=True,
            )
        return run_command(command, environment, locks, log_path)
    finally:
        locks.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, TimeoutError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
