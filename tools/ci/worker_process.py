"""Run one worker command without leaking descendants into the next contract."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from pathlib import Path


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict,
    check: bool,
    timeout: float,
    stdout=None,
    stderr=None,
):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        status = process.wait(timeout=timeout)
    finally:
        # Success, failure and timeout all retire the command's private group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    result = subprocess.CompletedProcess(command, status)
    if check:
        result.check_returncode()
    return result
