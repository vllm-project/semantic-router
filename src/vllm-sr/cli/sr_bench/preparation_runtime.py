"""Bounded child process for downloading data without blocking the HTTP owner."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from . import setup

MIN_FREE_BYTES = 1024**3
MAX_LOG_BYTES = 8 * 1024**2
MAX_DIAGNOSTIC_BYTES = 4096
PREPARATION_TIMEOUT = 1800
PHASES = {"checking_dependencies", "installing_dependencies", "downloading", "freezing"}


SECRET_NAME = r"(?:[\w-]*(?:api[_-]?key|token|password|passwd|secret|authorization|cookie|credentials?)[\w-]*)"
SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(Bearer|Basic)\s+[^\s\"',;]+"), r"\1 <redacted>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), "<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "<redacted>"),
    (
        re.compile(r"(?i)(--?" + SECRET_NAME + r"(?:\s+|=))[^\s\"'&,;]+"),
        r"\1<redacted>",
    ),
    (re.compile(r"(://)[^/\s:@]+:[^/\s@]+@"), r"\1<redacted>@"),
    (
        re.compile(r"(?i)(" + SECRET_NAME + r"[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;]+"),
        r"\1<redacted>",
    ),
)


class PreparationError(ValueError):
    """A controlled error safe to show to service clients.

    ``diagnostic`` is a bounded, redacted operator detail. It is logged and kept
    beside the preparation journal, never returned to service clients.
    """

    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic


def redact(text):
    """Remove credential-shaped values from operator diagnostics."""
    for name, value in os.environ.items():
        if len(value) >= 8 and re.search(SECRET_NAME, name, re.IGNORECASE):
            text = text.replace(value, "<redacted>")
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def worker_diagnostic(value, log, returncode):
    """Summarize a failed worker as a bounded, redacted operator diagnostic."""
    lines = [f"worker exit code: {returncode}"]
    if value.get("error_type"):
        detail = str(value.get("error_detail", ""))[:MAX_DIAGNOSTIC_BYTES]
        lines.append(f"{value['error_type']}: {detail}")
    log.flush()
    size = os.fstat(log.fileno()).st_size
    log.seek(max(0, size - MAX_DIAGNOSTIC_BYTES))
    tail = log.read().decode("utf-8", "replace")
    if size > MAX_DIAGNOSTIC_BYTES:
        tail = tail.partition("\n")[2]
    if tail.strip():
        lines += ["worker log tail:", tail.rstrip()]
    text = redact("\n".join(lines))
    return text[-MAX_DIAGNOSTIC_BYTES:]


def require_capacity(store):
    if shutil.disk_usage(store).free < MIN_FREE_BYTES:
        raise PreparationError(
            "Dataset preparation requires at least 1 GiB of free worker storage"
        )


def execute(request, store, progress, stopping):
    require_capacity(store)
    environment = os.environ.copy()
    # Readers and evaluators must resolve the same source-backed tasks as the
    # downloader. The standalone service chooses a persistent default at startup;
    # embedded owners keep their existing setup.home() without global mutation.
    environment.setdefault("SR_BENCH_HOME", str(setup.home()))
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_root, environment.get("PYTHONPATH")])
    )
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=store) as directory:
        state = Path(directory) / "state.json"
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "cli.sr_bench.preparation_worker",
                    str(store),
                    str(state),
                    json.dumps(request),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=environment,
                start_new_session=True,
            )
            deadline, last_phase = time.monotonic() + PREPARATION_TIMEOUT, None
            try:
                while process.poll() is None:
                    if stopping.wait(0.2):
                        raise PreparationError(
                            "Service stopped during preparation. Retry explicitly."
                        )
                    if time.monotonic() > deadline:
                        raise PreparationError(
                            "Dataset preparation exceeded its 30-minute deadline"
                        )
                    require_capacity(store)
                    if os.fstat(log.fileno()).st_size > MAX_LOG_BYTES:
                        raise PreparationError(
                            "Dataset preparation exceeded its diagnostic output limit"
                        )
                    if state.exists():
                        phase = json.loads(state.read_text()).get("phase")
                        if phase in PHASES and phase != last_phase:
                            progress(phase)
                            last_phase = phase
                value = json.loads(state.read_text()) if state.exists() else {}
                if process.returncode or "dataset" not in value:
                    # Only fixed worker error codes reach clients. The redacted
                    # log tail is an operator diagnostic, never a client message.
                    messages = {
                        "dependencies": "Required dataset dependencies could not be installed. Check package-index access and worker storage.",
                        "source": "Dataset download failed. Check source access, connectivity and storage; gated sources require HF_TOKEN on the service.",
                        "freeze": "The source could not be frozen for this profile. Check source availability and the selected case count.",
                    }
                    raise PreparationError(
                        messages.get(
                            value.get("error_code"),
                            "Dataset preparation worker failed. Check its runtime and retry explicitly.",
                        ),
                        worker_diagnostic(value, log, process.returncode),
                    )
                return value["dataset"]
            finally:
                # A finished leader can still have installer/download children.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                if process.poll() is None:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
