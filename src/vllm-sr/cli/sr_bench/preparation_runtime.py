"""Bounded child process for downloading data without blocking the HTTP owner."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIN_FREE_BYTES = 1024**3
MAX_LOG_BYTES = 8 * 1024**2
PREPARATION_TIMEOUT = 1800
PHASES = {"checking_dependencies", "installing_dependencies", "downloading", "freezing"}


class PreparationError(ValueError):
    """A controlled error safe to show to service clients."""


def require_capacity(store):
    if shutil.disk_usage(store).free < MIN_FREE_BYTES:
        raise PreparationError(
            "Dataset preparation requires at least 1 GiB of free worker storage"
        )


def execute(request, store, progress, stopping):
    require_capacity(store)
    environment = os.environ.copy()
    # Downloaded sources, dependencies and jobs survive container replacement.
    environment["SR_BENCH_HOME"] = str(store / "preparation-runtime" / "sources")
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
                    # Only fixed worker error codes are interpreted, never its logs.
                    messages = {
                        "dependencies": "Required dataset dependencies could not be installed. Check package-index access and worker storage.",
                        "source": "Dataset download failed. Check source access, connectivity and storage; gated sources require HF_TOKEN on the service.",
                        "freeze": "The source could not be frozen for this profile. Check source availability and the selected case count.",
                    }
                    raise PreparationError(
                        messages.get(
                            value.get("error_code"),
                            "Dataset preparation worker failed. Check its runtime and retry explicitly.",
                        )
                    )
                return value["dataset"]
            finally:
                # A finished leader can still have installer/download children.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=3)
