"""Install allowlisted data dependencies in the store, then freeze one source."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
import venv
from pathlib import Path

from .sources import HF_SOURCES, prepare_dataset

ARROW_PACKAGE = "pyarrow==18.1.0"


def dependency_directory(store):
    path = store
    for component in [
        "preparation-runtime",
        sysconfig.get_platform(),
        sys.implementation.cache_tag,
    ]:
        path = path / component
        if path.is_symlink():
            raise ValueError("Dependency cache must not contain symlinks")
        path.mkdir(exist_ok=True, mode=0o700)
    return path


def verify_arrow(path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import pyarrow.parquet",
            str(path),
        ],
        check=True,
        timeout=30,
    )


def install_arrow(parent, directory):
    if importlib.util.find_spec("pip") is not None:
        python = sys.executable
    else:
        # uv-created environments need not contain pip. Bootstrap an installer
        # under the store, never mutate the CLI or system interpreter.
        installer = parent / "installer"
        if installer.is_symlink():
            raise ValueError("Dependency installer must not be a symlink")
        if (installer / "bin" / "python").is_file():
            try:
                subprocess.run(
                    [str(installer / "bin" / "python"), "-m", "pip", "--version"],
                    check=True,
                    timeout=30,
                )
            except (OSError, subprocess.CalledProcessError):
                installer.rename(parent / (".invalid-installer-" + uuid.uuid4().hex))
        if not (installer / "bin" / "python").is_file():
            venv.EnvBuilder(with_pip=True).create(installer)
        python = str(installer / "bin" / "python")
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--target",
            directory,
            ARROW_PACKAGE,
        ],
        check=True,
        timeout=600,
        stdin=subprocess.DEVNULL,
    )


def emit(path, phase, **values):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"phase": phase, **values}))
    os.replace(temporary, path)


def ensure_dependencies(benchmark, store, progress):
    if benchmark not in HF_SOURCES and shutil.which("git") is None:
        raise ValueError("Git is required by the worker image for pinned task sources")
    if benchmark not in {"mmlu-pro", "hle"}:
        return
    if importlib.util.find_spec("pyarrow") is not None:
        return
    parent = dependency_directory(store)
    destination = parent / "pyarrow-18.1.0"
    if destination.is_symlink():
        raise ValueError("Dependency cache must not be a symlink")
    if destination.exists():
        try:
            verify_arrow(destination)
        except subprocess.CalledProcessError:
            destination.rename(parent / (".invalid-pyarrow-" + uuid.uuid4().hex))
    if not destination.exists():
        progress("installing_dependencies")
        with tempfile.TemporaryDirectory(
            prefix=".dependencies-", dir=parent
        ) as directory:
            install_arrow(parent, directory)
            verify_arrow(directory)
            os.rename(directory, destination)
    sys.path.insert(0, str(destination))
    importlib.invalidate_caches()
    importlib.import_module("pyarrow.parquet")


def run(store, state, request):
    phase = "checking_dependencies"

    def progress(value):
        nonlocal phase
        phase = value
        emit(state, phase)

    try:
        progress(phase)
        ensure_dependencies(request["benchmark"], store, progress)
        dataset = prepare_dataset(**request, store=store, progress=progress)
        emit(state, "completed", dataset=dataset)
        return 0
    except Exception:
        code = (
            "dependencies"
            if phase in {"checking_dependencies", "installing_dependencies"}
            else "source" if phase == "downloading" else "freeze"
        )
        emit(state, "failed", error_code=code)
        return 1


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])))
