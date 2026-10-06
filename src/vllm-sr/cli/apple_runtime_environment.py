"""Provision a native runtime from the exact Router image selected by serve.

Only the runtime's Python package and metadata cross the image boundary.
Native dependencies are installed as macOS wheels in a private environment;
Linux libraries and model-package code are never imported on the host.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

from cli.commands.runtime_paths import _create_or_harden_private_directory
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
from cli.utils import get_logger

log = get_logger(__name__)
UV_VERSION = "0.12.5"
PYTHON_VERSION = "3.12.12"

# Run inside the selected image. Query installed metadata without loading torch
# or a model. Lower-bound requirements become the image's concrete versions.
IMAGE_METADATA = """
import importlib.metadata as md, json, pathlib
from packaging.requirements import Requirement
d = md.distribution('vllm-srun')
requirements = []
for raw in d.requires or []:
    r = Requirement(raw)
    if r.marker and not r.marker.evaluate({'extra': ''}):
        continue
    requirements.append(r.name + '==' + md.version(r.name).split('+')[0])
print(json.dumps({'package': str(d.locate_file('vllm_srun')),
    'metadata': str(d._path), 'requirements': sorted(requirements),
    'version': d.version}))
"""


def validate_apple_host(target: str, runtime: str) -> None:
    """Reject an unsupported host before downloads or workspace mutation."""
    if target != "docker":
        raise ValueError("--platform apple is supported only on the Docker target")
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        raise ValueError(
            "--platform apple requires macOS on Apple silicon and an arm64 Python; "
            "run the CLI natively rather than under Rosetta"
        )
    if runtime != "docker":
        raise ValueError("--platform apple currently requires a local Docker engine")


def validate_local_docker() -> None:
    context = os.getenv("DOCKER_CONTEXT", "")
    endpoint = "" if context else os.getenv("DOCKER_HOST", "")
    if not endpoint:
        result = _run(
            ["docker", "context", "inspect", *([context] if context else [])],
            capture=True,
        )
        endpoint = json.loads(result)[0]["Endpoints"]["docker"]["Host"]
    if not endpoint.startswith("unix://"):
        raise ValueError(
            "--platform apple requires a local Docker Unix-socket context; "
            "a remote daemon cannot reach this Mac through host.docker.internal"
        )
    if (
        _run(["docker", "info", "--format", "{{.OSType}}"], capture=True).strip()
        != "linux"
    ):
        raise ValueError("Apple host bridging requires a local Linux Docker engine")


def cache_root() -> Path:
    return _create_or_harden_private_directory(
        Path.home() / "Library" / "Caches" / "vllm-sr" / "apple", parents=True
    )


def _run(command: list[str], *, capture: bool = False, timeout: int = 1800) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )
    return result.stdout or ""


def _copy_package(container: str, source: str, destination: Path) -> None:
    if not source.startswith("/") or ".." in Path(source).parts:
        raise ValueError("Router image returned an invalid runtime package path")
    _run(["docker", "cp", f"{container}:{source}", str(destination)])
    if destination.is_symlink() or any(p.is_symlink() for p in destination.rglob("*")):
        raise ValueError("Host runtime source must not contain symbolic links")


def prepare_environment(image: str) -> Path:
    """Return a cached arm64 interpreter with release-matched runtime plugins.

    Call only after normal image selection/pull policy has resolved this image.
    Installation is serialized across stacks and published only after MPS is
    usable. The user's interpreter and model cache are never modified.
    """
    image_id = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], capture=True
    ).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Apple runtime requires an immutable local Router image ID")
    identity = hashlib.sha256(
        f"{image_id}/{PYTHON_VERSION}/{UV_VERSION}/arm64-v1".encode()
    ).hexdigest()
    root = cache_root()
    with acquire_runtime_lifecycle_lock(
        runtime="docker", stack_name="apple-environment", timeout_seconds=1800
    ):
        environment = root / identity
        python = environment / "env" / "bin" / "python"
        if (environment / "ready.json").is_file():
            _check_mps(python)
            return python
        log.info("Preparing native Apple runtime from Router image %s", image_id)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as directory:
            staging = Path(directory)
            metadata = json.loads(
                _run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        image_id,
                        "-c",
                        IMAGE_METADATA,
                    ],
                    capture=True,
                )
            )
            installer = staging / "installer"
            venv.EnvBuilder(with_pip=True).create(installer)
            _run(
                [
                    str(installer / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--only-binary=:all:",
                    f"uv=={UV_VERSION}",
                ]
            )
            uv = str(installer / "bin" / "uv")
            env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(root / "python")}
            subprocess.run(
                [
                    uv,
                    "venv",
                    "--python",
                    PYTHON_VERSION,
                    "--python-preference",
                    "only-managed",
                    str(staging / "env"),
                ],
                env=env,
                check=True,
                timeout=600,
                stdin=subprocess.DEVNULL,
            )
            native_python = staging / "env" / "bin" / "python"
            requirements = metadata["requirements"]
            if not all(
                re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.-]+", r)
                for r in requirements
            ):
                raise ValueError("Router image returned invalid dependency pins")
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(native_python),
                    "--only-binary=:all:",
                    *requirements,
                ]
            )
            site = Path(
                _run(
                    [
                        str(native_python),
                        "-c",
                        "import sysconfig; print(sysconfig.get_path('purelib'))",
                    ],
                    capture=True,
                ).strip()
            )
            container = _run(["docker", "create", image_id], capture=True).strip()
            try:
                _copy_package(container, metadata["package"], site / "vllm_srun")
                _copy_package(
                    container,
                    metadata["metadata"],
                    site / Path(metadata["metadata"]).name,
                )
            finally:
                _run(["docker", "rm", container], capture=True)
            _check_mps(native_python)
            (staging / "ready.json").write_text(
                json.dumps(
                    {
                        "image": image_id,
                        "runtime": metadata["version"],
                        "dependencies": requirements,
                        "python": PYTHON_VERSION,
                    }
                )
                + "\n"
            )
            # uv's interpreter symlink remains valid after this move. Invoke the
            # environment's Python directly, never relocated console scripts.
            shutil.rmtree(installer)
            if environment.exists():
                raise ValueError(f"Incomplete Apple runtime cache: {environment}")
            staging.rename(environment)
        return python


def _check_mps(python: Path) -> None:
    _run(
        [
            str(python),
            "-c",
            "import platform,torch; "
            "assert platform.machine() == 'arm64', 'native arm64 runtime required'; "
            "assert torch.backends.mps.is_available(), 'MPS is unavailable'; "
            "x=torch.ones(1,device='mps'); assert (x+x).cpu().item()==2",
        ],
        timeout=60,
    )
