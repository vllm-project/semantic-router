"""Engine mode's container: `vllm-sr serve MODEL` runs `vllm-srun serve` in an image.

The model runtime ships only inside the router images, so engine mode runs it
there, in the foreground: the platform's image (`vllm-sr`, `vllm-sr-rocm` or
`vllm-sr-cuda`) with `vllm-srun` as the entrypoint. The runtime listens on the
container's port 8100, which the host publishes on `--host:--port`. One cache
directory persists what the runtime downloads and compiles across runs, and
the local package directories it serves are mounted read-only. Ctrl-C or
SIGTERM stops the container, and a stop the user asked for exits 0.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA
from cli.container_run_command import (
    append_amd_gpu_passthrough,
    append_custom_dns,
    append_env_vars,
    append_mount_specs,
    append_nvidia_gpu_passthrough,
    append_port_mappings,
)
from cli.utils import get_logger

log = get_logger(__name__)

RUNTIME_PORT = 8100
# The router image keeps models, Triton kernels and MIOpen databases under
# /app/models (its VLLM_SRUN_CACHE_DIR, TRITON_CACHE_DIR, MIOPEN_*), the layout
# a Router container's managed runtimes use too.
CACHE_MOUNT = "/app/models"
RUNTIME_CACHE = f"{CACHE_MOUNT}/model-runtime"
PACKAGES_MOUNT = "/app/packages"
CACHE_DIR_ENV = "VLLM_SR_ENGINE_CACHE_DIR"
# Hugging Face settings the runtime reads; inherited by name, so a token
# never appears in the container command.
HUB_ENV_NAMES = ("HF_TOKEN", "HF_ENDPOINT", "HF_HUB_OFFLINE")
# The accelerators the runtime has built in, and the platforms whose image
# runs each one. Other names belong to plugins, which the runtime checks.
BUILTIN_ACCELERATORS = {
    "cpu": frozenset({"", PLATFORM_AMD, PLATFORM_NVIDIA}),
    "rocm": frozenset({PLATFORM_AMD}),
    "cuda": frozenset({PLATFORM_NVIDIA}),
    "xpu": frozenset(),
    "mps": frozenset(),
}
_PLATFORM_FOR = {"rocm": PLATFORM_AMD, "cuda": PLATFORM_NVIDIA}
_DEVICE = re.compile(r"(?P<accelerator>[A-Za-z_][\w-]*)(?::\d+)?\Z")
# A container whose main process ended on the signal that stopped it.
_STOPPED = {0, 128 + signal.SIGINT, 128 + signal.SIGTERM}


@dataclass(frozen=True)
class EngineRequest:
    """What one `vllm-sr serve MODEL` serves, and where the host publishes it."""

    models: tuple[str, ...]
    models_file: str | None
    revision: str | None
    device: str
    runtime_profile: str
    host: str
    port: int
    log_level: str | None
    platform: str


def check_device(device: str, platform: str, where: str = "--device") -> None:
    """Refuse a built-in accelerator that the platform's image cannot run."""

    match = _DEVICE.fullmatch(device)
    if device == "auto" or match is None:
        return
    accelerator = match.group("accelerator").lower()
    platforms = BUILTIN_ACCELERATORS.get(accelerator)
    if platforms is None or platform in platforms:
        return
    if accelerator == "mps":
        raise ValueError(
            f"{where} {device}: mps needs macOS's Metal, which no Linux container "
            "gets; engine mode runs on the CPU there "
            "(https://github.com/vllm-project/semantic-router/issues/4636)"
        )
    if not platforms:
        raise ValueError(
            f"{where} {device}: no router image runs {accelerator}; use cpu, or "
            "rocm or cuda with --platform amd or nvidia"
        )
    raise ValueError(
        f"{where} {device} needs --platform {_PLATFORM_FOR[accelerator]}, whose "
        f"image runs {accelerator}"
    )


def engine_cache_dir() -> Path:
    """The host directory that persists the runtime's downloads across runs."""

    configured = os.getenv(CACHE_DIR_ENV, "").strip()
    if configured:
        directory = Path(configured).expanduser()
    else:
        cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
        root = Path(cache_home) if cache_home else Path.home() / ".cache"
        directory = root / "vllm-sr" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory.absolute()


@dataclass
class PackageMounts:
    """Local package directories, each mounted read-only once."""

    mounts: dict[str, str] = field(default_factory=dict)

    def container_path(self, model: str) -> str | None:
        """Where the container sees *model*, or None when it is not a local directory."""

        path = Path(model).expanduser()
        if not path.is_dir():
            return None
        host = os.path.abspath(path)
        if host not in self.mounts:
            name = os.path.basename(host) or "package"
            self.mounts[host] = f"{PACKAGES_MOUNT}/{len(self.mounts)}/{name}"
        return self.mounts[host]

    def specs(self) -> list[str]:
        return [f"{host}:{inside}:ro,z" for host, inside in self.mounts.items()]


def _container_models(models: Sequence[str], packages: PackageMounts) -> list[str]:
    return [packages.container_path(model) or model for model in models]


def _container_models_file(
    source: str, packages: PackageMounts, platform: str, directory: Path
) -> Path:
    """A copy of the --models file whose local packages name their mounts.

    Only `model` paths change; the runtime's loader stays the judge of the
    rest, so a file it would refuse is copied unchanged.
    """

    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"--models {source}: {error.strerror}") from error
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        document = None
    entries = document.get("models") if isinstance(document, dict) else None
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            device = entry.get("device")
            if isinstance(device, str):
                check_device(device, platform, f"{source}: models[{index}].device")
            model = entry.get("model")
            if isinstance(model, str):
                entry["model"] = packages.container_path(model) or model
        text = yaml.safe_dump(document, sort_keys=False)
    copy = directory / "models.yaml"
    copy.write_text(text, encoding="utf-8")
    return copy


def container_name(port: int) -> str:
    """One engine container per published port."""

    return f"vllm-sr-engine-{port}"


def engine_command(
    runtime: str,
    image: str,
    request: EngineRequest,
    *,
    cache_dir: Path,
    packages: PackageMounts,
    models_file: Path | None,
) -> list[str]:
    """The foreground `run` of one engine container."""

    arguments = runtime_arguments(
        request,
        models=_container_models(request.models, packages),
        models_file=f"{PACKAGES_MOUNT}/models.yaml" if models_file else None,
    )
    command = [
        runtime,
        "run",
        "--rm",
        "--init",
        "--name",
        container_name(request.port),
    ]
    if request.platform == PLATFORM_AMD:
        append_amd_gpu_passthrough(command, PLATFORM_AMD)
    elif request.platform == PLATFORM_NVIDIA:
        append_nvidia_gpu_passthrough(command, runtime)
    append_custom_dns(command)
    append_port_mappings(command, [(request.host, request.port, RUNTIME_PORT)])
    mounts = [f"{cache_dir}:{CACHE_MOUNT}:z", *packages.specs()]
    if models_file is not None:
        mounts.append(f"{models_file}:{PACKAGES_MOUNT}/models.yaml:ro,z")
    append_mount_specs(command, mounts)
    inherited = {name for name in HUB_ENV_NAMES if os.environ.get(name)}
    append_env_vars(command, dict.fromkeys(sorted(inherited), ""), inherited)
    return [*command, "--entrypoint", "vllm-srun", image, *arguments]


def runtime_arguments(
    request: EngineRequest, *, models: Sequence[str], models_file: str | None
) -> list[str]:
    """`vllm-srun serve` arguments as the container sees them."""

    arguments = ["serve", *models]
    if models_file:
        arguments += ["--models", models_file]
    arguments += ["--device", request.device, "--profile", request.runtime_profile]
    if request.revision:
        arguments += ["--revision", request.revision]
    arguments += ["--host", "0.0.0.0", "--port", str(RUNTIME_PORT)]
    arguments += ["--cache-dir", RUNTIME_CACHE]
    if request.log_level:
        arguments += ["--log-level", request.log_level]
    return arguments


def _ensure_port_is_free(runtime: str, name: str, port: int) -> None:
    """Remove a stopped leftover of this port's container; refuse a running one."""

    result = subprocess.run(
        [runtime, "container", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    if result.stdout.strip() == "true":
        raise ValueError(
            f"engine container {name} already serves port {port}; stop it "
            f"({runtime} stop {name}) or pass another --port"
        )
    subprocess.run([runtime, "rm", "-f", name], capture_output=True, check=False)


def run_foreground(command: Sequence[str]) -> int:
    """Run *command* until it exits, forwarding each SIGINT and SIGTERM once.

    The runtime CLI runs in its own session, so a terminal's Ctrl-C reaches it
    only through this process: once, whichever process group got it. A
    container that ends on the stop it was sent exits 0.
    """

    process = subprocess.Popen(list(command), start_new_session=True)
    stopping = False

    def forward(signum, _frame):
        nonlocal stopping
        stopping = True
        process.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        code = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if stopping and (code in _STOPPED or code in (-signal.SIGINT, -signal.SIGTERM)):
        return 0
    return code


def serve_engine(request: EngineRequest, *, runtime: str, image: str) -> int:
    """Run one engine container in the foreground and return its exit code.

    The request's own device is checked already (`check_device`); a --models
    file's devices are checked here, where the file is read.
    """

    cache_dir = engine_cache_dir()
    packages = PackageMounts()
    # Beside the cache, which a macOS container VM shares like the home directory.
    with tempfile.TemporaryDirectory(prefix=".models-file-", dir=cache_dir) as scratch:
        models_file = (
            _container_models_file(
                request.models_file, packages, request.platform, Path(scratch)
            )
            if request.models_file
            else None
        )
        command = engine_command(
            runtime,
            image,
            request,
            cache_dir=cache_dir,
            packages=packages,
            models_file=models_file,
        )
        name = container_name(request.port)
        _ensure_port_is_free(runtime, name, request.port)
        host = f"[{request.host}]" if ":" in request.host else request.host
        log.info(f"Engine mode: {image}, cache {cache_dir}")
        log.info(
            f"Serving on http://{host}:{request.port} "
            f"(container {name}; Ctrl-C stops it)"
        )
        return run_foreground(command)
