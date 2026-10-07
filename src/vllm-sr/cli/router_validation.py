"""The Router's own verdict on a config file, for `vllm-sr config validate`.

The CLI's offline checks cover the document's structure and the rules the CLI
owns. The rules the Router applies when it loads a document -- cross-field
contracts such as a modality decision's models, or a module's required
settings -- live in the Router alone, so `config validate` asks the Router:
the stack's Router image, run once with `-validate-config` and no network, or
a running Router through its management API (`--endpoint`). Either way one set
of rules decides what `vllm-sr serve` and `vllm-sr config apply` accept, and
the answer carries the warnings the Router logs when it loads the document.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cli.consts import (
    VLLM_SR_CONTAINER_IMAGE_CUDA,
    VLLM_SR_CONTAINER_IMAGE_DEFAULT,
    VLLM_SR_CONTAINER_IMAGE_ROCM,
)
from cli.container_runtime import container_image_exists, get_container_runtime
from cli.router_management_client import RouterManagementClient, RouterManagementError
from cli.runtime_stack import resolve_runtime_stack

ROUTER_BINARY = "/usr/local/bin/router"
VALIDATION_TIMEOUT_SECONDS = 120
_UNKNOWN_FLAG = "flag provided but not defined: -validate-config"


@dataclass(frozen=True)
class RouterWarning:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class RouterVerdict:
    """Whether the Router would load the document, and what it would warn."""

    source: str
    valid: bool
    error: str = ""
    warnings: tuple[RouterWarning, ...] = field(default_factory=tuple)


class RouterValidationUnavailableError(Exception):
    """The Router's validation could not run; the message says why."""


def validation_image(explicit: str | None = None) -> str:
    """The Router image to validate with: the stack's own, or one present here.

    An explicit image wins. Otherwise the image of this stack's Router
    container, then the images `vllm-sr serve` would choose, the first one
    present locally. Validation never pulls an image.
    """

    if not _container_runtime_installed():
        raise RouterValidationUnavailableError(
            "no container runtime (Docker or Podman) is installed"
        )
    # Validation output stays the command's result; which runtime it found
    # is not part of it.
    detection = logging.getLogger("cli.container_runtime")
    level = detection.level
    detection.setLevel(logging.WARNING)
    try:
        return _present_image(explicit)
    except SystemExit as error:
        raise RouterValidationUnavailableError(
            "the container runtime is not reachable"
        ) from error
    finally:
        detection.setLevel(level)


def _container_runtime_installed() -> bool:
    explicit = os.getenv("CONTAINER_RUNTIME", "").strip().lower()
    names = (explicit,) if explicit else ("docker", "podman")
    return any(shutil.which(name) for name in names if name)


def _present_image(explicit: str | None) -> str:
    if explicit:
        if not container_image_exists(explicit):
            raise RouterValidationUnavailableError(
                f"the Router image {explicit} is not present locally; pull it first"
            )
        return explicit
    candidates = [
        _container_image(resolve_runtime_stack().router_container_name),
        os.getenv("VLLM_SR_ROUTER_IMAGE", "").strip(),
        os.getenv("VLLM_SR_IMAGE", "").strip(),
        VLLM_SR_CONTAINER_IMAGE_DEFAULT,
        VLLM_SR_CONTAINER_IMAGE_ROCM,
        VLLM_SR_CONTAINER_IMAGE_CUDA,
    ]
    for image in dict.fromkeys(candidate for candidate in candidates if candidate):
        if container_image_exists(image):
            return image
    raise RouterValidationUnavailableError(
        "no Router image is present locally (`vllm-sr serve` pulls "
        f"{VLLM_SR_CONTAINER_IMAGE_DEFAULT})"
    )


def validate_with_image(
    config_path: Path, image: str, *, gateway: str, models_dir: Path | None = None
) -> RouterVerdict:
    """Run the image's Router once on the file, without network access.

    The file goes in on stdin. The models directory `serve` would mount is
    mounted read-only, so local model packages resolve as they do at runtime.
    """

    command = [
        get_container_runtime(),
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--entrypoint",
        ROUTER_BINARY,
    ]
    if models_dir is not None and models_dir.is_dir():
        command += ["-v", f"{models_dir.resolve()}:/app/models:ro,z"]
    command += [
        image,
        "-validate-config",
        "-config",
        "/dev/stdin",
        "-gateway",
        gateway,
    ]
    try:
        result = subprocess.run(
            command,
            input=config_path.read_bytes(),
            capture_output=True,
            timeout=VALIDATION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RouterValidationUnavailableError(
            f"the Router image {image} could not run: {error}"
        ) from error
    stderr = result.stderr.decode("utf-8", "replace")
    if _UNKNOWN_FLAG in stderr:
        raise RouterValidationUnavailableError(
            f"the Router image {image} predates `-validate-config`"
        )
    report = _report(result.stdout.decode("utf-8", "replace"))
    if report is None or result.returncode not in (0, 1):
        detail = stderr.strip().splitlines()[-1:] or [f"exit {result.returncode}"]
        raise RouterValidationUnavailableError(
            f"the Router image {image} gave no verdict: {detail[0]}"
        )
    return RouterVerdict(
        source=f"the Router in {image}",
        valid=bool(report.get("valid")),
        error=str(report.get("error") or ""),
        warnings=_warnings(report.get("warnings")),
    )


def validate_with_endpoint(
    config_path: Path, client: RouterManagementClient
) -> RouterVerdict:
    """Ask a running Router to validate the file without applying it."""

    source = f"the Router at {client.base_url}"
    try:
        response = client.validate_config(config_path.read_text(encoding="utf-8"))
    except RouterManagementError as error:
        if error.status in (400, 422):
            return RouterVerdict(
                source=source, valid=False, error=error.detail or str(error)
            )
        raise RouterValidationUnavailableError(str(error)) from error
    payload = response.payload if isinstance(response.payload, dict) else {}
    return RouterVerdict(
        source=source,
        valid=bool(payload.get("valid")),
        warnings=_warnings(payload.get("warnings")),
    )


def _report(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        try:
            report = json.loads(line)
        except ValueError:
            continue
        if isinstance(report, dict) and "valid" in report:
            return report
    return None


def _warnings(raw: object) -> tuple[RouterWarning, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        RouterWarning(
            code=str(item.get("code") or ""),
            field=str(item.get("field") or ""),
            message=str(item.get("message") or ""),
        )
        for item in raw
        if isinstance(item, dict) and item.get("message")
    )


def _container_image(container_name: str) -> str:
    try:
        result = subprocess.run(
            [
                get_container_runtime(),
                "inspect",
                "--format",
                "{{.Config.Image}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, SystemExit):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
