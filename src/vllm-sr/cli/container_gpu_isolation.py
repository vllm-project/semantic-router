"""Router-container GPU isolation for local accelerator stacks."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cli.commands.runtime_paths import (
    _create_or_harden_private_directory,
    private_runtime_state_nested_directory,
)
from cli.utils import get_logger

log = get_logger(__name__)

AMD_ROUTER_VISIBLE_DEVICES_ENV = "VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES"
COMGR_CACHE_CONTAINER_PATH = "/root/.cache/comgr"
_IMAGE_ID = re.compile(r"sha256:([0-9a-f]{64})\Z")


@dataclass(frozen=True)
class AMDCompilerCache:
    image_id: str
    mount: str


def router_compiler_cache(
    runtime: str,
    image: str,
    vllm_sr_dir: str,
    stack_name: str,
    platform: str,
) -> AMDCompilerCache | None:
    """Pin an AMD Router image and persist its private COMGR compiler cache.

    The image ID contains the ROCm userspace version. A changed tag therefore
    selects a fresh cache, while a second serve of the same image reuses it.
    If the image cannot be inspected, startup still works without this cache.
    """

    if platform != "amd":
        return None
    try:
        result = subprocess.run(
            [runtime, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log.warning("AMD compiler cache disabled: image inspect failed: %s", error)
        return None
    match = (
        _IMAGE_ID.fullmatch(result.stdout.strip()) if result.returncode == 0 else None
    )
    if match is None:
        log.warning("AMD compiler cache disabled: immutable image ID is unavailable")
        return None

    stack_key = hashlib.sha256(stack_name.encode("utf-8")).hexdigest()
    state_root = Path(vllm_sr_dir).parent
    stack_cache = private_runtime_state_nested_directory(
        state_root, "compiler-cache", stack_key
    )
    image_cache = _create_or_harden_private_directory(stack_cache / match.group(1))
    log.info("Persistent AMD compiler cache: %s", image_cache)
    return AMDCompilerCache(
        image_id=result.stdout.strip(),
        mount=f"{image_cache}:{COMGR_CACHE_CONTAINER_PATH}:z",
    )


def router_runtime_env(
    common_env: dict[str, str],
    platform: str,
) -> dict[str, str]:
    """Return router-only env with optional AMD device isolation."""

    router_env = dict(common_env)
    if platform != "amd":
        return router_env

    visible_devices = os.getenv(AMD_ROUTER_VISIBLE_DEVICES_ENV, "").strip()
    if not visible_devices:
        return router_env

    router_env["ROCR_VISIBLE_DEVICES"] = visible_devices
    log.info(
        "AMD router GPU isolation: ROCR_VISIBLE_DEVICES=%s",
        visible_devices,
    )
    return router_env
