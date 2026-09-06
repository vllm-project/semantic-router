"""Router-container GPU isolation for local accelerator stacks."""

from __future__ import annotations

import os

from cli.utils import get_logger

log = get_logger(__name__)

AMD_ROUTER_VISIBLE_DEVICES_ENV = "VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES"


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
