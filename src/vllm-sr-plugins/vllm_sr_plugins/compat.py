"""The one vLLM build these plugins are tested against, and plugin logging."""

from __future__ import annotations

import logging

PINNED_VLLM = "0.29.1rc1.dev187+gaf1c01499.rocm723"
PINNED_IMAGE = "decision20-train-fast:host2 (image id sha256:f83b1d10f14d...)"


def get_logger(name: str) -> logging.Logger:
    """A logger under ``vllm.`` so records use vLLM's configured handler."""
    return logging.getLogger(f"vllm.sr_plugins.{name}")


def check_vllm_version() -> str:
    """Return the installed vLLM build, warning when it is not the pinned one."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        running = version("vllm")  # vllm.__version__ omits the local build suffix
    except PackageNotFoundError:
        running = "unknown"
    if running != PINNED_VLLM:
        get_logger("compat").warning(
            "vllm-sr-plugins is tested against vLLM %s; running %s. Pooler, "
            "loader and endpoint-plugin interfaces change between vLLM versions.",
            PINNED_VLLM,
            running,
        )
    return running
