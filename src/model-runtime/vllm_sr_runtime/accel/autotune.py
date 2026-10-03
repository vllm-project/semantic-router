"""Persisted Triton autotuning, so GPU answers repeat across processes.

FLA's gated-delta kernels choose block sizes and warps by timing them in each
process. Two processes can therefore pick different configurations and answer
the Qwen3.5 sizes differently by rounding. Sharing one autotune cache makes
every process reuse the first process's choices.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

AUTOTUNE_ENV = "VLLM_SR_RUNTIME_AUTOTUNE_CACHE"


def freeze_autotune(directory: str) -> Path:
    """Record autotune choices in ``directory`` on first use and reuse them afterwards.

    Triton reads these settings when a kernel is decorated, so this must run
    before FLA is imported.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    if "fla" in sys.modules:
        log.warning(
            "FLA was imported before the autotune cache was set; its kernels keep per-process tuning"
        )
    return path
