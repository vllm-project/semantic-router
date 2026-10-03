"""Persisted and pinned GPU autotuning, so GPU answers repeat across processes.

FLA's gated-delta kernels choose block sizes and warps by timing them in each
process. Two processes can therefore pick different configurations and answer
the Qwen3.5 sizes differently by rounding. Sharing one autotune cache makes
every process reuse the first process's choices; pinning a model's recorded
choices makes every process run the same kernels without timing any.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

AUTOTUNE_ENV = "VLLM_SR_RUNTIME_AUTOTUNE_CACHE"
FLA_CONFIG_ENV = "FLA_CONFIG_DIR"
FLA_MODE_ENV = "FLA_CACHE_MODE"
VERSION = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.M)


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


def pin_kernel_choices(choices: dict[str, Any]) -> Path | None:
    """Run FLA's autotuned kernels with a model's recorded configurations instead of timing them.

    ``choices`` is one device class's entry of ``registry/kernel_choices.json``:
    the FLA version the choices were recorded with and, per kernel, each
    recorded tuning key with its configuration. Every kernel gets an FLA config
    file (``FLA_CACHE_MODE=full``): recorded keys match exactly, and any other
    key takes the entry of the same structure or the kernel's first entry, so
    no configuration depends on timing. FLA reads the mode when it is imported,
    so this must run before FLA is imported. Returns the config directory, or
    None when the choices do not apply to this process.
    """
    if "fla" in sys.modules:
        log.warning(
            "FLA was imported before its kernel choices were pinned; its kernels keep per-process tuning"
        )
        return None
    installed = fla_version()
    if installed != choices["fla"]:
        log.warning(
            "kernel choices were recorded with FLA %s, not %s; FLA keeps per-process tuning",
            choices["fla"],
            installed,
        )
        return None
    directory = Path(tempfile.mkdtemp(prefix="vllm-sr-runtime-fla-"))
    atexit.register(shutil.rmtree, directory, True)
    for name, entries in choices["kernels"].items():
        document = {
            "kernel_name": name,
            "autotune_entries": {
                fla_key_hash(entry["key"]): {
                    "autotune_key": entry["key"],
                    "config": entry["config"],
                }
                for entry in entries
            },
            "default_config": entries[0]["config"],
        }
        (directory / f"{name}.json").write_text(
            json.dumps(document, sort_keys=True), encoding="utf-8"
        )
    os.environ[FLA_CONFIG_ENV] = str(directory)
    os.environ[FLA_MODE_ENV] = "full"
    log.info("pinned FLA kernel choices for %d kernels", len(choices["kernels"]))
    return directory


def fla_version() -> str | None:
    """The version of the FLA that ``import fla`` would load, read without importing it."""
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.origin:
        return None
    match = VERSION.search(Path(spec.origin).read_text(encoding="utf-8"))
    return match.group(1) if match else None


def fla_key_hash(key: list[Any]) -> str:
    """FLA's ``AutotuneKey.key_hash`` of a tuning key."""
    serialized = json.dumps(key, separators=(",", ":"), sort_keys=True)
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()
