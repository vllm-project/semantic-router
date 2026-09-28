# Phase-local imports preserve memory limits and authenticated source loading.
# ruff: noqa: PLC0415
"""Only the reviewed immutable published source is executable during export."""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
from pathlib import Path

from contract import digest, sources

HUB_METADATA_TIMEOUT_SECONDS = 60
HUB_SNAPSHOT_ATTEMPTS = 4
HUB_RATE_LIMIT_STATUS = 429
HUB_SERVER_ERROR_MIN = 500
HUB_SERVER_ERROR_END = 600

# export.py imports this module before its ML dependencies, some of which load
# Hub constants. Set the finite transfer timeout before that first Hub import.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HUB_METADATA_TIMEOUT_SECONDS))


def _temporary_hub_failure(error: BaseException) -> bool:
    """Retry only network failures and rate limits/server errors hidden by Hub."""
    from requests.exceptions import (
        ConnectionError,
        HTTPError,
        ProxyError,
        SSLError,
        Timeout,
    )

    retryable = False
    while error is not None:
        if isinstance(error, (ProxyError, SSLError)):
            return False
        if isinstance(error, HTTPError):
            status = getattr(error.response, "status_code", None)
            if status == HUB_RATE_LIMIT_STATUS or (
                status is not None
                and HUB_SERVER_ERROR_MIN <= status < HUB_SERVER_ERROR_END
            ):
                retryable = True
            elif status is not None:
                return False
        elif isinstance(error, (ConnectionError, Timeout)):
            retryable = True
        error = error.__cause__ or error.__context__
    return retryable


def _download_pinned_snapshot(pinned: dict) -> Path:
    from huggingface_hub import snapshot_download

    for attempt in range(HUB_SNAPSHOT_ATTEMPTS):
        try:
            return Path(
                snapshot_download(
                    pinned["repo_id"],
                    revision=pinned["revision"],
                    allow_patterns=list(pinned["files"]),
                    etag_timeout=HUB_METADATA_TIMEOUT_SECONDS,
                )
            )
        except Exception as error:
            if attempt == HUB_SNAPSHOT_ATTEMPTS - 1 or not _temporary_hub_failure(
                error
            ):
                raise
            delay = 2**attempt + random.uniform(0, 1)
            print(
                f"Temporary Hub failure for pinned {pinned['repo_id']} "
                f"(attempt {attempt + 1}/{HUB_SNAPSHOT_ATTEMPTS}); "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable: snapshot attempts exhausted")


def provision(
    variant: str, source: Path | None, download: bool, *, weights: bool = True
) -> Path:
    pinned = sources()[variant]
    if not weights:
        pinned = {
            **pinned,
            "files": {
                name: value
                for name, value in pinned["files"].items()
                if name != "model.safetensors"
            },
        }
    if source is None:
        if not download:
            raise ValueError("provide --source or explicitly request --download")
        source = _download_pinned_snapshot(pinned)
    verify_source(source, pinned)
    return source.resolve()


def verify_source(directory: Path, pinned: dict) -> None:
    for name, expected in pinned["files"].items():
        # Hub snapshots use symlinks into the content-addressed blob cache.
        # Authenticate the bytes, rather than rejecting those legitimate links.
        path = directory / name
        if (
            path.stat().st_size != expected["size"]
            or digest(path, expected["algorithm"]) != expected["digest"]
        ):
            raise ValueError(f"published source identity mismatch: {name}")
    # Import resolution must not select an unreviewed module from this directory.
    allowed = {name for name in pinned["files"] if name.endswith(".py")}
    found = {p.relative_to(directory).as_posix() for p in directory.rglob("*.py")}
    if found != allowed:
        raise ValueError(
            f"unexpected/missing Python source files: {sorted(found ^ allowed)}"
        )


def load_reference(directory: Path):
    """Import only after provision() has authenticated every source input."""
    import torch

    if "omni_components" in sys.modules:
        raise RuntimeError(
            "export each variant in a fresh process; omni_components is already imported"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(
        "_pinned_vela_omni", directory / "vela_omni.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VelaOmni.from_pretrained(
        str(directory), device="cpu", dtype=torch.float32
    )
