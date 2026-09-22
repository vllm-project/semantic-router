# Phase-local imports preserve memory limits and authenticated source loading.
# ruff: noqa: PLC0415
"""Only the reviewed immutable published source is executable during export."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from contract import digest, sources


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
        from huggingface_hub import (
            snapshot_download,
        )

        source = Path(
            snapshot_download(
                pinned["repo_id"],
                revision=pinned["revision"],
                allow_patterns=list(pinned["files"]),
            )
        )
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
