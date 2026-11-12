#!/usr/bin/env python3
"""Download the pinned Vela Omni releases the model runtime serves, and verify every file it reads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/model-runtime"
DOWNLOAD_ATTEMPTS = 4

# The runtime's pins and file hashing, from this checkout (standard library only).
sys.path.insert(0, str(RUNTIME))
from vllm_srun.errors import PackageError  # noqa: E402
from vllm_srun.registry.artifacts import named_files  # noqa: E402
from vllm_srun.registry.tables import omni  # noqa: E402


def pins():
    """The runtime's Omni table."""
    return omni


def verify(directory: Path, pinned) -> bool:
    """Whether every file the family reads has its pinned SHA-256."""
    try:
        return named_files(directory, pinned.files) == dict(pinned.files)
    except PackageError:
        return False


def download(directory: Path, pinned) -> None:
    from huggingface_hub import (  # noqa: PLC0415 - only a download needs the Hub client
        snapshot_download,
    )

    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            snapshot_download(
                pinned.repo_id,
                revision=pinned.revision,
                allow_patterns=sorted(pinned.files),
                local_dir=directory,
            )
            return
        except Exception as error:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            print(f"Retrying {pinned.repo_id}: {type(error).__name__}", flush=True)


def prepare(output: Path, variants: list[str]) -> None:
    table = pins()
    output.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        pinned = table.lookup(f"vllm-sr/Vela-1.0-Omni-{variant.capitalize()}")
        directory = output / pinned.repo_id.split("/", 1)[1].lower()
        if verify(directory, pinned):
            print(f"Reusing verified {directory.name}", flush=True)
            continue
        download(directory, pinned)
        if not verify(directory, pinned):
            raise ValueError(
                f"{directory.name} differs from the runtime's pinned files"
            )
        print(f"Downloaded and verified {directory.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("nano", "mini"), required=True
    )
    args = parser.parse_args()
    prepare(args.output.resolve(), list(dict.fromkeys(args.variants)))


if __name__ == "__main__":
    main()
