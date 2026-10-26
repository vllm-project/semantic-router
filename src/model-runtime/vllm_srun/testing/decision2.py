"""Tiny random-weight Decision 2.0 packages for the fixture command; the writer is ``fixtures.write_package``."""

from __future__ import annotations

from pathlib import Path

from .fixtures import write_package

VARIANTS = ("qwen3_5", "qwen3")


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: a package on the ``variant`` backbone."""
    return write_package(output, backbone=variant or VARIANTS[0], seed=seed)
