"""Tiny Vela Omni packages for the fixture command: published snapshots, and prepared bundles."""

from __future__ import annotations

from pathlib import Path

from ..families.multimodal_embedding import bundle as bundles
from .omni import write_bundle, write_snapshot

BUNDLE = "-bundle"
VARIANTS = (*bundles.VARIANTS, *(f"{variant}{BUNDLE}" for variant in bundles.VARIANTS))


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: a snapshot of ``variant`` (``nano``, ``mini``), or its bundle (``nano-bundle``).

    Neither is pinned, so readiness checks the random weights for determinism
    instead of against the published model's golden answers.
    """
    variant = variant or VARIANTS[0]
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown Omni fixture variant {variant!r}; choose from {list(VARIANTS)}"
        )
    if variant.endswith(BUNDLE):
        size = variant.removesuffix(BUNDLE)
        source = {"repo_id": f"vllm-sr-fixtures/omni-{size}", "revision": "0" * 40}
        return write_bundle(Path(output), variant=size, source=source, seed=seed)
    return write_snapshot(Path(output), variant=variant, seed=seed)
