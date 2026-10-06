"""Tiny Vela Omni bundles for the fixture command; the writer is ``omni.write_bundle``."""

from __future__ import annotations

from pathlib import Path

from ..families.multimodal_embedding import bundle as bundles
from .omni import write_bundle

VARIANTS = tuple(bundles.VARIANTS)


def write_fixture(output: str | Path, variant: str | None, seed: int) -> Path:
    """The fixture command's writer: a bundle of ``variant``'s contract.

    The source is unpinned, so readiness checks the random graphs for
    determinism instead of against the pinned model's golden answers.
    """
    variant = variant or VARIANTS[0]
    source = {"repo_id": f"vllm-sr-fixtures/omni-{variant}", "revision": "0" * 40}
    return write_bundle(Path(output), variant=variant, source=source, seed=seed)
