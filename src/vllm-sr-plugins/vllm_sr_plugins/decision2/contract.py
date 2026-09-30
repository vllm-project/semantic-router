"""Which Decision 2.0 checkpoints the scoring model class implements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARCHITECTURE = "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"


def read_decision_config(root: Path) -> dict[str, Any]:
    """The package's decision_config.json, if it is a full Qwen3.5 shared-head checkpoint."""
    metadata = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    if (
        metadata.get("architecture") != ARCHITECTURE
        or metadata.get("head_variant", "shared") != "shared"
        or metadata.get("dec_residual") is not None
        or metadata.get("checkpoint_format", "full") != "full"
    ):
        raise ValueError(
            "Decision2Qwen3_5ForScoring serves full Qwen3.5 checkpoints with the "
            f"shared candidate head only; {root} declares "
            f"{metadata.get('architecture')!r} / {metadata.get('head_variant', 'shared')!r}"
        )
    return metadata
