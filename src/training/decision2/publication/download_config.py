"""Root Hub query file for native Decision 2.0 model downloads.

The Hub counts a default model download when it serves ``config.json``.
This file is also a truthful inventory pointer, following Decision 1.0's
root layout: the custom Decision runtime still loads ``model/`` natively.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MODEL_ID = re.compile(r"llm-semantic-router/DEV2\.0-(?:0\.6|0\.8|2|4|9|27)B\Z")


def build_download_config(root: Path, model_id: str) -> dict[str, Any]:
    """Describe only files that are present in a staged native package."""
    if MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("Unexpected Decision 2.0 model ID")
    required = (
        "model/decision_config.json",
        "model/backbone/config.json",
        "model/decision_head.safetensors",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
        "calibration.json",
    )
    if any(
        not (root / name).is_file() or (root / name).is_symlink() for name in required
    ):
        raise ValueError("Native package is missing a config target")
    backbone = root / "model/backbone"
    weights = sorted(
        path.relative_to(root).as_posix()
        for path in backbone.glob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    )
    if not weights:
        raise ValueError("Native package has no backbone weight files")
    result: dict[str, Any] = {
        "decision_format": "vllm-sr-decision",
        "format_version": 2,
        "model_name": model_id.rsplit("/", 1)[1],
        "runtime_family": "decision2-native",
        "model_config": "model/decision_config.json",
        "backbone": {"config": "model/backbone/config.json", "weights": weights},
        "tokenizer": {
            "json": "model/tokenizer.json",
            "config": "model/tokenizer_config.json",
        },
        "decision_weights": {"decision_head": "model/decision_head.safetensors"},
        "calibration": {"temperature_file": "calibration.json"},
    }
    index = backbone / "model.safetensors.index.json"
    if index.is_file() and not index.is_symlink():
        result["backbone"]["index"] = index.relative_to(root).as_posix()
    template = root / "model/chat_template.jinja"
    if template.is_file() and not template.is_symlink():
        result["tokenizer"]["chat_template"] = template.relative_to(root).as_posix()
    return result
