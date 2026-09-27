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
SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")


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


def build_adapter_download_config(
    root: Path, model_id: str, source_id: str, source_revision: str
) -> dict[str, Any]:
    """Describe an unmerged native adapter and its exact external backbone."""
    if (
        MODEL_ID.fullmatch(model_id) is None
        or SOURCE_ID.fullmatch(source_id) is None
        or REVISION.fullmatch(source_revision) is None
    ):
        raise ValueError("Adapter download config needs pinned model identities")
    required = (
        "model/decision_config.json",
        "model/decision_head.safetensors",
        "model/adapter/adapter_config.json",
        "model/adapter/adapter_model.safetensors",
        "model/tokenizer.json",
        "calibration.json",
    )
    if any(
        not (root / name).is_file() or (root / name).is_symlink() for name in required
    ):
        raise ValueError("Native adapter package is missing a config target")
    tokenizer: dict[str, str] = {"json": "model/tokenizer.json"}
    for key, name in (
        ("config", "model/tokenizer_config.json"),
        ("chat_template", "model/chat_template.jinja"),
    ):
        path = root / name
        if path.is_file() and not path.is_symlink():
            tokenizer[key] = name
    return {
        "decision_format": "vllm-sr-decision",
        "format_version": 2,
        "model_name": model_id.rsplit("/", 1)[1],
        "runtime_family": "decision2-native-adapter",
        "model_config": "model/decision_config.json",
        "backbone": {"repository": source_id, "revision": source_revision},
        "tokenizer": tokenizer,
        "decision_weights": {
            "decision_head": "model/decision_head.safetensors",
            "adapter_config": "model/adapter/adapter_config.json",
            "adapter_weights": "model/adapter/adapter_model.safetensors",
        },
        "calibration": {"temperature_file": "calibration.json"},
    }
