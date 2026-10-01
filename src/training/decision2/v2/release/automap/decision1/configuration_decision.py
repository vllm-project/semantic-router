# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""Configuration of a Decision 1.0 repository.

The repository's root ``config.json`` is its Decision file map
(``decision_format: vllm-sr-decision``): model name, runtime family, and the
paths of the backbone, tokenizer, decision weights and calibration. The keys
``model_type``, ``architectures``, ``auto_map`` and ``custom_pipelines`` let
Transformers load it with ``trust_remote_code=True``.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

try:
    from transformers import PreTrainedConfig as _BaseConfig
except ImportError:  # Transformers 4.x
    from transformers import PretrainedConfig as _BaseConfig

FAMILIES = ("vela-encoder", "qwen3.5-decision")
DESCRIPTOR_KEYS = (
    "decision_format",
    "format_version",
    "model_name",
    "runtime_family",
    "model_config",
    "backbone",
    "tokenizer",
    "decision_weights",
    "calibration",
)


def _relative(path: Any, where: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError(f"{where} must be a relative file path")
    parts = PurePosixPath(path)
    if parts.is_absolute() or ".." in parts.parts or "." in parts.parts:
        raise ValueError(f"{where} must stay inside the repository")
    return path


class DecisionConfig(_BaseConfig):
    model_type = "decision"

    def __init__(
        self,
        decision_format: str = "vllm-sr-decision",
        format_version: int = 1,
        model_name: str | None = None,
        runtime_family: str | None = None,
        model_config: str | None = None,
        backbone: dict[str, Any] | None = None,
        tokenizer: dict[str, Any] | None = None,
        decision_weights: dict[str, str] | None = None,
        calibration: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.decision_format = decision_format
        self.format_version = format_version
        self.model_name = model_name
        self.runtime_family = runtime_family
        self.model_config = model_config
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.decision_weights = decision_weights
        if calibration is not None:
            self.calibration = calibration
        super().__init__(**kwargs)

    def descriptor(self) -> dict[str, Any]:
        """The validated Decision file map."""
        if self.decision_format != "vllm-sr-decision" or self.format_version != 1:
            raise ValueError("config.json is not a Decision 1.0 file map")
        if self.runtime_family not in FAMILIES:
            raise ValueError(
                f"Unsupported Decision runtime family: {self.runtime_family!r}"
            )
        descriptor = {
            key: getattr(self, key)
            for key in DESCRIPTOR_KEYS
            if getattr(self, key, None) is not None
        }
        for key in ("model_config", "backbone", "tokenizer", "decision_weights"):
            if key not in descriptor:
                raise ValueError(f"config.json does not name {key}")
        return descriptor

    def files(self) -> list[str]:
        """Every repository file that inference reads."""
        descriptor = self.descriptor()
        backbone, tokenizer = descriptor["backbone"], descriptor["tokenizer"]
        names = [descriptor["model_config"], backbone["config"], *backbone["weights"]]
        if backbone.get("index"):
            names.append(backbone["index"])
        names += [value for value in tokenizer.values() if isinstance(value, str)]
        names += list(descriptor["decision_weights"].values())
        calibration = descriptor.get("calibration") or {}
        if calibration.get("temperature_file"):
            names.append(calibration["temperature_file"])
        return sorted({_relative(name, "config.json path") for name in names})
