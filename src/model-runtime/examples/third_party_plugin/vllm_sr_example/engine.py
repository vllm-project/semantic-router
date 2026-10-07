"""A counting engine: the smallest complete engine plugin.

It runs ``example_counts`` backbones only: a lookup table from keyword token
IDs to one-hot label vectors, given in the backbone config. The family reads
the hidden states it returns; the engine never sees the package.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm_srun.plugins.base import (
    Accelerator,
    DeviceInfo,
    EncoderBatch,
    EncoderOutput,
    Engine,
    EngineModel,
    EngineOptions,
    ForwardBatch,
    ForwardOutput,
    ModelSpec,
)


class CountsEngineModel(EngineModel):
    def __init__(self, table: torch.Tensor, device_info: DeviceInfo):
        self.table = table
        self.device = torch.device("cpu")
        self.device_info = device_info

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        raise NotImplementedError("example_counts serves encoders only")

    def encode(self, batch: EncoderBatch) -> EncoderOutput:
        hidden = self.table[batch.input_ids]
        return EncoderOutput(hidden=dict.fromkeys(batch.layers or (1,), hidden))

    def parameter_count(self) -> int:
        return 0


class CountsEngine(Engine):
    name = "example_counts"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            **super().descriptor(),
            "architectures": ["example_counts"],
            "outputs": ["hidden"],
            "devices": ["cpu"],
        }

    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        if spec.backbone.model_type != "example_counts":
            return f"no {spec.backbone.model_type!r} backbone"
        if device.accelerator not in ("cpu", "example_host"):
            return "example_counts runs on the host CPU only"
        return None

    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> EngineModel:
        labels = spec.backbone.config["labels"]
        label_of = spec.backbone.config["label_of"]
        table = torch.zeros((len(label_of), labels), dtype=torch.float32)
        for token, label in enumerate(label_of):
            if label >= 0:
                table[token, label] = 1.0
        return CountsEngineModel(table, device)
