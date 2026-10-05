"""CUDA accelerator. Implemented and unit-tested; not validated on NVIDIA hardware."""

from __future__ import annotations

from typing import Any

import torch

from .gpu import GPUAccelerator


class CUDAAccelerator(GPUAccelerator):
    name = "cuda"
    validated = False
    auto_priority = 1
    platform_attribute = "cuda"

    def available(self) -> bool:
        return not torch.version.hip and super().available()

    def arch(self, properties: Any) -> str | None:
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        return f"sm_{major}{minor}" if major is not None else None
