"""Apple MPS accelerator: FP32, pure-torch reference kernels. Unvalidated."""

from __future__ import annotations

import platform

import torch

from ..plugins.base import Accelerator, DeviceInfo
from .kernels import KernelSet, reference_kernels


class MPSAccelerator(Accelerator):
    name = "mps"
    validated = False

    def available(self) -> bool:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())

    def devices(self) -> list[DeviceInfo]:
        if not self.available():
            return []
        return [
            DeviceInfo(
                accelerator="mps",
                index=None,
                name=platform.processor() or "Apple GPU",
                total_memory=None,
                free_memory=None,
                bf16=False,
                arch=platform.machine(),
            )
        ]

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("mps")

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels("mps")

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {"bf16_autocast": False, "graphs": False, "triton": False}

    def synchronize(self, device: DeviceInfo) -> None:
        torch.mps.synchronize()
