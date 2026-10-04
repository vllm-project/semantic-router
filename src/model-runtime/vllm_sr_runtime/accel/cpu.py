"""CPU accelerator: FP32 everywhere, pure-torch reference kernels. Validated."""

from __future__ import annotations

import os
import platform

import torch

from ..plugins.base import Accelerator, DeviceInfo
from .kernels import KernelSet, reference_kernels


class CPUAccelerator(Accelerator):
    name = "cpu"
    validated = True

    def available(self) -> bool:
        return True

    def devices(self) -> list[DeviceInfo]:
        total = None
        free = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = pages * page_size
            free = os.sysconf("SC_AVPHYS_PAGES") * page_size
        except (ValueError, OSError, AttributeError):
            pass
        return [
            DeviceInfo(
                accelerator="cpu",
                index=None,
                name=platform.processor() or platform.machine(),
                total_memory=total,
                free_memory=free,
                bf16=False,
                arch=platform.machine(),
            )
        ]

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cpu")

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels("cpu")

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {"bf16_autocast": False, "graphs": False, "triton": False}
