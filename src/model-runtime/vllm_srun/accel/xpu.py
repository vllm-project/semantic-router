"""Intel XPU accelerator (``torch.xpu``): pure-torch reference kernels. Unvalidated."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ..plugins.base import Accelerator, DeviceInfo
from .kernels import KernelSet, reference_kernels


def _xpu() -> Any:
    return getattr(torch, "xpu", None)


class XPUAccelerator(Accelerator):
    name = "xpu"
    validated = False

    def available(self) -> bool:
        xpu = _xpu()
        return bool(xpu is not None and xpu.is_available())

    def devices(self) -> list[DeviceInfo]:
        if not self.available():
            return []
        xpu = _xpu()
        devices = []
        for index in range(xpu.device_count()):
            properties = xpu.get_device_properties(index)
            devices.append(
                DeviceInfo(
                    accelerator="xpu",
                    index=index,
                    name=getattr(properties, "name", "xpu"),
                    total_memory=getattr(properties, "total_memory", None),
                    free_memory=None,
                    bf16=True,
                    arch=None,
                )
            )
        return devices

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("xpu", device.index or 0)

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels(device.label)

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {"bf16_autocast": True, "graphs": False, "triton": False}

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="xpu", dtype=getattr(torch, dtype))

    def synchronize(self, device: DeviceInfo) -> None:
        _xpu().synchronize(device.index or 0)
