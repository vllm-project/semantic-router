"""Ascend NPU accelerator (``torch.npu``): pure-torch reference kernels. Unvalidated."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ..plugins.base import Accelerator, DeviceInfo
from .kernels import KernelSet, reference_kernels

# Errors after which the process's NPU context stays unusable (the gpu.py precedent).
NPU_FAULT_MARKERS = (
    "NPU error",
    "Device error",
)

_FALLBACK_NAME = "AscendNPU"


def _npu() -> Any:
    """``torch.npu`` once the ``torch_npu`` extension registered it, else ``None``."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        try:
            import torch_npu  # noqa: F401  (registers torch.npu)
        except Exception:
            return None
        npu = getattr(torch, "npu", None)
    return npu


class NPUAccelerator(Accelerator):
    name = "npu"
    validated = False

    def available(self) -> bool:
        npu = _npu()
        return bool(npu is not None and npu.is_available())

    def devices(self) -> list[DeviceInfo]:
        if not self.available():
            return []
        npu = _npu()
        devices = []
        for index in range(npu.device_count()):
            properties = npu.get_device_properties(index)
            try:
                free_memory = npu.mem_get_info(index)[0]
            except Exception:
                free_memory = None
            devices.append(
                DeviceInfo(
                    accelerator="npu",
                    index=index,
                    name=getattr(properties, "name", _FALLBACK_NAME),
                    total_memory=getattr(properties, "total_memory", None),
                    free_memory=free_memory,
                    bf16=bool(npu.is_bf16_supported()),
                    arch=None,
                )
            )
        return devices

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("npu", device.index or 0)

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels(device.label)

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {"bf16_autocast": True, "graphs": False, "triton": False}

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="npu", dtype=getattr(torch, dtype))

    def synchronize(self, device: DeviceInfo) -> None:
        _npu().synchronize(device.index or 0)

    def device_fault(self, error: BaseException) -> bool:
        """Device errors poison the context; running out of memory fails only the batch."""
        if isinstance(error, torch.OutOfMemoryError):
            return False
        if isinstance(error, getattr(torch, "AcceleratorError", ())):
            return True
        return isinstance(error, RuntimeError) and any(
            marker in str(error) for marker in NPU_FAULT_MARKERS
        )
