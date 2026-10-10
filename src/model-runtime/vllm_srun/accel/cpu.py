"""CPU accelerator: FP32 everywhere, pure-torch reference kernels. Validated.

For models that name them it registers the ``geglu`` slot's ``contiguous``
variant and, on x86, the ``linear`` slot's ``float32-packed`` one
(``onednn.py``): together they make an encoder's forward batch-invariant.

All CPU device work of a process runs on one thread (``execute``). PyTorch's
OpenMP backend keeps one thread team per calling thread; as soon as two teams
exist (a loader thread, one worker per model) the threads outnumber the cores,
libgomp stops spin-waiting between parallel regions and every one of a
forward's hundreds of regions pays a wake-up: a 307M encoder's short-input
forward took 25 ms instead of 13 ms on 16 cores. One thread, one team.
"""

from __future__ import annotations

import os
import platform
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from ..plugins.base import Accelerator, DeviceInfo
from . import onednn
from .kernels import (
    CONTIGUOUS,
    Kernel,
    KernelSet,
    geglu_contiguous,
    reference_kernels,
)

_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None


def native_bf16() -> bool:
    """Whether the CPU computes BF16 natively (AVX-512 BF16 or AMX)."""
    probes = ("_is_avx512_bf16_supported", "_is_amx_tile_supported")
    return any(getattr(torch.cpu, probe, lambda: False)() for probe in probes)


def device_thread() -> ThreadPoolExecutor:
    """The process's single CPU device thread."""
    global _EXECUTOR  # noqa: PLW0603 - one executor per process, created on first use
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vllm-sr-cpu"
            )
        return _EXECUTOR


class CPUAccelerator(Accelerator):
    name = "cpu"
    validated = True
    auto_priority = 100

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
                bf16=native_bf16(),
                arch=platform.machine(),
            )
        ]

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cpu")

    def kernels(self, device: DeviceInfo) -> KernelSet:
        kernels = reference_kernels("cpu")
        kernels.register(
            Kernel("geglu", geglu_contiguous, "torch", exact=True, variant=CONTIGUOUS)
        )
        if onednn.available():
            kernels.register(
                Kernel(
                    "linear",
                    onednn.PackedLinear,
                    "onednn",
                    exact=True,
                    variant=onednn.PACKED,
                )
            )
        return kernels

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {
            "bf16_autocast": False,
            "native_bf16": device.bf16,
            "graphs": False,
            "triton": False,
            # A private flag: a PyTorch that drops it is taken to have LAPACK,
            # and a model that needs it still fails its golden check if not.
            "lapack": bool(getattr(torch._C, "has_lapack", True)),
        }

    def lacks(self, device: DeviceInfo, capability: str) -> str:
        if capability == "lapack":
            return (
                f"{device.label}: this PyTorch is built without LAPACK, which the "
                "model's CPU kernels need; serve it with a PyTorch that has LAPACK "
                "(the router's CPU image) or on a GPU device"
            )
        return super().lacks(device, capability)

    def execute(self, device: DeviceInfo, work: Callable[[], Any]) -> Any:
        if threading.current_thread().name.startswith("vllm-sr-cpu"):
            return work()
        return device_thread().submit(work).result()
