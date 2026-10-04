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

A process that runs CPU device work keeps freed memory in its heap (glibc).
By default glibc hands a freed activation's pages back to the kernel, and the
next forward faults them in again, zero-filled, until its dynamic thresholds
have grown past the forward's blocks. In a fresh process a one-row 307M encoder
forward on 16 EPYC cores took 62 ms with 51,000 page faults at 128 tokens and
470 ms at 1,024, against 20 ms and 109 ms with the memory kept; in a serving
process, which has already run larger batches, inputs longer than those keep
paying it (Vela Halu's p95 2.76 s against 2.11 s).
"""

from __future__ import annotations

import ctypes
import ctypes.util
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
# glibc ``mallopt`` parameters. Blocks up to the mmap threshold (glibc's largest
# on 64-bit hosts) come from the heap, which returns memory to the kernel only
# past the trim threshold of free space at its top.
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
HEAP_MMAP_THRESHOLD = 32 << 20
HEAP_TRIM_THRESHOLD = 256 << 20


def keep_freed_memory() -> bool:
    """Keep freed CPU memory in the process heap instead of the kernel (glibc); whether it applied."""
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        return False
    mallopt = getattr(ctypes.CDLL(ctypes.util.find_library("c")), "mallopt", None)
    if mallopt is None:
        return False
    return bool(
        mallopt(M_MMAP_THRESHOLD, HEAP_MMAP_THRESHOLD)
        and mallopt(M_TRIM_THRESHOLD, HEAP_TRIM_THRESHOLD)
    )


def native_bf16() -> bool:
    """Whether the CPU computes BF16 natively (AVX-512 BF16 or AMX)."""
    probes = ("_is_avx512_bf16_supported", "_is_amx_tile_supported")
    return any(getattr(torch.cpu, probe, lambda: False)() for probe in probes)


def device_thread() -> ThreadPoolExecutor:
    """The process's single CPU device thread; creating it keeps freed memory in the heap."""
    global _EXECUTOR  # noqa: PLW0603 - one executor per process, created on first use
    with _LOCK:
        if _EXECUTOR is None:
            keep_freed_memory()
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vllm-sr-cpu"
            )
        return _EXECUTOR


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
        }

    def execute(self, device: DeviceInfo, work: Callable[[], Any]) -> Any:
        if threading.current_thread().name.startswith("vllm-sr-cpu"):
            return work()
        return device_thread().submit(work).result()
