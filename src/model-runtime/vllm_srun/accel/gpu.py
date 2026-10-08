"""Shared logic for CUDA and ROCm accelerators (both use the torch.cuda device API).

The released runtime runs the backbone under BF16 autocast and takes the
gated-delta and causal-convolution kernels from FLA and causal-conv1d when
they are installed, else the torch references. The same order applies here.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ..plugins.base import Accelerator, DeviceInfo
from .kernels import Kernel, KernelSet, reference_kernels

_DEVICE_LOCKS: dict[int, Any] = {}
_LOCKS = threading.Lock()

# Errors after which the process's device context stays unusable.
DEVICE_FAULT_MARKERS = (
    "CUDA error",
    "HIP error",
    "device-side assert",
    "illegal memory access",
)


def device_lock(index: int) -> Any:
    """The lock that serializes this process's device work on one GPU."""
    with _LOCKS:
        lock = _DEVICE_LOCKS.get(index)
        if lock is None:
            lock = _DEVICE_LOCKS[index] = threading.RLock()
        return lock


def _current_device(target: torch.device) -> AbstractContextManager[Any]:
    if target.type != "cuda" or not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.device(target)


def _optional(module: str, attribute: str) -> Any:
    try:
        return getattr(importlib.import_module(module), attribute)
    except Exception:
        return None


class GPUAccelerator(Accelerator):
    """Base for accelerators that drive devices through ``torch.cuda``."""

    platform_attribute: str = ""

    def available(self) -> bool:
        return (
            bool(getattr(torch.version, self.platform_attribute, None))
            and torch.cuda.is_available()
        )

    def devices(self) -> list[DeviceInfo]:
        if not self.available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
            except RuntimeError:
                free, total = None, properties.total_memory
            devices.append(
                DeviceInfo(
                    accelerator=self.name,
                    index=index,
                    name=properties.name,
                    total_memory=total,
                    free_memory=free,
                    bf16=torch.cuda.is_bf16_supported(),
                    arch=self.arch(properties),
                )
            )
        return devices

    def arch(self, properties: Any) -> str | None:
        return None

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cuda", device.index or 0)

    def kernels(self, device: DeviceInfo) -> KernelSet:
        kernels = reference_kernels(device.label)
        conv = _optional("causal_conv1d", "causal_conv1d_fn")
        if conv is not None:
            kernels.register(
                Kernel("causal_conv1d", _wrap_conv(conv), "causal-conv1d", exact=True)
            )
        delta = _optional("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
        if delta is not None:
            kernels.register(
                Kernel("chunk_gated_delta_rule", _wrap_delta(delta), "fla", exact=True)
            )
        return kernels

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {
            "bf16_autocast": device.bf16,
            "graphs": True,
            "triton": _optional("triton", "__version__") is not None,
            "fla": _optional("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
            is not None,
            "causal_conv1d": _optional("causal_conv1d", "causal_conv1d_fn") is not None,
        }

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))

    def synchronize(self, device: DeviceInfo) -> None:
        torch.cuda.synchronize(device.index or 0)

    def execute(self, device: DeviceInfo, work: Callable[[], Any]) -> Any:
        """Run device work on its device, holding the device's lock, so the process's models never launch on it at once.

        A model captures a HIP / CUDA graph the second time it sees a shape,
        and the capture fails when another thread launches work on the device,
        whatever the capture mode; every capture also synchronizes the device.
        Triton kernels (FLA's among them) launch on the thread's current
        device, so the work makes the model's GPU current: otherwise a model on
        any GPU but the first faults reading its own memory from GPU 0.
        """
        with device_lock(device.index or 0), _current_device(self.torch_device(device)):
            return work()

    def device_fault(self, error: BaseException) -> bool:
        """Device errors poison the context; running out of memory fails only the batch."""
        if isinstance(error, torch.OutOfMemoryError):
            return False
        if isinstance(error, getattr(torch, "AcceleratorError", ())):
            return True
        return isinstance(error, RuntimeError) and any(
            marker in str(error) for marker in DEVICE_FAULT_MARKERS
        )


def _wrap_conv(fn: Any) -> Any:
    def conv(hidden_states, weight, bias=None, activation=None):
        return fn(hidden_states, weight, bias=bias, activation=activation)

    return conv


def _wrap_delta(fn: Any) -> Any:
    """FLA's chunked gated delta rule, called as Transformers' fallback wrapper calls it.

    ``cu_seqlens`` (variable-length sequences packed in one row, shared-context
    tree mode) passes through, with ``cu_seqlens_cpu`` where FLA takes it.
    """
    accepted = set(inspect.signature(fn).parameters)

    def delta(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
    ):
        extra = {}
        if cu_seqlens is not None:
            extra["cu_seqlens"] = cu_seqlens
            if cu_seqlens_cpu is not None and "cu_seqlens_cpu" in accepted:
                extra["cu_seqlens_cpu"] = cu_seqlens_cpu
        return fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **extra,
        )

    return delta
