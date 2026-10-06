"""A host accelerator: the smallest complete accelerator plugin.

It offers one device, the host's CPU under its own name, with the runtime's
reference kernels. It sets no ``auto_priority``, so ``--device auto`` never
picks it; a model runs on it with ``--device example_host``.
"""

from __future__ import annotations

import platform

import torch
from vllm_srun.accel.kernels import KernelSet, reference_kernels
from vllm_srun.plugins.base import Accelerator, DeviceInfo


class HostAccelerator(Accelerator):
    name = "example_host"
    validated = False

    def available(self) -> bool:
        return True

    def devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(accelerator=self.name, index=None, name=platform.machine())]

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cpu")

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels("cpu")
