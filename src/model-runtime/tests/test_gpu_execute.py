"""GPU device work runs on its GPU, one call at a time per device: a graph capture needs the device to itself."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
import torch
from vllm_srun.accel.cuda import CUDAAccelerator
from vllm_srun.accel.rocm import ROCmAccelerator
from vllm_srun.plugins.base import DeviceInfo


def test_device_work_on_one_gpu_never_overlaps_while_other_gpus_run_in_parallel():
    accelerator = ROCmAccelerator()
    devices = [DeviceInfo("rocm", index, f"gpu{index}") for index in (0, 1)]
    guard = threading.Lock()
    running = {0: 0, 1: 0}
    peak = {0: 0, 1: 0, "both": 0}

    def work(index: int) -> int:
        with guard:
            running[index] += 1
            peak[index] = max(peak[index], running[index])
            peak["both"] = max(peak["both"], running[0] + running[1])
        time.sleep(0.005)
        with guard:
            running[index] -= 1
        return index

    def caller(device: DeviceInfo) -> None:
        for _ in range(8):
            assert (
                accelerator.execute(device, lambda: work(device.index)) == device.index
            )

    threads = [
        threading.Thread(target=caller, args=(devices[n % 2],)) for n in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == peak[1] == 1
    assert peak["both"] == 2


@pytest.mark.parametrize("accelerator", [ROCmAccelerator(), CUDAAccelerator()])
def test_device_work_may_nest_on_its_own_thread(accelerator):
    device = DeviceInfo(accelerator.name, 0, "gpu0")
    assert (
        accelerator.execute(device, lambda: accelerator.execute(device, lambda: 7)) == 7
    )


@pytest.mark.parametrize("accelerator", [ROCmAccelerator(), CUDAAccelerator()])
def test_device_work_runs_with_its_gpu_as_the_current_device(accelerator, monkeypatch):
    current = [0]
    entered = []

    @contextmanager
    def device(target):
        entered.append(target)
        previous, current[0] = current[0], target.index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    gpu1 = DeviceInfo(accelerator.name, 1, "gpu1")
    assert accelerator.execute(gpu1, lambda: current[0]) == 1
    assert current[0] == 0
    assert entered == [torch.device("cuda", 1)]


def test_device_work_of_a_host_backed_accelerator_leaves_the_current_gpu_alone(
    monkeypatch,
):
    class HostBacked(ROCmAccelerator):
        def torch_device(self, device: DeviceInfo) -> torch.device:
            return torch.device("cpu")

    def device(target):
        raise AssertionError(f"set the current GPU for a host device: {target}")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    assert HostBacked().execute(DeviceInfo("rocm", 1, "host"), lambda: 7) == 7


@pytest.mark.gpu
def test_device_work_on_a_second_gpu_launches_there():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    accelerator = ROCmAccelerator() if torch.version.hip else CUDAAccelerator()
    gpu1 = DeviceInfo(accelerator.name, 1, "gpu1")

    def work() -> int:
        values = torch.arange(4, device="cuda:1")
        assert int((values * 2).sum()) == 12
        return torch.cuda.current_device()

    assert accelerator.execute(gpu1, work) == 1
    assert torch.cuda.current_device() == 0
