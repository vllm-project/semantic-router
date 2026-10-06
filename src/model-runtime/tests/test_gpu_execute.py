"""GPU device work runs one call at a time per device: a graph capture needs the device to itself."""

from __future__ import annotations

import threading
import time

import pytest
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
