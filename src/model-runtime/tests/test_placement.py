import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest
import torch
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.accel.cuda import CUDAAccelerator
from vllm_sr_runtime.accel.rocm import ROCmAccelerator
from vllm_sr_runtime.errors import PlacementError
from vllm_sr_runtime.placement import parse_device, place
from vllm_sr_runtime.plugins.base import (
    BackboneSpec,
    DeviceInfo,
    DtypePolicy,
    ModelSpec,
)

SPEC = ModelSpec("tiny", BackboneSpec("qwen3", {}, ()), DtypePolicy(), 1024)
GPU = torch.cuda.is_available()


def test_parse_device():
    assert parse_device("auto") == ("auto", None)
    assert parse_device("rocm:3") == ("rocm", 3)
    assert parse_device("CPU") == ("cpu", None)
    for bad in ("gpu", "cuda:x", "rocm:-1", ""):
        with pytest.raises(PlacementError):
            parse_device(bad)


@pytest.mark.parametrize(
    ("avx512_bf16", "amx", "native"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_cpu_devices_report_native_bf16(monkeypatch, avx512_bf16, amx, native):
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: avx512_bf16)
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: amx)
    (device,) = CPUAccelerator().devices()
    assert device.bf16 is native
    assert CPUAccelerator().capabilities(device)["native_bf16"] is native


def test_cpu_placement_and_budget():
    placement = place(SPEC, "cpu", 1_000_000)
    assert placement.device.accelerator == "cpu" and placement.accelerator.validated
    with pytest.raises(PlacementError, match="memory-budget"):
        place(SPEC, "cpu", 10_000_000_000, memory_budget_gib=1)


@pytest.mark.skipif(GPU, reason="checks the CPU-only fallback")
def test_auto_falls_back_to_cpu_without_gpus():
    assert place(SPEC, "auto", 1000).device.accelerator == "cpu"
    with pytest.raises(PlacementError, match="not available"):
        place(SPEC, "rocm:0", 1000)


def test_cuda_is_marked_unvalidated_and_rocm_validated():
    assert CUDAAccelerator.validated is False
    assert ROCmAccelerator.validated is True


@pytest.mark.gpu
@pytest.mark.skipif(not GPU, reason="needs a CUDA or ROCm device")
def test_gpu_devices_report_memory_and_bf16():
    accelerator = ROCmAccelerator() if torch.version.hip else CUDAAccelerator()
    devices = accelerator.devices()
    assert devices and devices[0].total_memory and devices[0].index == 0


def test_gpu_device_work_runs_one_at_a_time_per_device():
    accelerator = ROCmAccelerator()
    first, second = (DeviceInfo("rocm", index, "fake") for index in (90, 91))
    active, peak, guard = Counter(), Counter(), threading.Lock()

    def work(index: int) -> int:
        with guard:
            active[index] += 1
            peak[index] = max(peak[index], active[index])
        time.sleep(0.005)
        with guard:
            active[index] -= 1
        return index

    with ThreadPoolExecutor(8) as pool:
        futures = [
            pool.submit(accelerator.execute, device, partial(work, device.index))
            for device in (first, second)
            for _ in range(6)
        ]
        assert sorted(future.result() for future in futures) == [90] * 6 + [91] * 6
    assert peak == Counter({90: 1, 91: 1})
    started = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        waits = pool.submit(accelerator.execute, first, lambda: started.wait(5))
        pool.submit(accelerator.execute, second, started.set).result()
        assert waits.result(), "work on one device waited for another device"
    nested = accelerator.execute(first, lambda: accelerator.execute(first, lambda: 7))
    assert nested == 7


def test_xpu_and_mps_devices_parse():
    assert parse_device("xpu:1") == ("xpu", 1)
    assert parse_device("mps") == ("mps", None)
