"""The NPU accelerator with ``torch.npu`` faked: unit-tested without torch_npu or a device."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm_srun.accel.npu import NPUAccelerator


@pytest.fixture
def fake_npu(monkeypatch):
    npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(
            name=f"Ascend{index}", total_memory=64 << 30
        ),
        mem_get_info=lambda index: (60 << 30, 64 << 30),
        is_bf16_supported=lambda: True,
        synchronize=lambda index: None,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    return npu


def test_without_torch_npu_the_accelerator_is_unavailable():
    """On hosts without the torch_npu extension there is nothing to enumerate."""
    if (
        getattr(torch, "npu", None) is not None
        or importlib.util.find_spec("torch_npu") is not None
    ):
        pytest.skip("needs an environment where torch_npu is absent")
    accelerator = NPUAccelerator()
    assert accelerator.available() is False
    assert accelerator.devices() == []


def test_devices_report_memory_and_bf16(fake_npu):
    accelerator = NPUAccelerator()
    devices = accelerator.devices()
    assert [device.index for device in devices] == [0, 1]
    assert devices[0].name == "Ascend0"
    assert devices[0].total_memory == 64 << 30
    assert devices[0].free_memory == 60 << 30
    assert devices[0].bf16 is True


def test_torch_device_maps_name_and_index(fake_npu, monkeypatch):
    """torch.device("npu") only constructs where torch_npu registered the backend."""
    made = []
    monkeypatch.setattr(torch, "device", lambda *args: made.append(args) or args)
    accelerator = NPUAccelerator()
    device = accelerator.devices()[1]
    assert accelerator.torch_device(device) == ("npu", 1)
    assert made == [("npu", 1)]


def test_capabilities_and_reference_kernels(fake_npu):
    accelerator = NPUAccelerator()
    device = accelerator.devices()[1]
    assert accelerator.capabilities(device) == {
        "bf16_autocast": True,
        "graphs": False,
        "triton": False,
    }
    kernels = accelerator.kernels(device)
    assert kernels.device == "npu:1"
    assert "sdpa" in kernels.available
    assert "geglu" in kernels.available


def test_autocast_without_a_dtype_is_a_nullcontext(fake_npu):
    accelerator = NPUAccelerator()
    manager = accelerator.autocast(accelerator.devices()[0], None)
    assert isinstance(manager, nullcontext)
    with manager as ctx:
        assert ctx is None


def test_device_fault_markers(fake_npu):
    accelerator = NPUAccelerator()
    assert accelerator.device_fault(RuntimeError("NPU error, error code is 0x7020001"))
    assert not accelerator.device_fault(RuntimeError("some batch failed"))
    assert not accelerator.device_fault(torch.OutOfMemoryError("NPU out of memory"))
    if hasattr(torch, "AcceleratorError"):
        assert accelerator.device_fault(torch.AcceleratorError("device reset"))
