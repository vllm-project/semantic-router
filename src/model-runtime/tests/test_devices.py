import json

import pytest
import torch
from vllm_srun.accel.cuda import CUDAAccelerator
from vllm_srun.accel.mps import MPSAccelerator
from vllm_srun.accel.rocm import ROCmAccelerator
from vllm_srun.cli import main
from vllm_srun.devices import report
from vllm_srun.plugins.base import DeviceInfo

GPU = torch.cuda.is_available()


def offer(monkeypatch, accelerator, count):
    """Make ``accelerator`` available with ``count`` devices."""
    found = [DeviceInfo(accelerator.name, index, "fake GPU") for index in range(count)]
    monkeypatch.setattr(accelerator, "available", lambda self: True)
    monkeypatch.setattr(accelerator, "devices", lambda self: found)


@pytest.mark.skipif(GPU, reason="checks a host without GPUs")
def test_the_devices_command_resolves_auto_to_the_cpu_without_gpus(capsys):
    assert main(["devices"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["auto"] == "cpu" and "cpu" in listed["devices"]


def test_auto_is_the_first_device_of_the_first_available_accelerator(monkeypatch):
    offer(monkeypatch, ROCmAccelerator, 2)
    listed = report()
    assert listed["auto"] == "rocm:0"
    assert {"cpu", "rocm:0", "rocm:1"} <= set(listed["devices"])
    offer(monkeypatch, CUDAAccelerator, 1)
    assert report()["auto"] == "rocm:0", "ROCm's auto_priority comes before CUDA's"
    monkeypatch.setattr(ROCmAccelerator, "available", lambda self: False)
    listed = report()
    assert listed["auto"] == "cuda:0" and "rocm:0" not in listed["devices"]


def test_only_an_accelerator_with_an_auto_priority_takes_auto(monkeypatch):
    monkeypatch.setattr(ROCmAccelerator, "available", lambda self: False)
    monkeypatch.setattr(CUDAAccelerator, "available", lambda self: False)
    monkeypatch.setattr(MPSAccelerator, "available", lambda self: True)
    listed = report()
    assert listed["auto"] == "cpu" and "mps" in listed["devices"]
    monkeypatch.setattr(MPSAccelerator, "auto_priority", -1)
    assert report()["auto"] == "mps", "a device without an index is its accelerator"
