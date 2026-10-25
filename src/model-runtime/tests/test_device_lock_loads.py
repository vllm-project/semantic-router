"""A model's weights are read before its device is taken: the device's other models answer meanwhile.

``LockedHost`` stands in for a GPU on a CPU-only host: its device work holds
the GPU device lock (``GPUAccelerator.execute``) while its tensors stay on the
host. A second model's checkpoint read is held open while the first model,
already serving on the same device, answers a request.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm_srun.accel.gpu import GPUAccelerator
from vllm_srun.accel.kernels import KernelSet, reference_kernels
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native import engine as native_engine
from vllm_srun.plugins import registry
from vllm_srun.plugins.base import (
    DeviceInfo,
    Engine,
    EngineOptions,
    PackageRef,
    RegistryOptions,
)
from vllm_srun.runtime import Runtime
from vllm_srun.testing import omni
from vllm_srun.testing.fixtures import write_fixture

from .conftest import QUESTIONS, STATE

DEVICE = "locked_host"
WAIT_S = 60.0


class LockedHost(GPUAccelerator):
    """A GPU's locking on the host's memory and kernels."""

    name = DEVICE
    validated = False

    def available(self) -> bool:
        return True

    def devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(DEVICE, 0, "host behind a device lock", bf16=True)]

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cpu")

    def kernels(self, device: DeviceInfo) -> KernelSet:
        return reference_kernels("cpu")

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {}

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        return nullcontext()

    def synchronize(self, device: DeviceInfo) -> None:
        pass


@pytest.fixture()
def locked_host(monkeypatch):
    found = {kind: dict(entries) for kind, entries in registry.discover().items()}
    found["accelerators"][DEVICE] = registry.PluginEntry(
        "accelerators", DEVICE, f"{__name__}:LockedHost", None, None
    )
    monkeypatch.setattr(registry, "discover", lambda: found)


@pytest.fixture()
def held_read(monkeypatch, qwen35_package):
    """Hold the read of ``qwen35_package``'s checkpoint open until ``release`` is set."""
    reading, release = threading.Event(), threading.Event()
    load_backbone = native_engine.load_backbone

    def gated(module, files, *args, **kwargs):
        if any(Path(path).is_relative_to(qwen35_package) for path in files):
            reading.set()
            assert release.wait(WAIT_S), "the test never released the read"
        return load_backbone(module, files, *args, **kwargs)

    monkeypatch.setattr(native_engine, "load_backbone", gated)
    yield reading, release
    release.set()


def serve_two(serving: Path, loading: Path) -> Runtime:
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(serving), name="serving", device=DEVICE),
                ModelConfig(model=str(loading), name="loading", device=DEVICE),
            )
        )
    )
    runtime.start(background=True)
    return runtime


async def ask(runtime: Runtime, timeout: float) -> tuple[int, dict[str, Any]]:
    body = {"model": "serving", "state": STATE, "questions": QUESTIONS}
    return await asyncio.wait_for(runtime.call("decisions", body), timeout)


def test_a_model_answers_while_another_on_its_device_reads_its_weights(
    locked_host, held_read, qwen3_package, qwen35_package
):
    reading, release = held_read
    runtime = serve_two(qwen3_package, qwen35_package)
    try:
        assert reading.wait(WAIT_S), "the second model never read its weights"
        serving, loading = runtime.lookup("serving"), runtime.lookup("loading")
        assert serving.health.ready

        status, body = asyncio.run(ask(runtime, timeout=WAIT_S))

        assert status == 200 and set(body["answers"]) == set(QUESTIONS)
        assert loading.health.state == "loading"
        release.set()
        assert runtime.wait(WAIT_S) and loading.health.ready
        assert serving.placement.device == loading.placement.device
    finally:
        release.set()
        runtime.stop()


@pytest.mark.parametrize(
    ("family", "write"),
    [
        ("multimodal_embedding", lambda root, _: omni.write_snapshot(root)),
        ("decision2", lambda root, adapter: adapter),
        (
            "decision1",
            lambda root, _: write_fixture(
                root, family="decision1", variant="vela-encoder"
            ),
        ),
    ],
    ids=["towers", "lora", "branches"],
)
def test_the_device_step_after_a_read_builds_the_model_load_builds(
    tmp_path, adapter_package, family, write
):
    base = adapter_package.parent / f"{adapter_package.name}-base"
    plugin = registry.plugin("families", family).load()(RegistryOptions(base_path=base))
    spec = plugin.describe(
        plugin.verify(PackageRef(write(tmp_path / "pkg", adapter_package)))
    )
    accelerator = LockedHost()
    (device,) = accelerator.devices()
    engine, options = native_engine.NativeEngine(), EngineOptions(threads=2)
    split = engine.read(spec, accelerator, device, options)()
    whole = engine.load(spec, accelerator, device, options)

    def tensors(model: Any) -> dict[str, torch.Tensor]:
        return {
            f"{index}.{name}": tensor
            for index, module in enumerate(model._modules())
            for name, tensor in (*module.named_parameters(), *module.named_buffers())
        }

    read, loaded = tensors(split), tensors(whole)
    assert len(split._modules()) == 1 + len(spec.backbone.branches) + len(spec.towers)
    assert read.keys() == loaded.keys()
    assert all(torch.equal(read[name], loaded[name]) for name in read)


def test_an_engine_without_a_read_step_reads_as_device_work(
    locked_host, held_read, monkeypatch, qwen3_package, qwen35_package
):
    """The default ``Engine.read`` keeps the whole load device work (engines written before the step)."""
    monkeypatch.setattr(native_engine.NativeEngine, "read", Engine.read)
    reading, release = held_read
    runtime = serve_two(qwen3_package, qwen35_package)
    try:
        assert reading.wait(WAIT_S)
        assert runtime.lookup("serving").health.ready
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(ask(runtime, timeout=0.5))
        release.set()
        assert runtime.wait(WAIT_S)
        status, _ = asyncio.run(ask(runtime, timeout=WAIT_S))
        assert status == 200
    finally:
        release.set()
        runtime.stop()
