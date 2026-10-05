import pytest
import torch
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.accel.cuda import CUDAAccelerator
from vllm_sr_runtime.accel.rocm import ROCmAccelerator
from vllm_sr_runtime.errors import PlacementError, UnsupportedDeviceError
from vllm_sr_runtime.placement import auto_order, device_kind, parse_device, place
from vllm_sr_runtime.plugins.base import BackboneSpec, DtypePolicy, ModelSpec

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


def test_xpu_and_mps_devices_parse():
    assert parse_device("xpu:1") == ("xpu", 1)
    assert parse_device("mps") == ("mps", None)


def test_device_names_and_the_auto_order_come_from_the_accelerators():
    assert auto_order() == ["rocm", "cuda", "cpu"]
    assert CPUAccelerator.descriptor()["auto_priority"] == 100
    with pytest.raises(PlacementError, match="one of cpu, cuda, mps, rocm, xpu"):
        parse_device("tpu:0")
    assert device_kind("rocm:1") == "rocm"
    if not GPU:
        assert device_kind("auto") == "cpu"


GATED_DELTA = ModelSpec(
    "Decision-2.0-Eos-0.8B",
    BackboneSpec("qwen3_5_text", {}, ()),
    DtypePolicy(),
    1024,
    requires={"cpu": ("lapack",)},
)


@pytest.mark.parametrize("lapack", [True, False])
def test_a_cpu_without_lapack_refuses_models_that_require_it(monkeypatch, lapack):
    monkeypatch.setattr(torch._C, "has_lapack", lapack)
    (device,) = CPUAccelerator().devices()
    assert CPUAccelerator().capabilities(device)["lapack"] is lapack
    assert place(SPEC, "cpu", 1000).device.accelerator == "cpu"
    if lapack:
        assert place(GATED_DELTA, "cpu", 1000).device.accelerator == "cpu"
        return
    with pytest.raises(UnsupportedDeviceError) as refused:
        place(GATED_DELTA, "cpu", 1000)
    message = str(refused.value)
    assert message.startswith("no device can serve Decision-2.0-Eos-0.8B: cpu: ")
    assert "built without LAPACK" in message
    assert "the router's CPU image" in message and "a GPU device" in message


def test_the_refusal_comes_before_any_weights_load_and_is_final(
    monkeypatch, qwen3_package, qwen35_package
):
    from vllm_sr_runtime.config import ModelConfig, ServeConfig
    from vllm_sr_runtime.engines.native.engine import NativeEngine
    from vllm_sr_runtime.runtime import Runtime

    import vllm_sr_runtime.runtime as runtime_module

    monkeypatch.setattr(torch._C, "has_lapack", False)
    loads, placements = [], []
    load, placed = NativeEngine.load, runtime_module.place

    def counted(self, spec, *args, **kwargs):
        loads.append(spec.name)
        return load(self, spec, *args, **kwargs)

    def placing(spec, *args, **kwargs):
        placements.append(spec.name)
        return placed(spec, *args, **kwargs)

    monkeypatch.setattr(NativeEngine, "load", counted)
    monkeypatch.setattr(runtime_module, "place", placing)
    models = (
        ModelConfig(model=str(qwen35_package), name="gated", device="cpu"),
        ModelConfig(model=str(qwen3_package), name="dense", device="cpu"),
    )
    runtime = Runtime(ServeConfig(models=models, load_retry_seconds=0.01))
    runtime.start(background=True)
    try:
        runtime.wait(timeout=60)
        gated, dense = runtime.lookup("gated"), runtime.lookup("dense")
        assert gated.health.state == "failed"
        assert gated.health.reason.startswith("UnsupportedDeviceError: ")
        assert "built without LAPACK" in gated.health.reason
        assert dense.health.ready
        assert loads == ["Decision-2.0-Tiny-Qwen3"]
        assert placements.count("Decision-2.0-Tiny-Qwen3.5") == 1
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("family", "variant", "requires"),
    [
        ("decision2", "qwen3_5", {"cpu": ("lapack",)}),
        ("decision2", "qwen3", {}),
        ("decision1", "qwen3.5-decision", {"cpu": ("lapack",)}),
        ("decision1", "vela-encoder", {}),
        ("vela2", "decoder", {"cpu": ("lapack",)}),
        ("vela2", "encoder", {}),
    ],
)
def test_the_qwen3_5_families_require_lapack_on_the_cpu(
    tmp_path, family, variant, requires
):
    from vllm_sr_runtime.plugins import registry
    from vllm_sr_runtime.plugins.base import PackageRef
    from vllm_sr_runtime.testing.fixtures import write_fixture

    package = write_fixture(tmp_path / "package", family=family, variant=variant)
    plugin = registry.plugin("families", family).load()()
    assert plugin.describe(plugin.verify(PackageRef(package))).requires == requires
