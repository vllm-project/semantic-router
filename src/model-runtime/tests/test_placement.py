from dataclasses import replace

import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.accel.cuda import CUDAAccelerator
from vllm_srun.accel.rocm import ROCmAccelerator
from vllm_srun.errors import PlacementError, UnsupportedDeviceError
from vllm_srun.placement import (
    auto_order,
    device_kind,
    estimate_bytes,
    parse_device,
    place,
)
from vllm_srun.plugins.base import BackboneSpec, DeviceInfo, DtypePolicy, ModelSpec

SPEC = ModelSpec("tiny", BackboneSpec("qwen3", {}, ()), DtypePolicy(), 1024)
GPU = torch.cuda.is_available()
GIB = 1 << 30


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


@pytest.mark.parametrize(
    ("device", "reason"),
    [
        pytest.param(
            "rocm:0",
            "rocm: not available on this host",
            marks=pytest.mark.skipif(GPU, reason="needs a host without ROCm"),
        ),
        ("cpu:7", "cpu: no device 7"),
    ],
)
def test_an_unusable_device_fails_before_the_model_is_resolved(
    monkeypatch, device, reason
):
    import vllm_srun.runtime as runtime_module
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    def resolve(*args, **kwargs):
        raise AssertionError("the model was resolved before the device was checked")

    monkeypatch.setattr(runtime_module, "resolve", resolve)
    model = ModelConfig(model="vllm-sr/Decision-2.0-Kai-0.6B", device=device)
    with pytest.raises(PlacementError, match=reason):
        Runtime(ServeConfig(models=(model,))).load()


def test_cuda_is_marked_unvalidated_and_rocm_validated():
    assert CUDAAccelerator.validated is False
    assert ROCmAccelerator.validated is True


@pytest.mark.gpu
@pytest.mark.skipif(not GPU, reason="needs a CUDA or ROCm device")
def test_gpu_devices_report_memory_and_bf16():
    accelerator = ROCmAccelerator() if torch.version.hip else CUDAAccelerator()
    devices = accelerator.devices()
    assert devices and devices[0].total_memory and devices[0].index == 0


def test_unvalidated_devices_parse():
    assert parse_device("xpu:1") == ("xpu", 1)
    assert parse_device("mps") == ("mps", None)
    assert parse_device("npu:1") == ("npu", 1)


def test_device_names_and_the_auto_order_come_from_the_accelerators():
    assert auto_order() == ["rocm", "cuda", "cpu"]
    assert CPUAccelerator.descriptor()["auto_priority"] == 100
    with pytest.raises(PlacementError, match="one of cpu, cuda, mps, npu, rocm, xpu"):
        parse_device("tpu:0")
    assert device_kind("rocm:1") == "rocm"
    if not GPU:
        assert device_kind("auto") == "cpu"


GATED_DELTA_BACKBONE = BackboneSpec("qwen3_5_text", {}, ())
GATED_DELTA = ModelSpec(
    "Decision-2.0-Eos-0.8B",
    GATED_DELTA_BACKBONE,
    DtypePolicy(),
    1024,
    requires=GATED_DELTA_BACKBONE.requires,
)


def test_only_the_qwen3_5_backbone_requires_lapack_on_the_cpu():
    assert GATED_DELTA_BACKBONE.requires == {"cpu": ("lapack",)}
    for model_type in ("qwen3", "modernbert"):
        assert BackboneSpec(model_type, {}, ()).requires == {}


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


def test_a_pytorch_without_the_private_lapack_flag_still_places_on_the_cpu(
    monkeypatch,
):
    monkeypatch.delattr(torch._C, "has_lapack")
    (device,) = CPUAccelerator().devices()
    assert CPUAccelerator().capabilities(device)["lapack"] is True
    assert place(GATED_DELTA, "cpu", 1000).device.accelerator == "cpu"


def test_the_refusal_comes_before_any_weights_load_and_is_final(
    monkeypatch, qwen3_package, qwen35_package
):
    import vllm_srun.runtime as runtime_module
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.runtime import Runtime

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
    from vllm_srun.plugins import registry
    from vllm_srun.plugins.base import PackageRef
    from vllm_srun.testing.fixtures import write_fixture

    package = write_fixture(tmp_path / "package", family=family, variant=variant)
    plugin = registry.plugin("families", family).load()()
    assert plugin.describe(plugin.verify(PackageRef(package))).requires == requires


D3_PARAMETERS = 26_086_635_760
D3 = ModelSpec(
    "d3",
    GATED_DELTA_BACKBONE,
    DtypePolicy(
        autocast=None,
        bf16_resident=False,
        gpu_weights="bfloat16",
        cpu_weights="bfloat16",
    ),
    262_144,
    requires=GATED_DELTA_BACKBONE.requires,
)
# Loaded parameters, and the device memory above the idle baseline that
# loading plus the model card's text, image and video requests reached in
# Engine mode on MI325X (docs/records/decision3-device-memory.md).
DECISION3_PEAKS_GIB = {
    "d3": (D3_PARAMETERS, 50.28),
    "d3-flash": (8_393_739_504, 17.21),
    "d3-mini": (4_539_918_336, 10.09),
    "d3-nano": (2_213_763_904, 5.74),
    "d3-lite": (853_247_040, 3.13),
    "d3-edge": (594_582_720, 2.61),
}


def test_the_estimate_counts_weights_in_the_dtype_the_device_holds():
    billion = 1_000_000_000
    assert estimate_bytes(SPEC, billion, "rocm") == 4 * billion + 2 * GIB
    assert estimate_bytes(SPEC, billion, "cpu") == 4 * billion + 2 * GIB
    assert estimate_bytes(D3, billion, "rocm") == 2_200_000_000 + 2 * GIB
    assert estimate_bytes(D3, billion, "cpu") == 2_200_000_000 + 2 * GIB
    fp32_on_cpu = replace(D3, dtype=replace(D3.dtype, cpu_weights=None))
    assert estimate_bytes(fp32_on_cpu, billion, "cpu") == 4 * billion + 2 * GIB


@pytest.mark.parametrize("model", sorted(DECISION3_PEAKS_GIB))
def test_the_estimate_covers_each_measured_decision3_peak_with_margin(model):
    parameters, peak = DECISION3_PEAKS_GIB[model]
    needed = estimate_bytes(D3, parameters, "rocm") / GIB
    assert needed >= 1.1 * peak and needed >= peak + 0.6
    assert needed < (4 * parameters + 2 * GIB) / GIB


def _one_rocm_device(monkeypatch, free_gib):
    device = DeviceInfo(
        accelerator="rocm",
        index=0,
        name="AMD Instinct MI300X",
        total_memory=192 * GIB,
        free_memory=int(free_gib * GIB),
        bf16=True,
    )
    monkeypatch.setattr(ROCmAccelerator, "available", lambda self: True)
    monkeypatch.setattr(ROCmAccelerator, "devices", lambda self: [device])
    monkeypatch.setattr(CUDAAccelerator, "available", lambda self: False)
    monkeypatch.setattr(torch._C, "has_lapack", False)


def test_d3_fits_the_free_memory_of_a_shared_gpu(monkeypatch):
    _one_rocm_device(monkeypatch, 70.2)
    placement = place(D3, "auto", D3_PARAMETERS)
    assert placement.device.label == "rocm:0"
    assert placement.estimated_bytes / GIB == pytest.approx(55.45, abs=0.01)


def test_a_full_gpu_states_need_and_free_memory_before_the_cpu(monkeypatch):
    _one_rocm_device(monkeypatch, 40.0)
    with pytest.raises(PlacementError) as refused:
        place(D3, "auto", D3_PARAMETERS)
    assert type(refused.value) is PlacementError
    message = str(refused.value)
    assert message.startswith(
        "no device can serve d3: rocm:0: d3 needs about 55.4 GiB, 40.0 GiB free; cpu: "
    )
    assert "built without LAPACK" in message
    assert "cuda" not in message


def test_a_host_without_gpus_still_names_the_accelerators_it_lacks(monkeypatch):
    monkeypatch.setattr(ROCmAccelerator, "available", lambda self: False)
    monkeypatch.setattr(CUDAAccelerator, "available", lambda self: False)
    monkeypatch.setattr(torch._C, "has_lapack", False)
    with pytest.raises(PlacementError) as refused:
        place(D3, "auto", D3_PARAMETERS)
    assert str(refused.value).startswith(
        "no device can serve d3: rocm: not available on this host; "
        "cuda: not available on this host; cpu: this PyTorch is built without LAPACK"
    )
