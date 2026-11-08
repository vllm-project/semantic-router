from dataclasses import replace

import pytest
import torch
from vllm_srun.accel import gpu
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
from vllm_srun.quantization import FLOAT8, FLOAT8_DEVICES, QuantizedWeights

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


def test_cuda_and_rocm_are_validated():
    assert CUDAAccelerator.validated is True
    assert ROCmAccelerator.validated is True
    assert CUDAAccelerator.descriptor()["validated"] is True


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


def test_nvfp4_weights_require_fp4_tensor_cores_everywhere():
    from vllm_srun.plugins.registry import names
    from vllm_srun.quantization import NVFP4, QuantizedWeights

    nvfp4 = QuantizedWeights(NVFP4, targets=("Linear",), ignore=(), group_size=16)
    everywhere = dict.fromkeys(names("accelerators"), ("nvfp4",))
    assert BackboneSpec("qwen3_5_text", {}, (), quantized=nvfp4).requires == {
        **everywhere,
        "cpu": ("lapack", "nvfp4"),
    }
    assert BackboneSpec("qwen3", {}, (), quantized=nvfp4).requires == everywhere


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


def test_a_bf16_resident_family_counts_its_bf16_linear_weights_on_gpus():
    billion, linear = 1_000_000_000, 750_000_000
    held = 1_500_000_000 + 150_000_000 + 4 * 250_000_000
    assert estimate_bytes(SPEC, billion, "cuda", linear) == held + 2 * GIB
    assert estimate_bytes(SPEC, billion, "cpu", linear) == 4 * billion + 2 * GIB
    fp32 = replace(SPEC, dtype=replace(SPEC.dtype, bf16_resident=False))
    assert estimate_bytes(fp32, billion, "cuda", linear) == 4 * billion + 2 * GIB
    assert estimate_bytes(D3, billion, "cuda", linear) == 2_200_000_000 + 2 * GIB


# docs/records/bf16-resident-device-memory.md: parameters, BF16-stored Linear
# parameters and the peak with short requests on an RTX 4090, which had
# 22.8 GiB free.
BF16_RESIDENT_PEAKS_GIB = {
    "Decision-2.0-Lux-9B": (7_940_895_744, 6_918_504_448, 19.03),
    "Vela-2.0-9B": (7_949_850_627, 6_918_504_448, 18.48),
}
RTX_4090_FREE_GIB = 22.8


@pytest.mark.parametrize("model", sorted(BF16_RESIDENT_PEAKS_GIB))
def test_the_estimate_covers_each_measured_bf16_resident_peak_on_a_24_gb_gpu(model):
    parameters, bf16, peak = BF16_RESIDENT_PEAKS_GIB[model]
    needed = estimate_bytes(SPEC, parameters, "cuda", bf16) / GIB
    assert peak + 0.6 <= needed < RTX_4090_FREE_GIB


def test_an_explicit_dtype_is_estimated_as_it_holds_the_weights(monkeypatch):
    _one_rocm_device(monkeypatch, 22.8)
    parameters, bf16, _ = BF16_RESIDENT_PEAKS_GIB["Decision-2.0-Lux-9B"]
    published = place(SPEC, "rocm:0", parameters, bf16_parameters=bf16)
    with pytest.raises(PlacementError, match=r"needs about 31\.6 GiB, 22\.8 GiB free"):
        place(SPEC, "rocm:0", parameters, bf16_parameters=bf16, dtype="float32")
    # BF16 autocast is the published precision on a GPU, so bfloat16 keeps it.
    held = place(SPEC, "rocm:0", parameters, bf16_parameters=bf16, dtype="bfloat16")
    assert held.estimated_bytes == published.estimated_bytes
    _one_rocm_device(monkeypatch, 70.2)
    with pytest.raises(PlacementError, match=r"needs about 99\.2 GiB, 70\.2 GiB free"):
        place(D3, "rocm:0", D3_PARAMETERS, dtype="float32")


def test_float32_needs_no_bf16_support(monkeypatch):
    _one_rocm_device(monkeypatch, 40.0)
    (device,) = ROCmAccelerator().devices()
    monkeypatch.setattr(
        ROCmAccelerator, "devices", lambda self: [replace(device, bf16=False)]
    )
    with pytest.raises(PlacementError, match="no BF16 support"):
        place(SPEC, "rocm:0", 1_000_000)
    assert place(SPEC, "rocm:0", 1_000_000, dtype="float32").device.index == 0


# "all" is a stock checkpoint (the Qwen base under Decision-2.0-Vega-27B): its
# embeddings, norms and convolutions are BF16 too, but stay FP32 on the GPU.
@pytest.mark.parametrize("stored", ["linear", "all"])
def test_the_bf16_count_is_what_the_native_engine_holds_on_a_gpu(tmp_path, stored):
    from vllm_srun.engines.native.engine import (
        bf16_resident_parameters,
        build_on_meta,
    )
    from vllm_srun.engines.native.weights import keep_linear_bf16, load_backbone
    from vllm_srun.testing.fixtures import qwen3_5_config, random_backbone, save

    config = qwen3_5_config(384)
    state = random_backbone("qwen3_5_text", config, seed=3)
    if stored == "all":
        state = {name: value.to(torch.bfloat16) for name, value in state.items()}
    save({f"backbone.{n}": v for n, v in state.items()}, tmp_path / "m.safetensors")
    spec = BackboneSpec(
        config["model_type"], config, (tmp_path / "m.safetensors",), "backbone."
    )
    module = build_on_meta(spec)
    load_backbone(module, spec.weight_files, spec.weight_prefix)
    keep_linear_bf16(module)
    held = sum(p.numel() for p in module.parameters() if p.dtype == torch.bfloat16)
    assert 0 < held < sum(p.numel() for p in module.parameters())
    assert bf16_resident_parameters(spec) == held


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


def test_an_fp8_package_refuses_devices_without_fp8_units_before_loading(
    monkeypatch,
):
    _one_rocm_device(monkeypatch, 70.2)
    backbone = replace(D3.backbone, quantized=QuantizedWeights(FLOAT8, ("Linear",), ()))
    fp8 = replace(D3, backbone=backbone, requires=backbone.requires)
    monkeypatch.setattr(gpu, "float8_format", lambda device: torch.float8_e4m3fnuz)
    assert place(fp8, "rocm:0", D3_PARAMETERS).device.label == "rocm:0"
    monkeypatch.setattr(gpu, "float8_format", lambda device: None)
    for device in ("rocm:0", "cpu"):
        with pytest.raises(UnsupportedDeviceError) as refused:
            place(fp8, device, D3_PARAMETERS)
        assert f"{device}: {FLOAT8_DEVICES}" in str(refused.value)


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
