import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.accel.cuda import CUDAAccelerator
from vllm_srun.accel.rocm import ROCmAccelerator
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import PlacementError
from vllm_srun.placement import place
from vllm_srun.plugins.base import BackboneSpec, DeviceInfo, DtypePolicy, ModelSpec

POLICIES = [
    pytest.param(
        DtypePolicy(
            autocast=None,
            bf16_resident=False,
            gpu_weights="bfloat16",
            cpu_weights="bfloat16",
        ),
        True,
        id="decision3-bf16-weights-without-autocast",
    ),
    pytest.param(DtypePolicy(), True, id="bf16-autocast"),
    pytest.param(DtypePolicy(autocast=None, bf16_resident=False), False, id="fp32"),
    pytest.param(
        DtypePolicy(autocast=None, bf16_resident=False, gpu_weights="float16"),
        False,
        id="fp16-weights",
    ),
    pytest.param(
        DtypePolicy(autocast=None, bf16_resident=False, cpu_weights="bfloat16"),
        False,
        id="cpu-only-bf16-weights",
    ),
]


def model_spec(dtype):
    return ModelSpec("test", BackboneSpec("qwen3_5_text", {}, ()), dtype, 1024)


def mock_device(monkeypatch, accelerator, device):
    monkeypatch.setattr(accelerator, "available", lambda self: True)
    monkeypatch.setattr(accelerator, "devices", lambda self: [device])


@pytest.mark.parametrize("accelerator", [CUDAAccelerator, ROCmAccelerator])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(("dtype", "needs_bf16"), POLICIES)
def test_placement_checks_gpu_weight_dtype(
    monkeypatch, accelerator, bf16, dtype, needs_bf16
):
    device = DeviceInfo(accelerator.name, 0, "synthetic GPU", bf16=bf16)
    mock_device(monkeypatch, accelerator, device)
    if needs_bf16 and not bf16:
        with pytest.raises(PlacementError, match="no BF16 support"):
            place(model_spec(dtype), device.label, parameters=1)
    else:
        assert place(model_spec(dtype), device.label, parameters=1).device == device


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(("dtype", "needs_bf16"), POLICIES)
def test_native_engine_checks_gpu_weight_dtype(accelerator, bf16, dtype, needs_bf16):
    device = DeviceInfo(accelerator, 0, "synthetic GPU", bf16=bf16)
    reason = NativeEngine().supports(model_spec(dtype), device)
    if needs_bf16 and not bf16:
        assert reason == f"{device.label} has no BF16 support"
    else:
        assert reason is None


def test_auto_placement_falls_back_to_cpu_for_bf16_weights(monkeypatch):
    for accelerator in (ROCmAccelerator, CUDAAccelerator):
        mock_device(
            monkeypatch,
            accelerator,
            DeviceInfo(accelerator.name, 0, "legacy GPU", bf16=False),
        )
    cpu = DeviceInfo("cpu", None, "CPU", bf16=False)
    mock_device(monkeypatch, CPUAccelerator, cpu)
    spec = model_spec(
        DtypePolicy(autocast=None, gpu_weights="bfloat16", cpu_weights="bfloat16")
    )
    assert place(spec, "auto", parameters=1).device == cpu
    assert NativeEngine().supports(spec, cpu) is None


def test_decision3_refuses_legacy_gpu_before_reading_weights(
    monkeypatch, d3_text_package
):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    mock_device(
        monkeypatch, CUDAAccelerator, DeviceInfo("cuda", 0, "legacy GPU", bf16=False)
    )

    def unexpected_read(*args, **kwargs):
        pytest.fail("BF16-ineligible device reached the native weight loader")

    monkeypatch.setattr(NativeEngine, "read", unexpected_read)
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(d3_text_package), device="cuda"),))
    )
    try:
        with pytest.raises(PlacementError, match="cuda:0: no BF16 support"):
            runtime.load()
    finally:
        runtime.stop()
