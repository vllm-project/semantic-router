"""The quantized layers on a GPU: int4 unpacking against the CPU, FP8 on its matrix units, and packages' answers."""

import asyncio

import pytest
import torch
from torch import nn
from vllm_srun.engines.native.quantized import Int4Linear
from vllm_srun.testing.decision3 import pack_int4, write_quantized_fixture

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("needs Triton")
    return torch.device("cuda:0")


def layer(rows, columns, group, symmetric, seed=0):
    torch.manual_seed(seed)
    module = Int4Linear(nn.Linear(columns, rows, bias=False), group, symmetric)
    module.weight_packed.copy_(pack_int4(torch.randint(-8, 8, (rows, columns))))
    module.weight_scale.copy_(torch.rand(rows, columns // group) * 0.02 + 1e-3)
    if not symmetric:
        zero = torch.randint(-8, 8, (rows, columns // group))
        module.weight_zero_point.copy_(pack_int4(zero, dim=0))
    return module


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("shape", [(24, 64, 32), (13, 256, 128), (5120, 1024, 128)])
def test_the_kernel_unpacks_as_the_cpu_does(device, symmetric, shape):
    rows, columns, group = shape
    module = layer(rows, columns, group, symmetric)
    want = module.unpacked(torch.bfloat16)
    got = module.to(device).unpacked(torch.bfloat16)
    assert torch.equal(got.cpu(), want)


def test_the_layer_multiplies_by_the_unpacked_weight(device):
    module = layer(96, 256, 128, symmetric=False).to(device)
    x = torch.randn(7, 256, device=device, dtype=torch.bfloat16)
    want = x @ module.unpacked(torch.bfloat16).t()
    assert torch.equal(module(x), want)


def test_the_layer_replays_in_a_graph(device):
    module = layer(64, 128, 32, symmetric=True).to(device)
    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    eager = module(x)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(x)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = module(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("symmetric", [True, False])
def test_an_int4_package_answers_as_its_dequantized_weights_do(
    device, tmp_path, d3_text_package, symmetric
):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    from tests.test_quantized import QUESTIONS, STATES

    def answers(package):
        name = "rocm:0" if torch.version.hip else "cuda:0"
        runtime = Runtime(
            ServeConfig(models=(ModelConfig(model=str(package), device=name),))
        )
        runtime.start(background=False)
        try:
            return [
                asyncio.run(
                    runtime.call("decisions", {"state": state, "questions": QUESTIONS})
                )
                for state in STATES
            ]
        finally:
            runtime.stop()

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "int4", symmetric=symmetric
    )
    plain = write_quantized_fixture(
        d3_text_package, tmp_path / "plain", symmetric=symmetric, dequantized=True
    )
    got, want = answers(packed), answers(plain)
    assert all(status == 200 for status, _ in got + want)
    assert [body["answers"] for _, body in got] == [body["answers"] for _, body in want]


def gpu_answers(package):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    from tests.test_quantized import QUESTIONS, STATES

    name = "rocm:0" if torch.version.hip else "cuda:0"
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=name),))
    )
    runtime.start(background=False)
    try:
        bodies = [
            asyncio.run(runtime.call("decisions", {"state": s, "questions": QUESTIONS}))
            for s in STATES
        ]
        engine = runtime.lookup(None).model.engine_model
        return bodies, engine.parameter_count(), engine.memory_bytes()
    finally:
        runtime.stop()


def numbers(bodies):
    out = []
    for status, body in bodies:
        assert status == 200, body
        for answer in body["answers"].values():
            out.extend(
                answer["probabilities"].values()
                if "probabilities" in answer
                else [answer["noul"]]
            )
    return torch.tensor(out)


def test_an_fp8_package_answers_close_to_its_weights_in_bf16(
    device, tmp_path, d3_text_package
):
    from vllm_srun.engines.native.quantized import float8_format

    if float8_format(device) is None:
        pytest.skip("needs FP8 matrix units")
    packed = write_quantized_fixture(d3_text_package, tmp_path / "fp8", scheme="float8")
    plain = write_quantized_fixture(
        d3_text_package, tmp_path / "plain", scheme="float8", dequantized=True
    )
    got, parameters, size = gpu_answers(packed)
    want, plain_parameters, plain_size = gpu_answers(plain)
    # only the activations' FP8 rounding separates them
    assert (numbers(got) - numbers(want)).abs().max() < 0.05
    assert numbers(gpu_answers(packed)[0]).equal(numbers(got))
    assert parameters == plain_parameters and size < plain_size


def test_mi300_reads_e4m3fn_weights_without_changing_a_value(device):
    from vllm_srun.engines.native.quantized import Float8Linear, float8_format

    if float8_format(device) != torch.float8_e4m3fnuz:
        pytest.skip("needs a gfx942 GPU")
    torch.manual_seed(0)
    module = Float8Linear(nn.Linear(64, 32, bias=False)).to(device)
    stored = (torch.randn(32, 64, device=device) * 100).to(torch.float8_e4m3fn)
    stored[0, 0] = -0.0
    module.weight = stored
    module.weight_scale = torch.rand(32, 1, device=device)
    want = stored.float() * module.weight_scale
    module.prepare()
    assert module.weight.dtype == torch.float8_e4m3fnuz
    assert torch.equal(module.weight.float() * module.weight_scale, want)


@pytest.mark.parametrize("rows", [1, 16, 37, 256])
@pytest.mark.parametrize("symmetric", [True, False])
def test_the_fused_int4_kernel_computes_what_the_unpacked_product_does(
    device, rows, symmetric
):
    from vllm_srun.accel import triton_int4

    module = layer(320, 512, 128, symmetric).to(device)
    x = torch.randn(rows, 512, device=device, dtype=torch.bfloat16)
    want = (x @ module.unpacked(torch.bfloat16).t()).float()
    got = triton_int4.matmul(
        x, module.weight_packed, module.weight_scale, module.weight_zero_point, 128
    ).float()
    # the same products, summed in another order
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-2


def test_only_approximate_kernels_take_the_fused_int4_kernel(device):
    module = layer(96, 256, 128, symmetric=True).to(device)
    x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    exact = module(x)
    assert torch.equal(exact, x @ module.unpacked(torch.bfloat16).t())
    module.fused = True
    assert (
        module(x).float() - exact.float()
    ).abs().max() < 1e-2 * exact.float().abs().max()
    big = torch.randn(512, 256, device=device, dtype=torch.bfloat16)
    assert torch.equal(module(big), big @ module.unpacked(torch.bfloat16).t())


def test_the_unscaled_fp8_product_stays_within_a_few_bf16_ulps(device):
    from vllm_srun.engines.native.quantized import (
        UNSCALED_ROWS,
        Float8Linear,
        float8_format,
    )

    fmt = float8_format(device)
    if fmt is None:
        pytest.skip("needs FP8 matrix units")
    torch.manual_seed(0)
    # d3's k_proj shape, activations with outlier channels
    module = Float8Linear(nn.Linear(5120, 1024, bias=False)).to(device)
    # within e4m3fnuz's range (240): larger values turn NaN on MI300
    module.weight = (torch.randn(1024, 5120, device=device) * 30).to(fmt)
    module.weight_scale = torch.rand(1024, 1, device=device) * 0.01
    module.prepare()
    module.one = torch.ones((), device=device)
    x = torch.randn(UNSCALED_ROWS, 5120, device=device, dtype=torch.bfloat16)
    x[:, :8] *= 30

    def product(rows, unscaled):
        module.unscaled_product = unscaled
        return module(rows).float()

    scaled = product(x, False)
    difference = (product(x, True) - scaled).abs()
    ulp = torch.finfo(torch.bfloat16).eps * scaled.abs()
    # outputs near zero come from cancelling sums, whose relative error grows
    away_from_zero = scaled.abs() > 1e-2 * scaled.abs().max()
    assert (difference <= 4 * ulp)[away_from_zero].all()
    # fewer rows keep the scaled product
    few = x[: UNSCALED_ROWS - 1]
    assert torch.equal(product(few, True), product(few, False))


def test_only_approximate_kernels_take_the_unscaled_fp8_product(
    device, tmp_path, d3_text_package
):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.engines.native import quantized
    from vllm_srun.runtime import Runtime

    if torch.version.hip or torch.cuda.get_device_capability(device) != (8, 9):
        pytest.skip("only Ada takes the FP8 product unscaled")
    packed = write_quantized_fixture(d3_text_package, tmp_path / "fp8", scheme="float8")
    # an explicit --dtype other than the published one runs exact kernels only
    for dtype, unscaled in (("auto", True), ("float16", False)):
        runtime = Runtime(
            ServeConfig(
                models=(ModelConfig(model=str(packed), device="cuda:0", dtype=dtype),)
            )
        )
        runtime.start(background=False)
        try:
            backbone = runtime.lookup(None).model.engine_model.backbone
            layers = quantized.layers(backbone)
            assert layers and {layer.unscaled_product for layer in layers} == {unscaled}
        finally:
            runtime.stop()


def test_fp8_quantizes_rows_past_two_billion_elements(device):
    from vllm_srun.accel.triton_fp8 import quantize_rows
    from vllm_srun.engines.native.quantized import float8_format

    dtype = float8_format(device)
    if dtype is None:
        pytest.skip("needs FP8 matrix units")
    columns = 4096
    rows = 2**31 // columns + 2
    x = torch.zeros(rows, columns, device=device, dtype=torch.bfloat16)
    x[-1] = torch.linspace(-3, 3, columns, device=device)
    q, scale = quantize_rows(x, dtype)
    expected = x[-1].float().abs().max() / torch.finfo(dtype).max
    assert scale[-1].item() == pytest.approx(expected.item(), rel=1e-6)
    torch.testing.assert_close(
        q[-1].float() * scale[-1], x[-1].float(), atol=0.07, rtol=0.07
    )
