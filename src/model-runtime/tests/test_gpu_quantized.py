"""The quantized layers on a GPU: int4 unpacking against the CPU, FP8 and NVFP4 on their matrix units, and packages' answers."""

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
    # captured in inference mode, as the engine captures its graphs
    with torch.inference_mode():
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


def gpu_answers(package, dtype=None):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    from tests.test_quantized import QUESTIONS, STATES

    name = "rocm:0" if torch.version.hip else "cuda:0"
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=name, dtype=dtype),))
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


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_an_fp8_package_answers_at_another_dtype(
    device, tmp_path, d3_text_package, dtype
):
    from vllm_srun.engines.native import quantized

    if quantized.float8_format(device) is None:
        pytest.skip("needs FP8 matrix units")
    packed = write_quantized_fixture(d3_text_package, tmp_path / "fp8", scheme="float8")
    want = numbers(gpu_answers(packed)[0])
    got = numbers(gpu_answers(packed, dtype)[0])
    # the product is the same BF16 one; only the layers around it change dtype
    assert (got - want).abs().max() < 0.05


def test_a_whole_model_cast_keeps_the_fp8_weights(device):
    from vllm_srun.engines.native.quantized import Float8Linear

    module = Float8Linear(nn.Linear(64, 32, bias=False))
    module.float().to(device)
    assert module.weight.dtype == torch.float8_e4m3fn
    assert module.weight_scale.dtype == torch.float32


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
    # fewer rows keep the scaled product where a graph can run it (PyTorch 2.12)
    if module.unscaled_rows:
        few = x[: module.unscaled_rows - 1]
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


def test_later_calls_launch_the_compiled_kernel_and_write_the_same_bytes(device):
    from vllm_srun.accel import triton_fp8
    from vllm_srun.engines.native.quantized import float8_format

    dtype = float8_format(device)
    if dtype is None:
        pytest.skip("needs FP8 matrix units")
    x = torch.randn(37, 3 * 1024 + 16, device=device, dtype=torch.bfloat16)
    first = triton_fp8.quantize_rows(x, dtype)
    assert [k for k, v in triton_fp8._launches.items() if k[3] == x.shape[1] and v]
    later = triton_fp8.quantize_rows(x, dtype)
    assert torch.equal(later[0].view(torch.uint8), first[0].view(torch.uint8))
    assert torch.equal(later[1], first[1])


def test_a_triton_without_knobs_launches_through_the_jit(device, monkeypatch):
    import triton
    from vllm_srun.accel import triton_fp8
    from vllm_srun.engines.native.quantized import float8_format

    dtype = float8_format(device)
    if dtype is None:
        pytest.skip("needs FP8 matrix units")
    x = torch.randn(37, 2 * 1024 + 16, device=device, dtype=torch.bfloat16)
    first = triton_fp8.quantize_rows(x, dtype)
    monkeypatch.delattr(triton, "knobs")
    monkeypatch.setattr(triton_fp8, "_launch", None)  # the direct launch is not taken
    later = triton_fp8.quantize_rows(x, dtype)
    assert torch.equal(later[0].view(torch.uint8), first[0].view(torch.uint8))
    assert torch.equal(later[1], first[1])


@pytest.fixture
def blackwell(device):
    from vllm_srun.accel.gpu import nvfp4_unavailable

    reason = nvfp4_unavailable(device.index)
    if reason:
        pytest.skip(reason)
    return device


def nvfp4_layer(rows, columns, device, seed=0):
    """An NVFP4Linear holding a random weight quantized by the reference, its input global scale for N(0, 1) inputs."""
    from vllm_srun.accel import nvfp4
    from vllm_srun.engines.native.quantized import NVFP4Linear

    torch.manual_seed(seed)
    weight = torch.randn(rows, columns) * 0.05
    module = NVFP4Linear(nn.Linear(columns, rows, bias=False))
    global_ = nvfp4.global_scale(weight)
    packed, scale = nvfp4.quantize(weight, global_)
    module.weight_packed.copy_(packed)
    module.weight_scale.copy_(scale)
    module.weight_global_scale.copy_(global_)
    module.input_global_scale.fill_(448 * 6 / 5.0)  # N(0, 1) blocks rarely pass 5
    module.to(device).prepare()
    return module, nvfp4.dequantize(packed, scale, global_).to(device)


@pytest.mark.parametrize("shape", [(1, 64), (37, 256), (300, 5120), (8, 17408)])
def test_nvfp4_activations_quantize_as_the_reference_does(blackwell, shape):
    from vllm_srun.accel import nvfp4, triton_nvfp4

    torch.manual_seed(0)
    x = (torch.randn(*shape, device=blackwell) * 3).to(torch.bfloat16)
    global_ = (
        nvfp4.global_scale(x) * 0.7
    )  # below this input's range: some blocks saturate
    codes, scales = triton_nvfp4.quantize(x, float(global_))
    want_codes, want_scales = nvfp4.quantize(x, global_)
    assert codes.equal(want_codes)
    # the kernel writes cuBLAS's padded 32x4x4 layout, zero where it pads
    assert scales.view(torch.uint8).equal(nvfp4.swizzle(want_scales).view(torch.uint8))


def test_later_nvfp4_calls_launch_the_compiled_kernel_and_write_the_same_bytes(
    blackwell,
):
    from vllm_srun.accel import nvfp4, triton_nvfp4

    torch.manual_seed(0)
    x = torch.randn(300, 2048, device=blackwell, dtype=torch.bfloat16)
    first = triton_nvfp4.quantize(x, 100.0)
    assert [v for k, v in triton_nvfp4._launches.items() if k[2] == 2048 and v]
    again = triton_nvfp4.quantize(x, 100.0)  # the direct launch
    assert again[0].equal(first[0])
    assert again[1].view(torch.uint8).equal(first[1].view(torch.uint8))
    for rows in (1, 37, 129):  # every row count takes the same compiled kernel
        codes, scales = triton_nvfp4.quantize(x[:rows].contiguous(), 100.0)
        want_codes, want_scales = nvfp4.quantize(
            x[:rows], torch.tensor([100.0], device=blackwell)
        )
        assert codes.equal(want_codes)
        assert scales.view(torch.uint8).equal(
            nvfp4.swizzle(want_scales).view(torch.uint8)
        )
    assert len([k for k in triton_nvfp4._launches if k[2] == 2048]) == 1


def test_a_triton_without_knobs_launches_nvfp4_through_the_jit(blackwell, monkeypatch):
    import triton
    from vllm_srun.accel import triton_nvfp4

    x = torch.randn(37, 2048, device=blackwell, dtype=torch.bfloat16)
    first = triton_nvfp4.quantize(x, 100.0)
    monkeypatch.delattr(triton, "knobs")
    monkeypatch.setattr(triton_nvfp4, "_launch", None)  # the direct launch is not taken
    later = triton_nvfp4.quantize(x, 100.0)
    assert later[0].equal(first[0])
    assert later[1].view(torch.uint8).equal(first[1].view(torch.uint8))


@pytest.mark.parametrize(
    "shape", [(32, 64, 1), (256, 512, 37), (5120, 17408, 300), (1024, 2048, 9000)]
)
def test_the_nvfp4_layer_multiplies_the_fp4_values_it_holds(blackwell, shape):
    from vllm_srun.accel import nvfp4

    rows, columns, inputs = shape
    module, weight = nvfp4_layer(rows, columns, blackwell)
    x = torch.randn(inputs, columns, device=blackwell).to(torch.bfloat16)
    got = module(x).float()
    codes, scales = nvfp4.quantize(x.float(), module.input_global_scale)
    want = nvfp4.dequantize(codes, scales, module.input_global_scale) @ weight.T
    # the same FP4 products in FP32; the layer writes BF16 (9000 rows: in slices)
    assert ((got - want).norm() / want.norm()) < 4e-3


def test_the_nvfp4_layer_replays_in_a_graph(blackwell):
    module, _ = nvfp4_layer(256, 512, blackwell)
    x = torch.randn(37, 512, device=blackwell).to(torch.bfloat16)
    want = module(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = module(x)
    graph.replay()
    torch.cuda.synchronize()
    assert got.equal(want)


def captured(module, x):
    """``module(x)`` captured in a graph and replayed once."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = module(x)
    graph.replay()
    torch.cuda.synchronize()
    return got


@pytest.fixture
def mm_fp4(blackwell):
    from vllm_srun.accel import flashinfer_nvfp4

    product, reason = flashinfer_nvfp4.load(blackwell)
    if product is None:
        pytest.skip(reason)
    return product


@pytest.mark.parametrize(
    "shape", [(384, 256, 1), (384, 256, 37), (5120, 17408, 300), (34816, 5120, 512)]
)
def test_flashinfer_multiplies_nvfp4_as_cublas_does(blackwell, mm_fp4, shape):
    from vllm_srun.engines.native import quantized

    rows, columns, inputs = shape
    module, _ = nvfp4_layer(rows, columns, blackwell)
    x = torch.randn(inputs, columns, device=blackwell).to(torch.bfloat16)
    q, scale = module.quantize(x, module.input_global)
    got = module._flashinfer_product(mm_fp4, q, scale).float()
    want = module._product(q, scale).float()
    # the same FP4 products, summed in another order
    assert ((got - want).norm() / want.norm()) < quantized.FLASHINFER_TOLERANCE
    generator = torch.Generator(device=blackwell).manual_seed(0)
    assert module.flashinfer_disagrees(mm_fp4, generator) is None


def test_an_autotune_cache_records_flashinfer_tiles_once_per_shape(
    blackwell, mm_fp4, tmp_path, monkeypatch
):
    import importlib

    from vllm_srun.accel import autotune, flashinfer_nvfp4

    flashinfer = importlib.import_module("flashinfer")
    monkeypatch.setitem(autotune._FROZEN, "directory", tmp_path)
    timed = []
    real = flashinfer.autotune

    def recorded(tune_mode=True, cache=None, **kwargs):
        timed.append(tune_mode)
        return real(tune_mode, cache=cache, **kwargs)

    monkeypatch.setattr(flashinfer, "autotune", recorded)
    flashinfer_nvfp4.tune(mm_fp4, [(384, 256), (256, 512)], blackwell)
    assert timed == [True, True] and len(list(tmp_path.glob("flashinfer-fp4-*"))) == 2
    # a later process loads them untimed: its tiles are the first one's
    flashinfer_nvfp4.tune(mm_fp4, [(384, 256), (256, 512)], blackwell)
    assert timed == [True, True, False, False]


def test_only_graph_captures_of_up_to_its_rows_take_flashinfer(blackwell):
    from vllm_srun.accel import flashinfer_nvfp4

    module, _ = nvfp4_layer(256, 512, blackwell)
    calls = []

    def product(q, weight_t, *_, **__):
        calls.append(q.shape[0])
        return torch.zeros(
            q.shape[0], weight_t.shape[1], dtype=torch.bfloat16, device=q.device
        )

    module.flashinfer = product
    small = torch.randn(37, 512, device=blackwell).to(torch.bfloat16)
    big = torch.randn(flashinfer_nvfp4.GRAPH_ROWS + 1, 512, device=blackwell).to(
        torch.bfloat16
    )
    assert module(small).abs().max() > 0 and not calls  # eager forwards keep cuBLAS
    assert not captured(module, small).any() and calls == [37]
    assert captured(module, big).equal(module(big)) and calls == [37]


def test_shapes_flashinfer_multiplies_otherwise_keep_cublas(blackwell, monkeypatch):
    from vllm_srun.accel import flashinfer_nvfp4
    from vllm_srun.engines.native import quantized

    wrong, _ = nvfp4_layer(256, 512, blackwell)
    failing, _ = nvfp4_layer(128, 256, blackwell)
    right, _ = nvfp4_layer(384, 256, blackwell)
    monkeypatch.setattr(flashinfer_nvfp4, "load", lambda device: (None, "absent"))
    assert quantized.use_flashinfer([right], blackwell) == {"skipped": "absent"}
    assert right.flashinfer is None

    def product(q, weight_t, scale, *_, **__):
        columns = weight_t.shape[1]
        if columns == 128:
            raise RuntimeError("no tile for this shape")
        if columns == 256:  # far from cuBLAS's product
            return torch.zeros(q.shape[0], 256, dtype=torch.bfloat16, device=q.device)
        return right._product(q, scale.flatten())

    tuned = []
    monkeypatch.setattr(flashinfer_nvfp4, "load", lambda device: (product, None))
    monkeypatch.setattr(
        flashinfer_nvfp4, "tune", lambda mm, shapes, device: tuned.extend(shapes)
    )
    receipt = quantized.use_flashinfer([wrong, failing, right], blackwell)
    assert receipt["cublas_shapes"] == {
        "256x512": "relative error 1.0e+00 at 1 rows",
        "128x256": "RuntimeError: no tile for this shape",
    }
    assert wrong.flashinfer is None and failing.flashinfer is None
    assert right.flashinfer is product
    assert tuned == [(256, 512), (128, 256), (384, 256)]  # tuned before the check


def test_projections_of_one_input_scale_share_its_quantization(blackwell):
    from vllm_srun.accel import triton_nvfp4
    from vllm_srun.engines.native.quantized import project

    gate, _ = nvfp4_layer(256, 512, blackwell, seed=1)
    up, _ = nvfp4_layer(128, 512, blackwell, seed=2)
    other, _ = nvfp4_layer(64, 512, blackwell, seed=3)
    other.input_global = gate.input_global / 2  # another input global scale
    other.input_decode = 1.0 / other.input_global * torch.ones(1, device=blackwell)
    plain = nn.Linear(512, 32, bias=False).to(blackwell, torch.bfloat16)
    x = torch.randn(37, 512, device=blackwell).to(torch.bfloat16)
    alone = [layer(x) for layer in (gate, up, other, plain)]
    calls = []

    def counted(rows, global_scale):
        calls.append(global_scale)
        return triton_nvfp4.quantize(rows, global_scale)

    for layer in (gate, up, other):
        layer.quantize = counted
    shared = project(x, gate, up, other, plain)
    assert calls == [gate.input_global, other.input_global]
    assert all(got.equal(want) for got, want in zip(shared, alone, strict=True))


def sibling_layers(shapes, device):
    """NVFP4 layers of one input under one weight global scale, as a package's q / k / v or gate / up are."""
    from vllm_srun.accel import nvfp4
    from vllm_srun.engines.native.quantized import NVFP4Linear

    torch.manual_seed(0)
    weights = [torch.randn(rows, columns) * 0.05 for rows, columns in shapes]
    global_ = nvfp4.global_scale(torch.cat(weights))
    layers = []
    for weight in weights:
        layer = NVFP4Linear(nn.Linear(weight.shape[1], weight.shape[0], bias=False))
        packed, scale = nvfp4.quantize(weight, global_)
        layer.weight_packed.copy_(packed)
        layer.weight_scale.copy_(scale)
        layer.weight_global_scale.copy_(global_)
        layer.input_global_scale.fill_(448 * 6 / 5.0)
        layers.append(layer.to(device))
    return layers


def test_sibling_nvfp4_layers_merge_into_one_product_with_the_same_outputs(
    blackwell,
):
    from vllm_srun.engines.native import quantized

    q, k, v = sibling_layers([(512, 256), (128, 256), (128, 256)], blackwell)
    attention = nn.Module()
    attention.q_proj, attention.k_proj, attention.v_proj = q, k, v
    for layer in (q, k, v):
        layer.prepare()
    xs = [
        torch.randn(rows, 256, device=blackwell).to(torch.bfloat16)
        for rows in (1, 37, 300)
    ]
    before = [[layer(x) for layer in (q, k, v)] for x in xs]
    stored = torch.cuda.memory_allocated(blackwell)

    assert quantized.merge_siblings(attention) == 1
    group = q.merged
    assert group is not None and group.parts == (q, k, v)
    # the parts hold views of the merged weight: no second copy (only its few global-scale scalars)
    weights = sum(layer.stored_bytes() for layer in (q, k, v))
    assert torch.cuda.memory_allocated(blackwell) - stored < weights / 10
    base = group.layer.weight_packed.untyped_storage().data_ptr()
    assert all(
        layer.weight_packed.untyped_storage().data_ptr() == base for layer in (q, k, v)
    )
    for x, want in zip(xs, before, strict=True):
        merged = quantized.project(x, q, k, v)
        assert all(got.equal(w) for got, w in zip(merged, want, strict=True))
        assert all(layer(x).equal(w) for layer, w in zip((q, k, v), want, strict=True))


def test_siblings_of_other_global_scales_stay_apart(blackwell):
    from vllm_srun.engines.native import quantized

    gate, up = sibling_layers([(256, 256), (256, 256)], blackwell)
    up.weight_global_scale.mul_(2)
    mlp = nn.Module()
    mlp.gate_proj, mlp.up_proj = gate, up
    quantized.prepare(mlp)
    assert gate.merged is None and up.merged is None


def test_silu_mul_reads_column_slices_as_it_reads_contiguous_rows(device):
    pytest.importorskip("triton")
    from vllm_srun.accel import triton_gfx942

    both = torch.randn(3, 37, 2 * 1040, device=device).to(torch.bfloat16)
    gate, up = torch.split(both, 1040, dim=-1)
    want = triton_gfx942.silu_mul(gate.contiguous(), up.contiguous())
    assert triton_gfx942.silu_mul(gate, up).equal(want)


# d3's widths: hidden 5120; q (with its gate) 24 x 2 x 256, k / v 4 x 256; gate / up 17408
D3_QKV = [(12288, 5120), (1024, 5120), (1024, 5120)]
D3_GATE_UP = [(17408, 5120), (17408, 5120)]


@pytest.mark.parametrize("shapes", [D3_QKV, D3_GATE_UP], ids=["q/k/v", "gate/up"])
def test_d3_siblings_merge_bit_for_bit_at_the_package_widths(blackwell, shapes):
    from vllm_srun.engines.native import quantized

    parts = sibling_layers(shapes, blackwell)
    owner = nn.Module()
    names = quantized.SIBLINGS[0] if len(parts) == 3 else quantized.SIBLINGS[1]
    for name, layer in zip(names, parts, strict=True):
        setattr(owner, name, layer)
        layer.prepare()
    # 9,000 rows go in two slices of NVFP4_ROWS
    xs = [
        torch.randn(rows, 5120, device=blackwell).to(torch.bfloat16)
        for rows in (1, 300, 9000)
    ]
    before = [[layer(x) for layer in parts] for x in xs]
    assert quantized.merge_siblings(owner) == 1
    for x, want in zip(xs, before, strict=True):
        assert all(
            got.equal(w)
            for got, w in zip(quantized.project(x, *parts), want, strict=True)
        )
        reversed_ = quantized.project(x, *parts[::-1])  # any order: one product
        assert all(got.equal(w) for got, w in zip(reversed_, want[::-1], strict=True))
    del before


def test_merged_q_k_v_reach_attn_prep_as_views_with_no_higher_peak(blackwell):
    pytest.importorskip("triton")
    from vllm_srun.accel import triton_gfx942
    from vllm_srun.engines.native import quantized

    q, k, v = sibling_layers(D3_QKV, blackwell)
    attention = nn.Module()
    attention.q_proj, attention.k_proj, attention.v_proj = q, k, v
    for layer in (q, k, v):
        layer.prepare()
    torch.manual_seed(1)
    x = torch.randn(1, 2048, 5120, device=blackwell).to(torch.bfloat16)
    norm = [torch.randn(256, device=blackwell) for _ in range(2)]
    cos, sin = (
        torch.randn(1, 2048, 64, device=blackwell).to(torch.bfloat16) for _ in range(2)
    )

    def prep():
        qp, kp, vp = quantized.project(x, q, k, v)
        return (*triton_gfx942.attn_prep(qp, kp, *norm, cos, sin, 24, 4, 256, 1e-6), vp)

    def peak():
        prep()  # the first call compiles and sizes the cuBLAS workspace
        torch.cuda.synchronize(blackwell)
        torch.cuda.reset_peak_memory_stats(blackwell)
        start = torch.cuda.memory_allocated(blackwell)
        out = prep()
        torch.cuda.synchronize(blackwell)
        return out, torch.cuda.max_memory_allocated(blackwell) - start

    separate, separate_peak = peak()
    assert quantized.merge_siblings(attention) == 1
    merged, merged_peak = peak()
    assert all(got.equal(want) for got, want in zip(merged, separate, strict=True))
    # q and k are not copied out of the merged product
    assert merged_peak <= separate_peak


def test_a_merged_nvfp4_layer_refuses_a_move_after_the_merge(blackwell):
    from vllm_srun.engines.native import quantized

    gate, up = sibling_layers([(256, 256), (256, 256)], blackwell)
    mlp = nn.Module()
    mlp.gate_proj, mlp.up_proj = gate, up
    quantized.prepare(mlp)
    assert gate.merged is not None
    mlp.to(blackwell)  # where it already is: still views
    with pytest.raises(RuntimeError, match=r"before quantized\.prepare"):
        mlp.to("cpu")


def test_a_whole_model_cast_keeps_the_nvfp4_weights(device):
    from vllm_srun.engines.native.quantized import NVFP4Linear

    module = NVFP4Linear(nn.Linear(64, 32, bias=False))
    module.float().to(torch.bfloat16).to(device)
    assert {name: b.dtype for name, b in module.named_buffers()} == {
        "weight_packed": torch.uint8,
        "weight_scale": torch.float8_e4m3fn,
        "weight_global_scale": torch.float32,
        "input_global_scale": torch.float32,
    }


def test_cuda_offers_nvfp4_where_its_fp4_tensor_cores_run(device):
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.accel.gpu import nvfp4_unavailable

    if torch.version.hip:
        pytest.skip("CUDA accelerator")
    accelerator = CUDAAccelerator()
    info = accelerator.devices()[0]
    offered = accelerator.capabilities(info)["nvfp4"]
    assert offered is (nvfp4_unavailable(info.index) is None)
    if not offered:
        assert "Blackwell" in accelerator.lacks(info, "nvfp4")


def served(package, dtype):
    """Each fixture state answered twice (graphs capture on a shape's first uses) and the engine's receipt."""
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    from tests.test_quantized import QUESTIONS, STATES

    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(package), device="cuda:0", dtype=dtype),)
        )
    )
    runtime.start(background=False)
    try:
        bodies = [
            asyncio.run(runtime.call("decisions", {"state": s, "questions": QUESTIONS}))
            for _ in range(2)
            for s in STATES
        ]
        return bodies[len(STATES) :], runtime.lookup(None).model.engine_model.receipt()
    finally:
        runtime.stop()


def test_graphs_of_an_nvfp4_package_take_flashinfer_with_close_answers(
    mm_fp4, tmp_path, d3_text_package, monkeypatch
):
    from vllm_srun.accel import flashinfer_nvfp4

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "nvfp4", scheme="nvfp4"
    )
    product, captured_calls = flashinfer_nvfp4.product, []

    def counted(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            captured_calls.append(args[1].shape[0])
        return product(*args, **kwargs)

    monkeypatch.setattr(flashinfer_nvfp4, "product", counted)
    # the fixture's forwards run 612 to 8,192 rows
    monkeypatch.setattr(flashinfer_nvfp4, "GRAPH_ROWS", 8192)
    got, receipt = served(packed, "auto")
    assert receipt["nvfp4_product"] == {"graphs": "flashinfer-cutlass", "rows": 8192}
    assert captured_calls
    monkeypatch.setattr(flashinfer_nvfp4, "load", lambda device: (None, "absent"))
    want, receipt = served(packed, "auto")
    assert receipt["nvfp4_product"] == {"skipped": "absent"}
    # the same FP4 products summed in another order
    assert (numbers(got) - numbers(want)).abs().max() < 0.01
    # an explicit --dtype runs exact kernels only, which keep cuBLAS
    assert "nvfp4_product" not in served(packed, "float16")[1]


def test_an_nvfp4_package_answers_as_an_fp32_emulation_of_its_fp4_products(
    blackwell, tmp_path, d3_text_package, monkeypatch
):
    from vllm_srun.accel import nvfp4
    from vllm_srun.engines.native.quantized import NVFP4Linear

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "nvfp4", scheme="nvfp4"
    )
    plain = write_quantized_fixture(
        d3_text_package, tmp_path / "plain", scheme="nvfp4", dequantized=True
    )
    got, parameters, size = gpu_answers(packed)
    assert numbers(gpu_answers(packed)[0]).equal(numbers(got))
    _, plain_parameters, plain_size = gpu_answers(plain)
    assert parameters == plain_parameters and size < plain_size

    prepare = NVFP4Linear.prepare

    def keep_weight(self):
        self.reference = nvfp4.dequantize(
            self.weight_packed, self.weight_scale, self.weight_global_scale
        )
        prepare(self)
        self.quantize = lambda rows, global_scale: (rows, None)  # the BF16 rows

    def emulated(self, rows, _):
        # each layer quantizes the rows it gets itself, so shared slices are checked too
        q, scales = nvfp4.quantize(rows, self.input_global_scale)
        out = nvfp4.dequantize(q, scales, self.input_global_scale) @ self.reference.T
        return out.to(torch.bfloat16)

    monkeypatch.setattr(NVFP4Linear, "prepare", keep_weight)
    monkeypatch.setattr(NVFP4Linear, "_product", emulated)
    want = gpu_answers(packed)[0]
    # the same FP4 values multiplied in FP32: only the products' BF16 rounding separates them
    assert (numbers(got) - numbers(want)).abs().max() < 0.02
