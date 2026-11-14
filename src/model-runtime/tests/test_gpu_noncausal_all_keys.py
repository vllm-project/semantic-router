"""GPU: unpadded noncausal forwards over ``MAX_GRAPH_TOKENS`` attend to every key without a mask, near the masked forward."""

import pytest
import torch
from vllm_srun.engines.native import fast
from vllm_srun.testing.decision3 import write_quantized_fixture

from tests.test_gpu_fast_path import qwen3_5

pytestmark = pytest.mark.gpu


@pytest.fixture
def accelerator():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if torch.version.hip:
        from vllm_srun.accel.rocm import ROCmAccelerator

        return ROCmAccelerator()
    from vllm_srun.accel.cuda import CUDAAccelerator

    return CUDAAccelerator()


def backbone(accelerator):
    from tests.test_gpu_fast_path import build

    config = {**qwen3_5(4, 1024, 16, 8), "attention_mode": "noncausal_full_attention"}
    return build(config, 7, False, accelerator, accelerator.devices()[0])


def forward(module, lengths, width, masks):
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 1024, (len(lengths), width), generator=generator)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        mask[row, width - length :] = 1
    ids, mask = ids.to("cuda"), mask.to("cuda")
    padded = any(n != width for n in lengths)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return module(ids, mask, masks=masks.build(mask, padded)).float()


def test_a_long_unpadded_row_drops_the_mask_near_the_masked_forward(accelerator):
    module = backbone(accelerator)
    unmasked = fast.NoncausalMasks(fast.MAX_GRAPH_TOKENS)
    got = forward(module, [6000], 6000, unmasked)
    want = forward(module, [6000], 6000, fast.NoncausalMasks())
    assert not torch.equal(got, want)
    cosine = torch.nn.functional.cosine_similarity(got, want, dim=-1)
    assert cosine.min() > 0.999, cosine.min()


@pytest.mark.parametrize(
    ("lengths", "width"), [([6000, 5990], 6000), ([4096], 4096), ([2000, 2000], 2000)]
)
def test_padded_and_graph_sized_batches_keep_the_mask(accelerator, lengths, width):
    module = backbone(accelerator)
    unmasked = fast.NoncausalMasks(fast.MAX_GRAPH_TOKENS)
    got = forward(module, lengths, width, unmasked)
    want = forward(module, lengths, width, fast.NoncausalMasks())
    assert torch.equal(got, want)


@pytest.mark.parametrize(
    ("quantized", "dtype", "threshold"),
    [
        (True, None, fast.MAX_GRAPH_TOKENS),
        (True, "float32", None),
        (False, None, None),
    ],
)
def test_only_approximate_kernels_drop_the_mask(
    accelerator, tmp_path, d3_text_package, quantized, dtype, threshold
):
    """A quantized package at ``--dtype auto``; another dtype (exact kernels only) and BF16 keep it."""
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    package = (
        write_quantized_fixture(d3_text_package, tmp_path / "int4")
        if quantized
        else d3_text_package
    )
    name = "rocm:0" if torch.version.hip else "cuda:0"
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=name, dtype=dtype),))
    )
    runtime.start(background=False)
    try:
        engine = runtime.lookup(None).model.engine_model
        assert engine.unmasked_above == threshold
        if engine.masks is not None:
            assert engine.masks.unmasked_above == threshold
    finally:
        runtime.stop()
