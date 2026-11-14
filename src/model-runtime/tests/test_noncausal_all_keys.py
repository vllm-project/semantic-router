"""Unpadded noncausal forwards longer than a graph holds attend to every key without a mask (``ALL_KEYS``)."""

import pytest
import torch
from vllm_srun.accel.kernels import reference_kernels
from vllm_srun.engines.native import fast
from vllm_srun.engines.native.models.common import ALL_KEYS, attention
from vllm_srun.engines.native.models.qwen3_5 import noncausal_masks
from vllm_srun.testing.decision3 import write_quantized_fixture

from tests.conftest import start_runtime


def qkv(length, heads=4, kv_heads=2, dim=16):
    generator = torch.Generator().manual_seed(0)
    shape = (1, heads, length, dim)
    query = torch.randn(shape, generator=generator)
    key, value = (
        torch.randn((1, kv_heads, length, dim), generator=generator) for _ in range(2)
    )
    return query, key, value


def test_all_keys_attends_to_every_key_and_not_causally():
    query, key, value = qkv(12)
    kernels = reference_kernels("cpu")
    every = torch.ones(1, 1, 1, 12, dtype=torch.bool)
    got = attention(kernels, query, key, value, ALL_KEYS, groups=2, scaling=0.25)
    masked = attention(kernels, query, key, value, every, groups=2, scaling=0.25)
    causal = attention(kernels, query, key, value, None, groups=2, scaling=0.25)
    torch.testing.assert_close(got, masked)
    assert not torch.allclose(got, causal)


@pytest.mark.parametrize(
    ("rows", "length", "padded", "unmasked_above", "unmasked"),
    [
        (1, 10, False, 8, True),
        (2, 5, False, 8, True),
        (1, 8, False, 8, False),
        (1, 10, True, 8, False),
        (1, 10, False, None, False),
    ],
)
def test_only_unpadded_rows_over_the_threshold_drop_the_mask(
    rows, length, padded, unmasked_above, unmasked
):
    attention_mask = torch.ones(rows, length, dtype=torch.long)
    if padded:
        attention_mask[0, 0] = 0
    full, linear = noncausal_masks(attention_mask, padded, unmasked_above)
    masks = fast.NoncausalMasks(unmasked_above).build(attention_mask, padded)
    assert masks["full"] is full if unmasked else torch.equal(masks["full"], full)
    if unmasked:
        assert full is ALL_KEYS and linear is None
    else:
        assert torch.equal(full, attention_mask.bool()[:, None, None, :])


def test_the_cpu_keeps_the_mask_for_a_quantized_package(tmp_path, d3_text_package):
    package = write_quantized_fixture(d3_text_package, tmp_path / "int4")
    runtime = start_runtime(package)
    try:
        engine = runtime.lookup(None).model.engine_model
        assert engine.kernels.allow_approximate
        assert engine.unmasked_above is None
    finally:
        runtime.stop()
