"""Length buckets and per-bucket graphs of the native encoder path."""

import pytest
import torch
from vllm_srun.engines.native.encoder import (
    ROW_BUCKETS,
    WIDTH_BUCKETS,
    EncoderGraphs,
    bucket,
)

from .test_modernbert import CONFIGS, batch, native


def test_buckets_round_up_and_stop_at_the_largest():
    assert bucket(1, ROW_BUCKETS) == 1
    assert bucket(3, ROW_BUCKETS) == 4
    assert bucket(17, WIDTH_BUCKETS) == 32
    assert bucket(384, WIDTH_BUCKETS) == 384
    assert bucket(385, WIDTH_BUCKETS) is None


def test_large_batches_run_packed_without_padding():
    backbone, _ = native(CONFIGS["yarn"])
    graphs = EncoderGraphs(backbone, torch.device("cpu"), max_tokens=64)
    assert graphs.shape([10, 3]) == (2, 16)
    assert graphs.shape([40, 3]) is None
    assert graphs.shape([3] * 65) is None


def test_buckets_that_pad_much_past_the_launch_bound_run_packed():
    backbone, _ = native(CONFIGS["yarn"])
    graphs = EncoderGraphs(backbone, torch.device("cpu"))
    assert graphs.shape([16] * 64) == (64, 16)
    assert graphs.shape([256] + [16] * 15) is None
    assert graphs.shape([250] * 8) == (8, 256)
    assert graphs.shape([200] * 8) is None


@pytest.mark.parametrize("lengths", [[9], [12, 3, 7], [21, 5, 1, 16]])
def test_bucketed_rows_match_the_packed_layout(lengths):
    backbone, _ = native(CONFIGS["yarn"], seed=8)
    graphs = EncoderGraphs(backbone, torch.device("cpu"), capture_after=1 << 30)
    ids, mask = batch(lengths, seed=5)
    flat = ids[mask.bool()]
    with torch.inference_mode():
        bucketed = graphs(flat, lengths, (2, 4), False)
        packed = backbone.encode(flat, backbone.packed(lengths, "cpu"), (2, 4))
        again = graphs(flat, lengths, (2, 4), False)
    for layer in (2, 4):
        assert bucketed[layer].shape == packed[layer].shape
        torch.testing.assert_close(bucketed[layer], packed[layer], rtol=0, atol=1e-5)
        assert torch.equal(bucketed[layer], again[layer])
    assert graphs.receipt()["eager"] == 2 and graphs.receipt()["captures"] == 0


@pytest.mark.gpu
def test_graph_replays_are_the_eager_bucket_bit_for_bit():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm device")
    device = torch.device("cuda")
    backbone, _ = native(CONFIGS["yarn"], seed=9)
    backbone = backbone.to(device)
    graphs = EncoderGraphs(backbone, device)
    lengths = [13, 4]
    ids, mask = batch(lengths, seed=6)
    flat = ids[mask.bool()]
    with torch.inference_mode():
        outputs = [graphs(flat, lengths, (4,), False)[4] for _ in range(4)]
        wide = graphs(torch.randint(3, 300, (400,)), [400], (4,), False)[4]
    assert graphs.receipt()["captures"] == 1 and graphs.receipt()["replays"] == 3
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])
    assert wide.shape == (400, CONFIGS["yarn"]["hidden_size"])
    assert graphs.receipt()["packed"] == 1
