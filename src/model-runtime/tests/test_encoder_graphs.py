"""Length buckets and per-bucket graphs of the native encoder path."""

import pytest
import torch
from vllm_srun.engines.native.encoder import (
    ROW_BUCKETS,
    WIDTH_BUCKETS,
    EncoderGraphs,
    bucket,
    graph_module,
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


def test_graph_module_resolves_by_device_type():
    assert graph_module("cpu") is None
    if graph_module("npu") is not None:
        assert graph_module("npu") is torch.npu
    if torch.cuda.is_available():
        assert graph_module(torch.device("cuda")) is torch.cuda


def test_bucketed_padded_rows_match_the_padded_layout():
    backbone, _ = native(CONFIGS["yarn"], seed=8)
    graphs = EncoderGraphs(backbone, torch.device("cpu"), max_tokens=64)
    bucketed = None
    for lengths in ([9], [12, 3, 7]):
        ids, mask = batch(lengths, seed=5)
        with torch.inference_mode():
            bucketed = graphs.padded(ids, mask, (2, 4), False)
            engine = backbone.encode(
                ids, backbone.padded(mask, len(lengths), max(lengths), "cpu"), (2, 4)
            )
        for layer in (2, 4):
            assert bucketed[layer].shape == engine[layer].shape
            torch.testing.assert_close(
                bucketed[layer], engine[layer], rtol=0, atol=1e-5
            )
    with torch.inference_mode():
        again = graphs.padded(ids, mask, (2, 4), False)
    assert torch.equal(bucketed[4], again[4])
    assert graphs.receipt()["padded_eager"] == 3 and graphs.receipt()["captures"] == 0


def test_padded_grids_above_the_buckets_run_the_engine_path():
    backbone, _ = native(CONFIGS["yarn"])
    graphs = EncoderGraphs(backbone, torch.device("cpu"))
    ids, mask = batch([16] * 65, seed=2)
    assert graphs.padded(ids, mask, (2,), False) is None
    assert graphs.receipt()["padded_skip"] == 1


@pytest.mark.gpu
def test_padded_graph_replays_are_the_eager_bucket_bit_for_bit():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm device")
    device = torch.device("cuda")
    backbone, _ = native(CONFIGS["yarn"], seed=9)
    backbone = backbone.to(device)
    graphs = EncoderGraphs(backbone, device)
    ids, mask = batch([13, 4], seed=6)
    with torch.inference_mode():
        outputs = [graphs.padded(ids, mask, (4,), False)[4] for _ in range(4)]
        engine = backbone.encode(
            ids.to(device), backbone.padded(mask, 2, 13, device), (4,)
        )
        big = graphs.padded(ids.repeat(65, 1), mask.repeat(65, 1), (4,), False)
    assert graphs.receipt()["captures"] == 1 and graphs.receipt()["replays"] == 3
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])
    torch.testing.assert_close(outputs[0], engine[4], rtol=0, atol=1e-5)
    assert big is None
    assert graphs.receipt()["padded_skip"] == 1
