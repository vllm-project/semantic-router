"""CPU admission checks; actual graph capture and replay are covered on GPU."""

import pytest
import torch
from vllm_srun.engines.native.fast import Graphs, Masks


@pytest.mark.parametrize("tokens,ordinary_after", [(16, 1), (33, 2)])
@pytest.mark.parametrize("pressure", [None, "graphs", "bytes"])
def test_pressure_overrides_first_use_capture(
    monkeypatch, tokens, ordinary_after, pressure
):
    graphs = Graphs(
        torch.nn.Identity(),
        Masks(),
        capture_after=2,
        hot_capture_after=4,
        first_capture_tokens=32,
        max_graphs=4,
        max_bytes=100,
    )
    if pressure == "graphs":
        graphs.graphs.update({key: {} for key in range(3)})
    elif pressure == "bytes":
        graphs.budget = 100
        graphs.spent = 75
    attempts = []

    def capture(key, inputs, body):
        attempts.append(key)

    monkeypatch.setattr(graphs, "_capture", capture)
    needed = 4 if pressure else ordinary_after
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for observed in range(1, needed + 1):
            got = graphs.run("late", inputs, lambda x: x + 1, tokens)
            assert torch.equal(got, inputs[0] + 1)
            assert attempts == (["late"] if observed == needed else [])
    stats = graphs.receipt()
    reason = "pressure" if pressure else "warmup"
    assert stats["eager_reasons"][reason] == needed - 1
    assert stats["lookups"] == stats["misses"] == stats["eager"] == needed
    assert sum(stats["eager_reasons"].values()) == stats["eager"]
