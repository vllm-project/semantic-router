"""CPU admission checks; actual graph capture and replay are covered on GPU."""

from types import SimpleNamespace

import pytest
import torch
from vllm_srun.engines.native import fast


def cache(**options):
    return fast.Graphs(torch.nn.Identity(), fast.Masks(), **options)


def capture_on_cpu(graphs, monkeypatch):
    """Replace only device capture, leaving admission and replay accounting intact."""
    attempts = []

    def capture(key, inputs, body):
        attempts.append(key)
        static = tuple(value.clone() for value in inputs)
        output = body(*static)

        def replay():
            output.copy_(body(*static))

        entry = {
            "inputs": static,
            "graph": SimpleNamespace(replay=replay),
            "output": output,
        }
        graphs.graphs[key] = entry
        graphs.stats["captures"] += 1
        return entry

    monkeypatch.setattr(graphs, "_capture", capture)
    return attempts


@pytest.mark.parametrize("tokens,ordinary_after", [(32, 1), (33, 2)])
@pytest.mark.parametrize("pressure", [None, "graphs", "bytes"])
def test_pressure_preserves_first_use_capture(
    monkeypatch, tokens, ordinary_after, pressure
):
    graphs = cache(
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
    attempts = capture_on_cpu(graphs, monkeypatch)
    needed = 1 if ordinary_after == 1 else 4 if pressure else ordinary_after
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for observed in range(1, needed + 1):
            got = graphs.run("late", inputs, lambda x: x + 1, tokens)
            assert torch.equal(got, inputs[0] + 1)
            assert attempts == (["late"] if observed == needed else [])
    stats = graphs.receipt()
    reason = "pressure" if pressure else "warmup"
    assert stats["eager_reasons"][reason] == needed - 1
    assert stats["lookups"] == stats["misses"] == needed
    assert stats["eager"] == needed - 1
    assert stats["captures"] == stats["replays"] == 1
    assert stats["candidates"] == 0
    assert sum(stats["eager_reasons"].values()) == stats["eager"]


def test_pressure_never_lowers_the_ordinary_threshold(monkeypatch):
    graphs = cache(capture_after=4, hot_capture_after=2, max_graphs=4)
    graphs.graphs.update({key: {} for key in range(3)})
    attempts = capture_on_cpu(graphs, monkeypatch)
    with torch.inference_mode():
        for _ in range(3):
            graphs.run("late", (torch.tensor([3.0]),), lambda x: x + 1, 16)
        assert not attempts
        graphs.run("late", (torch.tensor([4.0]),), lambda x: x + 1, 16)
    assert attempts == ["late"]


@pytest.mark.parametrize("limit", [{"max_graphs": 0}, {"max_bytes": 0}])
def test_full_shapes_do_not_remain_candidates(limit):
    graphs = cache(capture_after=2, hot_capture_after=2, **limit)
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for observed in range(1, 5):
            assert torch.equal(
                graphs.run("late", inputs, lambda x: x + 1, 16), inputs[0] + 1
            )
            if observed % 2 == 0:
                assert "late" not in graphs.seen
    stats = graphs.receipt()
    assert stats["candidates"] == 0
    assert stats["full"] == stats["eager_reasons"]["full"] == 2
    assert stats["eager_reasons"]["warmup"] == 2
    assert stats["eager"] == stats["lookups"] == stats["misses"] == 4


def test_failed_capture_drops_the_candidate_and_is_not_retried(monkeypatch):
    graphs = cache()
    attempts = []

    def fail(key, inputs, body):
        attempts.append(key)
        graphs.failed.add(key)
        graphs.stats["failed"] += 1

    monkeypatch.setattr(graphs, "_capture", fail)
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for _ in range(3):
            assert torch.equal(
                graphs.run("late", inputs, lambda x: x + 1, 16), inputs[0] + 1
            )
    stats = graphs.receipt()
    assert attempts == ["late"]
    assert stats["candidates"] == 0 and stats["failed"] == 1
    assert stats["eager_reasons"]["warmup"] == 1
    assert stats["eager_reasons"]["capture_failed"] == 2
    assert sum(stats["eager_reasons"].values()) == stats["eager"] == 3


def test_cold_shape_churn_keeps_recent_hot_evidence(monkeypatch):
    monkeypatch.setattr(fast, "MAX_GRAPH_CANDIDATES", 4)
    graphs = cache(max_graphs=4, hot_capture_after=4)
    graphs.graphs.update({key: {} for key in range(3)})
    attempts = capture_on_cpu(graphs, monkeypatch)
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for sighting in range(3):
            graphs.run("hot", inputs, lambda x: x + 1, 16)
            for cold in range(3):
                graphs.run((sighting, cold), inputs, lambda x: x + 1, 16)
                assert len(graphs.seen) <= fast.MAX_GRAPH_CANDIDATES
            assert graphs.seen["hot"] == sighting + 1
        assert not attempts
        graphs.run("hot", inputs, lambda x: x + 1, 16)
    assert attempts == ["hot"]
    assert graphs.receipt()["cached"] == 4
    assert "hot" not in graphs.seen
    assert len(graphs.seen) == 3


@pytest.mark.parametrize(
    "max_graphs,cached,pressure",
    [
        (4, 0, False),
        (4, 1, False),
        (4, 2, False),
        (4, 3, True),
        (512, 383, False),
        (512, 384, True),
    ],
)
def test_uninitialized_byte_budget_uses_only_graph_count(max_graphs, cached, pressure):
    graphs = cache(max_graphs=max_graphs)
    graphs.graphs.update({key: {} for key in range(cached)})
    graphs.spent = 1 << 60
    assert graphs.budget is None
    assert graphs._under_pressure() is pressure


def test_default_pressure_requires_eight_sightings(monkeypatch):
    graphs = cache()
    graphs.graphs.update({key: {} for key in range(384)})
    attempts = capture_on_cpu(graphs, monkeypatch)
    inputs = (torch.tensor([3.0]),)
    with torch.inference_mode():
        for _ in range(7):
            graphs.run("late", inputs, lambda x: x + 1, 16)
        assert not attempts
        graphs.run("late", inputs, lambda x: x + 1, 16)
    stats = graphs.receipt()
    assert attempts == ["late"]
    assert stats["cached"] == 385 and stats["candidates"] == 0
    assert stats["lookups"] == stats["misses"] == 8
    assert stats["eager_reasons"]["pressure"] == 7
    assert stats["captures"] == stats["replays"] == 1


@pytest.mark.parametrize("max_graphs", [1, 2, 3])
def test_tiny_caches_have_no_graph_count_reserve(max_graphs):
    graphs = cache(max_graphs=max_graphs)
    for key in range(max_graphs):
        assert not graphs._under_pressure()
        graphs.graphs[key] = {}
    assert not graphs._under_pressure()
    graphs.budget = 100
    graphs.spent = 75
    assert graphs._under_pressure()


@pytest.mark.parametrize("spent,pressure", [(74, False), (75, True), (100, True)])
def test_byte_pressure_boundary(spent, pressure):
    graphs = cache()
    graphs.budget = 100
    graphs.spent = spent
    assert graphs._under_pressure() is pressure


@pytest.mark.parametrize("reason", ["too_large", "not_inference", "capture_failed"])
def test_ineligible_shapes_do_not_become_candidates(reason):
    graphs = cache(max_tokens=32)
    inputs = (torch.tensor([3.0]),)
    if reason == "capture_failed":
        graphs.failed.add("late")
    with torch.inference_mode(reason != "not_inference"):
        got = graphs.run(
            "late", inputs, lambda x: x + 1, 33 if reason == "too_large" else 16
        )
    assert torch.equal(got, inputs[0] + 1)
    stats = graphs.receipt()
    assert stats["lookups"] == stats["misses"] == stats["eager"] == 1
    assert stats["candidates"] == 0
    assert stats["eager_reasons"][reason] == sum(stats["eager_reasons"].values()) == 1


def test_receipt_has_an_explicit_schema_and_copies_eager_reasons():
    graphs = cache()
    graphs.stats["internal_counter"] = 7
    receipt = graphs.receipt()
    assert set(receipt) == {
        "lookups",
        "misses",
        "captures",
        "replays",
        "eager",
        "full",
        "failed",
        "cached",
        "candidates",
        "spent_bytes",
        "budget_bytes",
        "eager_reasons",
    }
    receipt["eager_reasons"]["full"] = 99
    assert graphs.receipt()["full"] == graphs.eager_reasons["full"] == 0
