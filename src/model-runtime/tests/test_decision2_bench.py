"""Decision 2.0 bench CLI: simultaneous paired vs alternating loads."""

from __future__ import annotations

import subprocess
import sys
import weakref
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decision2_bench as bench

TOOL = Path(__file__).resolve().parents[1] / "tools" / "decision2_bench.py"


def test_decision2_bench_requires_a_mode():
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paired" in completed.stdout
    assert "sequential" in completed.stdout
    assert (
        "simultaneous" in completed.stdout.lower() or "alternating" in completed.stdout
    )
    extra = subprocess.run(
        [sys.executable, str(TOOL), "paired", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--rounds" in extra.stdout
    assert "--extend" in extra.stdout
    assert "--warmup-passes" in extra.stdout


def test_kernel_choices_reads_an_autotune_cache(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "kernel_choices.py"
    spec = importlib.util.spec_from_file_location("kernel_choices_tool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    folder = tmp_path / "cache" / "hash"
    folder.mkdir(parents=True)
    (folder / "chunk_fwd_kernel_o.autotune.json").write_text(
        """{"key": [16, 16], "configs_timings": [
          [{"kwargs": {"BK": 32}, "num_warps": 2, "num_ctas": 1, "num_stages": 3,
            "maxnreg": null, "ir_override": null}, 0.02],
          [{"kwargs": {"BK": 64}, "num_warps": 4, "num_ctas": 1, "num_stages": 3,
            "maxnreg": null, "ir_override": null}, 0.04]
        ]}"""
    )
    kernels = module.choices(tmp_path / "cache")
    assert list(kernels) == ["chunk_fwd_kernel_o"]
    assert kernels["chunk_fwd_kernel_o"][0]["config"]["kwargs"] == {"BK": 32}


BODY = {"id": "p0", "state": [], "questions": {"q": {"type": "choice"}}}
ANSWERS = {"q": {"choice": "a"}}


def test_close_releases_the_loaded_model():
    class Model:
        pass

    model = Model()
    ref = weakref.ref(model)
    side = bench.BenchmarkSide(model, "cpu", native=True)
    del model
    side.close()
    assert ref() is None


def test_native_graph_stats_and_deltas_include_lookups_and_misses():
    counters = {
        "lookups": 20,
        "misses": 9,
        "captures": 3,
        "replays": 14,
        "eager": 6,
        "cached": 3,
        "candidates": 1,
        "spent_bytes": 75,
        "budget_bytes": 100,
        "eager_reasons": {"warmup": 4, "pressure": 2},
    }
    model = Namespace(engine_model=Namespace(receipt=lambda: {"graphs": counters}))
    side = bench.BenchmarkSide(model, "cpu", native=True)
    before = side.graph_stats()
    assert before == counters
    counters.update(lookups=30, misses=13, captures=4, replays=21, eager=9)
    assert bench.graph_delta(before, side.graph_stats()) == {
        "lookups": 10,
        "misses": 4,
        "captures": 1,
        "replays": 7,
        "eager": 3,
    }


def test_encoder_receipts_without_lookup_counters_remain_compatible():
    model = Namespace(
        engine_model=Namespace(
            receipt=lambda: {
                "encoder_graphs": {"captures": 1, "replays": 2, "eager": 3}
            }
        )
    )
    side = bench.BenchmarkSide(model, "cpu", native=True)
    assert side.graph_stats() == {
        "lookups": 0,
        "misses": 0,
        "captures": 1,
        "replays": 2,
        "eager": 3,
    }
    assert bench.BenchmarkSide(model, "cpu").graph_stats() == {}


def test_sequential_unloads_each_side_before_loading_the_next(monkeypatch):
    events = []
    active = []

    class Side:
        def __init__(self, name):
            assert not active
            active.append(name)
            events.append(("load", name))

        def __call__(self, body):
            return ANSWERS

        def graph_stats(self):
            return {}

        def close(self):
            events.append(("close", active.pop()))

    monkeypatch.setattr(bench, "bundled", lambda args: Side("released"))
    monkeypatch.setattr(bench, "native", lambda args: Side("native"))
    monkeypatch.setattr(bench, "pin_status", lambda *args: (True, None))
    monkeypatch.setattr(bench, "requests", lambda args: [BODY])
    args = Namespace(repo="test", device="cpu", warmup_passes=2, rounds=2, extend=2)
    result = bench.run_sequential(args)
    assert [row["order"] for row in result["rounds"]] == [
        ["released", "native"],
        ["native", "released"],
    ]
    assert events == [
        (action, name)
        for name in ["released", "native", "native", "released"]
        for action in ["load", "close"]
    ]
    assert not active


@pytest.mark.parametrize("failure", ["answer", "capture"])
def test_failed_measurement_still_releases_model(monkeypatch, failure):
    closed = []

    class Side:
        calls = 0

        def __call__(self, body):
            self.calls += 1
            return (
                {"q": {"error": "invalid_model_output"}}
                if failure == "answer"
                else ANSWERS
            )

        def graph_stats(self):
            return {
                "captures": self.calls if failure == "capture" else 0,
                "cached": 3,
                "spent_bytes": 75,
                "budget_bytes": 100,
                "eager_reasons": {"pressure": self.calls},
            }

        def close(self):
            closed.append(True)

    monkeypatch.setattr(bench, "native", lambda args: Side())
    with pytest.raises(SystemExit, match=r"error|captures") as error:
        bench._load_and_time("native", Namespace(warmup_passes=0), [BODY])
    if failure == "capture":
        message = str(error.value)
        assert "--warmup-passes" in message
        assert '"spent_bytes": 75' in message and '"budget_bytes": 100' in message
        assert '"pressure": 1' in message
    assert closed == [True]


def test_paired_load_failure_closes_released_model(monkeypatch):
    closed = []
    monkeypatch.setattr(bench, "pin_status", lambda *args: (True, None))
    monkeypatch.setattr(bench, "requests", lambda args: [BODY])
    monkeypatch.setattr(
        bench, "bundled", lambda args: Namespace(close=lambda: closed.append(True))
    )

    def fail(args):
        raise RuntimeError("native load failed")

    monkeypatch.setattr(bench, "native", fail)
    with pytest.raises(RuntimeError, match="native load failed"):
        bench.run_paired(Namespace(repo="test", device="cpu"))
    assert closed == [True]
