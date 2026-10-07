"""The embed A/B tool (``tools/embed_legacy.py``): its confidence intervals and the runtime side's load."""

import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "embed_legacy.py"


@pytest.fixture(scope="module")
def el():
    spec = importlib.util.spec_from_file_location("embed_legacy", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_intervals_bound_runtime_minus_legacy(el):
    legacy = [int((10 + i % 7) * 1e6) for i in range(60)]
    faster = {
        "legacy": legacy,
        "runtime": [ns - 2_000_000 for ns in legacy],
        "throughput": {"legacy": [100.0, 102.0, 98.0], "runtime": [120.0, 118, 121]},
    }
    bounds = el.intervals(faster, replicates=200)
    assert bounds["p50_ms"] == [-2.0, -2.0] and bounds["p95_ms"] == [-2.0, -2.0]
    assert 0 < bounds["per_s"][0] <= bounds["per_s"][1]


def test_the_baseline_child_serves_its_engine_from_its_own_tree(el):
    args = SimpleNamespace(
        baseline_engine="onnxruntime",
        cache="/cache",
        prepared="/bundles",
        jobs="omni_mini_audio",
        device="cpu",
        profile="exact",
        threads=16,
        baseline_runtime="/staging/src/model-runtime",
    )
    command = el.baseline_command(args)
    assert command[2] == "serve"
    flags = dict(zip(command[3::2], command[4::2], strict=True))
    assert flags["--omni-engine"] == "onnxruntime"
    assert flags["--runtime-src"] == "/staging/src/model-runtime"
    assert flags["--threads"] == "16"
    args.baseline_runtime = None
    assert "--runtime-src" not in el.baseline_command(args)


def test_intervals_straddle_zero_when_the_sides_match(el):
    calls = [int((10 + i % 7) * 1e6) for i in range(60)]
    same = el.intervals({"legacy": calls, "runtime": calls[1:] + calls[:1]})
    assert same["p50_ms"][0] <= 0 <= same["p50_ms"][1]
    assert "per_s" not in same


def test_the_runtime_load_is_concurrent_requests_on_the_servers_loop(el):
    class Runtime:
        def __init__(self):
            self.active = self.peak = 0
            self.threads, self.sizes = set(), []

        async def call(self, surface, body, size):
            self.threads.add(threading.get_ident())
            self.sizes.append((surface, body["input"], size))
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.002)
            self.active -= 1
            return 200, {}

    runtime, loop = Runtime(), el.LoopThread()
    requests = [
        el.Request("embeddings", {"input": "a"}, 11),
        el.Request("embeddings", {"input": "bb"}, 12),
    ]
    try:
        window = el.runtime_load(
            runtime, loop, "job", requests, SimpleNamespace(concurrency=3, seconds=0.1)
        )
    finally:
        loop.close()
    assert window["job"] == "job" and window["concurrency"] == 3
    assert window["calls"] == len(window["latency_ns"]) == len(runtime.sizes) > 3
    assert runtime.peak == 3 and runtime.threads == {loop.thread.ident}
    assert set(runtime.sizes) == {("embeddings", "a", 11), ("embeddings", "bb", 12)}
