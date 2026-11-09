"""The embed A/B tool (``tools/embed_legacy.py``): its confidence intervals and the runtime side's load."""

import asyncio
import importlib.util
import json
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
        max_num_seqs=1,
        threads=16,
        baseline_runtime="/staging/src/model-runtime",
    )
    command = el.baseline_command(args)
    assert command[2] == "serve"
    flags = dict(zip(command[3::2], command[4::2], strict=True))
    assert flags["--omni-engine"] == "onnxruntime"
    assert flags["--runtime-src"] == "/staging/src/model-runtime"
    assert flags["--threads"] == "16" and flags["--max-num-seqs"] == "1"
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


def test_embed_legacy_ab_records_failed_pairs(el, monkeypatch, tmp_path):
    """The embed run_ab() pair loop must count failed pairs (total/failed)
    instead of silently discarding them."""
    specs = [
        {
            "Job": "embedding",
            "Repo": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
            "Path": "/cache/x",
            "Revision": "r",
            "Mode": "embedding",
            "Adapter": "mmbert",
            "Provider": "candle",
            "Modality": "text",
            "MaxTokens": 8192,
            "Overflow": "truncate",
            "Dimension": 0,
            "Layer": 0,
            "Inputs": [{"id": f"t{i}", "text": f"t{i}"} for i in range(3)],
            "Repeats": 1,
            "Concurrency": 0,
            "Seconds": 0.0,
        }
    ]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    class FakeProc:
        def __init__(self):
            self.stdin = SimpleNamespace()
            self.stdin.write = lambda *a, **k: None
            self.stdin.close = lambda: None
            self._lines = ["READY", "-1", "-1", "-1"]
            self._i = 0

        @property
        def stdout(self):
            return self

        def readline(self):
            line = self._lines[min(self._i, len(self._lines) - 1)]
            self._i += 1
            return line + "\n"

        def poll(self):
            return None

        def wait(self):
            return None

    class FakeRuntime:
        def stop(self):
            pass

    class FakeLoop:
        def call(self, runtime, request):
            # Non-200 status: the request failed.
            return 500, 500, {}

        def close(self):
            pass

    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(
        el,
        "serve_runtime",
        lambda args, specs: (
            FakeRuntime(),
            {"vllm-sr/Vela-1.0-Encoder-307M-Embedding": "embedding"},
        ),
    )
    monkeypatch.setattr(el, "LoopThread", FakeLoop)
    monkeypatch.setattr(
        el,
        "runtime_requests",
        lambda spec, model: [
            el.Request("embeddings", {"input": "x"}, 1) for _ in spec["Inputs"]
        ],
    )

    args = SimpleNamespace(
        binary=str(binary),
        tree="/tree",
        libs=None,
        cache="/cache",
        device="cpu",
        threads=None,
        max_num_seqs=None,
        rounds=1,
        concurrency=0,
        seconds=0.0,
        out=str(out),
        jobs="",
        legacy_cpus=None,
        gap_ms=0.0,
        baseline_engine=None,
        baseline_env=[],
        baseline_runtime=None,
        omni_engine="native",
        runtime_src=None,
        prepared="/prepared",
    )
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 3
    assert row["successful_pairs"] == 0
    assert row["failed_pairs"] == 3
    assert row["success_rate"] == 0.0
    assert row["pairs"] == 0
    assert row["median_speedup"] is None


def test_embed_legacy_ab_records_successful_pairs(el, monkeypatch, tmp_path):
    specs = [
        {
            "Job": "embedding",
            "Repo": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
            "Path": "/cache/x",
            "Revision": "r",
            "Mode": "embedding",
            "Adapter": "mmbert",
            "Provider": "candle",
            "Modality": "text",
            "MaxTokens": 8192,
            "Overflow": "truncate",
            "Dimension": 0,
            "Layer": 0,
            "Inputs": [{"id": f"t{i}", "text": f"t{i}"} for i in range(2)],
            "Repeats": 1,
            "Concurrency": 0,
            "Seconds": 0.0,
        }
    ]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    class FakeProc:
        def __init__(self):
            self.stdin = SimpleNamespace()
            self.stdin.write = lambda *a, **k: None
            self.stdin.close = lambda: None
            self._lines = ["READY", "2000", "2000"]
            self._i = 0

        @property
        def stdout(self):
            return self

        def readline(self):
            line = self._lines[min(self._i, len(self._lines) - 1)]
            self._i += 1
            return line + "\n"

        def poll(self):
            return None

        def wait(self):
            return None

    class FakeRuntime:
        def stop(self):
            pass

    class FakeLoop:
        def call(self, runtime, request):
            return 1000, 200, {}

        def close(self):
            pass

    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(
        el,
        "serve_runtime",
        lambda args, specs: (
            FakeRuntime(),
            {"vllm-sr/Vela-1.0-Encoder-307M-Embedding": "embedding"},
        ),
    )
    monkeypatch.setattr(el, "LoopThread", FakeLoop)
    monkeypatch.setattr(
        el,
        "runtime_requests",
        lambda spec, model: [
            el.Request("embeddings", {"input": "x"}, 1) for _ in spec["Inputs"]
        ],
    )

    args = SimpleNamespace(
        binary=str(binary),
        tree="/tree",
        libs=None,
        cache="/cache",
        device="cpu",
        threads=None,
        max_num_seqs=None,
        rounds=1,
        concurrency=0,
        seconds=0.0,
        out=str(out),
        jobs="",
        legacy_cpus=None,
        gap_ms=0.0,
        baseline_engine=None,
        baseline_env=[],
        baseline_runtime=None,
        omni_engine="native",
        runtime_src=None,
        prepared="/prepared",
    )
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 2
    assert row["successful_pairs"] == 2
    assert row["failed_pairs"] == 0
    assert row["success_rate"] == 1.0
    assert row["pairs"] == 2
    assert row["median_speedup"] == 2.0


def _make_ab_spec(job="embedding", input_ids=None):
    if input_ids is None:
        input_ids = ["t0", "t1", "t2"]
    return {
        "Job": job,
        "Repo": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
        "Path": "/cache/x",
        "Revision": "r",
        "Mode": "embedding",
        "Adapter": "mmbert",
        "Provider": "candle",
        "Modality": "text",
        "MaxTokens": 8192,
        "Overflow": "truncate",
        "Dimension": 0,
        "Layer": 0,
        "Inputs": [{"id": i, "text": i} for i in input_ids],
        "Repeats": 1,
        "Concurrency": 0,
        "Seconds": 0.0,
    }


def _make_ab_args(binary, out):
    return SimpleNamespace(
        binary=str(binary),
        tree="/tree",
        libs=None,
        cache="/cache",
        device="cpu",
        threads=None,
        max_num_seqs=None,
        rounds=1,
        concurrency=0,
        seconds=0.0,
        out=str(out),
        jobs="",
        legacy_cpus=None,
        gap_ms=0.0,
        baseline_engine=None,
        baseline_env=[],
        baseline_runtime=None,
        omni_engine="native",
        runtime_src=None,
        prepared="/prepared",
    )


def _patch_ab_monkeypatch(el, monkeypatch, legacy_lines, runtime_status):
    """Patch run_ab's external boundaries: legacy subprocess and runtime calls."""
    proc = SimpleNamespace()
    proc.stdin = SimpleNamespace()
    proc.stdin.write = lambda *a, **k: None
    proc.stdin.close = lambda: None
    proc._lines = legacy_lines
    proc._i = 0

    def readline():
        line = proc._lines[min(proc._i, len(proc._lines) - 1)]
        proc._i += 1
        return line + "\n"

    proc.stdout = SimpleNamespace()
    proc.stdout.readline = readline
    proc.poll = lambda: None
    proc.wait = lambda: None

    runtime = SimpleNamespace()
    runtime.stop = lambda: None

    class FakeLoop:
        def call(self, runtime, request):
            return (runtime_status, runtime_status, {})

        def close(self):
            pass

    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        el,
        "serve_runtime",
        lambda args, specs: (
            runtime,
            {"vllm-sr/Vela-1.0-Encoder-307M-Embedding": "embedding"},
        ),
    )
    monkeypatch.setattr(el, "LoopThread", FakeLoop)
    monkeypatch.setattr(
        el,
        "runtime_requests",
        lambda spec, model: [
            el.Request("embeddings", {"input": "x"}, 1) for _ in spec["Inputs"]
        ],
    )


def test_embed_legacy_ab_legacy_requests_failing(el, monkeypatch, tmp_path):
    """Legacy-side failures (-1 from the subprocess) are counted as failed."""
    specs = [_make_ab_spec(input_ids=["t0", "t1"])]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    _patch_ab_monkeypatch(el, monkeypatch, ["READY", "-1", "-1"], 200)

    args = _make_ab_args(binary, out)
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 2
    assert row["successful_pairs"] == 0
    assert row["failed_pairs"] == 2
    assert row["success_rate"] == 0.0
    assert row["failure_rate"] == 1.0
    assert row["pairs"] == 0
    assert row["median_speedup"] is None
    assert row["legacy"]["p50_ms"] is None
    assert row["legacy"]["p95_ms"] is None


def test_embed_legacy_ab_both_sides_failing(el, monkeypatch, tmp_path):
    """Both legacy and runtime failures are counted as failed pairs."""
    specs = [_make_ab_spec(input_ids=["t0", "t1"])]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    _patch_ab_monkeypatch(el, monkeypatch, ["READY", "-1", "-1"], 500)

    args = _make_ab_args(binary, out)
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 2
    assert row["successful_pairs"] == 0
    assert row["failed_pairs"] == 2
    assert row["success_rate"] == 0.0
    assert row["failure_rate"] == 1.0
    assert row["pairs"] == 0
    assert row["median_speedup"] is None


def test_embed_legacy_ab_mixed_success_and_failure(el, monkeypatch, tmp_path):
    """One successful pair, one legacy failure, one runtime failure."""
    specs = [_make_ab_spec(input_ids=["t0", "t1", "t2"])]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    # Warmup: 3 call_runtime(spec, 0) before the pair loop.
    # Pair loop: 3 call_runtime calls (index 0, 1, 2).
    # t0: both succeed; t1: legacy fails (-1); t2: runtime fails (500).
    proc_lines = ["READY", "2000", "-1", "2000"]
    runtime_statuses = [200, 200, 200, 200, 200, 500]  # 3 warmup + 3 pair

    proc = SimpleNamespace()
    proc.stdin = SimpleNamespace()
    proc.stdin.write = lambda *a, **k: None
    proc.stdin.close = lambda: None
    proc._lines = proc_lines
    proc._i = 0

    def readline():
        line = proc._lines[min(proc._i, len(proc._lines) - 1)]
        proc._i += 1
        return line + "\n"

    proc.stdout = SimpleNamespace()
    proc.stdout.readline = readline
    proc.poll = lambda: None
    proc.wait = lambda: None

    runtime = SimpleNamespace()
    runtime.stop = lambda: None

    statuses = list(runtime_statuses)

    class FakeLoop:
        def call(self, runtime, request):
            status = statuses.pop(0) if statuses else 200
            return (1000, status, {})

        def close(self):
            pass

    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        el,
        "serve_runtime",
        lambda args, specs: (
            runtime,
            {"vllm-sr/Vela-1.0-Encoder-307M-Embedding": "embedding"},
        ),
    )
    monkeypatch.setattr(el, "LoopThread", FakeLoop)
    monkeypatch.setattr(
        el,
        "runtime_requests",
        lambda spec, model: [
            el.Request("embeddings", {"input": "x"}, 1) for _ in spec["Inputs"]
        ],
    )

    args = _make_ab_args(binary, out)
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 3
    assert row["successful_pairs"] == 1
    assert row["failed_pairs"] == 2
    assert row["success_rate"] == pytest.approx(1 / 3)
    assert row["failure_rate"] == pytest.approx(2 / 3)
    assert row["pairs"] == 1
    assert row["median_speedup"] == 2.0


def test_embed_legacy_ab_empty_latency_is_null(el, monkeypatch, tmp_path):
    """Empty latency percentiles serialize as JSON null, never NaN."""
    specs = [_make_ab_spec(input_ids=["t0"])]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    _patch_ab_monkeypatch(el, monkeypatch, ["READY", "-1"], 500)

    args = _make_ab_args(binary, out)
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    payload = json.dumps(row)
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert row["legacy"]["p50_ms"] is None
    assert row["legacy"]["p95_ms"] is None
    assert row["runtime"]["p50_ms"] is None
    assert row["runtime"]["p95_ms"] is None
    assert row["median_speedup"] is None


def test_embed_legacy_ab_zero_attempted_pairs(el, monkeypatch, tmp_path):
    """Zero attempted inputs: run_ab() must report zero pairs with null rates
    and null latency percentiles, never crash on the empty warmup."""
    specs = [_make_ab_spec(input_ids=[])]
    binary = tmp_path / "legacy"
    (binary.with_suffix(".jobs.json")).write_text(json.dumps(specs))
    out = tmp_path / "ab.json"

    proc = SimpleNamespace()
    proc.stdin = SimpleNamespace()
    proc.stdin.write = lambda *a, **k: None
    proc.stdin.close = lambda: None
    proc._lines = ["READY"]
    proc._i = 0

    def readline():
        line = proc._lines[min(proc._i, len(proc._lines) - 1)]
        proc._i += 1
        return line + "\n"

    proc.stdout = SimpleNamespace()
    proc.stdout.readline = readline
    proc.poll = lambda: None
    proc.wait = lambda: None

    runtime = SimpleNamespace()
    runtime.stop = lambda: None

    class FakeLoop:
        def call(self, runtime, request):
            return (1000, 200, {})

        def close(self):
            pass

    # Empty inputs must be handled safely by the actual warmup and pair loop.
    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        el,
        "serve_runtime",
        lambda args, specs: (
            runtime,
            {"vllm-sr/Vela-1.0-Encoder-307M-Embedding": "embedding"},
        ),
    )
    monkeypatch.setattr(el, "LoopThread", FakeLoop)
    monkeypatch.setattr(
        el,
        "runtime_requests",
        lambda spec, model: [],
    )

    args = _make_ab_args(binary, out)
    el.run_ab(args)

    record = json.loads(out.read_text(encoding="utf-8"))
    row = record["embedding"]
    assert row["total_pairs"] == 0
    assert row["successful_pairs"] == 0
    assert row["failed_pairs"] == 0
    assert row["success_rate"] is None
    assert row["failure_rate"] is None
    assert row["pairs"] == 0
    assert row["median_speedup"] is None
    assert row["legacy"]["p50_ms"] is None
    assert row["legacy"]["p95_ms"] is None
    assert row["runtime"]["p50_ms"] is None
    assert row["runtime"]["p95_ms"] is None
