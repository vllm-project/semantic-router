"""The local harness bridge waits for parent-owned inference and journals once."""

import concurrent.futures
import threading
import time
import urllib.error
from contextlib import contextmanager

import pytest
from cli.sr_bench import engine as engine_module
from cli.sr_bench import harness_worker
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Context, Engine
from cli.sr_bench.external import _Bridge
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure


@contextmanager
def bridge_context(tmp_path, monkeypatch, policy):
    target = {
        "id": "single",
        "kind": "single",
        "model": "physical",
        "base_url": "http://127.0.0.1:1/v1",
        "native_limits": {
            "physical": {"context_window": 8192, "max_output_tokens": 4096}
        },
        "prices": {
            "physical": {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        },
    }
    frozen = plan(
        {
            "version": "sr-bench-1.0",
            "output_policy": policy,
            "targets": [target],
            "cases": [
                {
                    "id": "case",
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": "A"}],
                    "answer": "A",
                }
            ],
            "limits": {
                "total_timeout_s": 3600,
                "case_timeout_s": 7200,
                "max_run_seconds": 7200,
            },
        }
    )
    store = Store(tmp_path)
    engine = Engine(store)
    run, _ = store.create(frozen)
    cancel = threading.Event()
    context = Context(
        engine,
        run["id"],
        frozen,
        frozen["cases"][0],
        frozen["targets"][0],
        cancel,
        time.monotonic() + 7200,
    )
    server = _Bridge(context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SR_BENCH_BRIDGE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("SR_BENCH_BRIDGE_TOKEN", server.token)
    try:
        yield context, store, run, cancel
    finally:
        cancel.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def receipt():
    return {
        "model": "physical",
        "final": "A",
        "reasoning": "Separate reasoning",
        "finish_reason": "stop",
        "output_complete": True,
        "usage": {
            "input_tokens": 1,
            "cached_input_tokens": 2,
            "cache_write_tokens": 3,
            "output_tokens": 4,
        },
        "cost_usd": 0.00001,
        "latency_s": 1,
    }


@pytest.mark.parametrize("policy", ["native", "bounded"])
def test_bridge_waits_for_delayed_parent_result_without_a_second_timeout(
    tmp_path, monkeypatch, policy
):
    started, release = threading.Event(), threading.Event()
    dispatches, socket_timeouts = [], []
    original_open = harness_worker.urllib.request.urlopen

    def open_request(request, **kwargs):
        socket_timeouts.append(kwargs.get("timeout", "ambient"))
        return original_open(request, **kwargs)

    def chat(_target, _messages, _sampling, limits, cancelled, *_args, **_kwargs):
        dispatches.append(limits["total_timeout_s"])
        started.set()
        assert release.wait(2)
        assert not cancelled()
        return receipt()

    monkeypatch.setattr(harness_worker.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(engine_module, "chat", chat)
    with bridge_context(tmp_path, monkeypatch, policy) as (context, store, run, _):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(harness_worker.call, context.case["messages"])
            try:
                assert started.wait(2)
                # The bridge has no response yet. Only the parent knows whether
                # the streaming provider is healthy or its frozen deadline expired.
                with pytest.raises(concurrent.futures.TimeoutError):
                    pending.result(timeout=0.05)
                active = store.calls(run["id"])
                assert len(active) == 1 and active[0]["status"] == "sent"
            finally:
                release.set()
            assert pending.result(timeout=2) == receipt()
        assert socket_timeouts == [None]
        assert dispatches == [3600]
        calls = store.calls(run["id"])
        assert len(calls) == 1 and calls[0]["status"] == "completed"
        assert calls[0]["cost_usd"] == receipt()["cost_usd"]
        assert calls[0]["usage"] == receipt()["usage"]
        assert calls[0]["final"] == "A"
        assert context.engine.reserved[run["id"]] == 0


@pytest.mark.parametrize("stop", ["cancel", "deadline", "transport_failure"])
def test_parent_failure_retains_call_accounting_and_never_redispatches(
    tmp_path, monkeypatch, stop
):
    started, release = threading.Event(), threading.Event()
    dispatches = []

    def chat(_target, _messages, _sampling, _limits, cancelled, *_args, **_kwargs):
        dispatches.append(True)
        started.set()
        while not release.wait(0.005):
            if cancelled():
                raise CallFailure("Parent call cancelled", receipt())
        raise CallFailure("Provider transport failed", receipt())

    monkeypatch.setattr(engine_module, "chat", chat)
    with bridge_context(tmp_path, monkeypatch, "native") as (
        context,
        store,
        run,
        cancel,
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(harness_worker.call, context.case["messages"])
            try:
                assert started.wait(2)
                if stop == "cancel":
                    cancel.set()
                elif stop == "deadline":
                    context.deadline = time.monotonic() - 1
                else:
                    release.set()
                with pytest.raises(urllib.error.HTTPError) as failure:
                    pending.result(timeout=2)
                assert failure.value.code == 502
            finally:
                release.set()
        # A failed bridge refuses another harness call; it does not retry the
        # already dispatched provider request even when its cost is known.
        with pytest.raises(urllib.error.HTTPError):
            harness_worker.call(context.case["messages"])
        calls = store.calls(run["id"])
        assert dispatches == [True] and len(calls) == 1
        assert calls[0]["status"] == (
            "failed" if stop == "transport_failure" else "cancelled"
        )
        assert calls[0]["cost_usd"] == receipt()["cost_usd"]
        assert calls[0]["usage"] == receipt()["usage"]
        assert context.engine.spent[run["id"]] == receipt()["cost_usd"]
        assert context.engine.reserved[run["id"]] == 0
