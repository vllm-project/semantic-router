"""Adaptive CPU intra-op threads: the resolver learns, and answers never change.

The record (``cpu-forward-thread-scaling-arm``) proved responses bit-identical
across intra-op thread counts on the real models; here the same guarantee is
checked on a fixture, through the real request path, with the resolver
exploring live.
"""

import vllm_srun.engines.native.threads as adaptive
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app

from .conftest import QUESTIONS, start_runtime


def test_bucket_edges():
    assert adaptive._bucket(0) == 0
    assert adaptive._bucket(31) == 0
    assert adaptive._bucket(32) == 1
    assert adaptive._bucket(2000) == 6


def test_env_gate(monkeypatch):
    monkeypatch.delenv(adaptive.ENV, raising=False)
    assert adaptive.enabled() is False
    monkeypatch.setenv(adaptive.ENV, "1")
    assert adaptive.enabled() is True


def test_resolver_learns_buckets_and_falls_back():
    resolver = adaptive.AdaptiveThreads(base=8, explore=48)
    resolver.start()
    for _ in range(24):
        small = resolver.pick(16)
        resolver.record(16, 10.0 if small == 1 else 50.0)
        large = resolver.pick(2000)
        resolver.record(2000, 100.0 if large == 8 else 300.0)
    assert resolver.pick(16) == 1  # small input: one thread won
    assert resolver.pick(2000) == 8  # large input: every thread won
    assert resolver.pick(80) == 8  # unsampled bucket keeps the configured count


def test_resolver_never_exceeds_the_configured_count():
    resolver = adaptive.AdaptiveThreads(base=2, explore=6)
    resolver.start()
    for _ in range(12):
        resolver.record(64, 10.0)
    assert resolver.counts == [1, 2]
    assert resolver.pick(64) in (1, 2)


def test_exploration_is_locked_until_the_golden_check_passes():
    """The startup golden check's two runs must see one thread count.

    A host whose answers change across thread counts (x86) fails the bitwise
    startup check when exploration cycles counts during it, so the resolver
    stays on the configured count until the check has passed.
    """
    resolver = adaptive.AdaptiveThreads(base=8, explore=8)
    assert [resolver.pick(16) for _ in range(4)] == [8, 8, 8, 8]
    resolver.record(16, 1.0)
    assert resolver._seen == 0  # load-time forwards are not sampled
    resolver.start()
    cycling = set()
    for _ in range(4):
        cycling.add(resolver.pick(16))
        resolver.record(16, 1.0)
    assert cycling == {1, 2, 4, 8}  # cycling now
    assert resolver._seen == 4


def test_siblings_apply_their_own_counts_on_the_shared_context(monkeypatch):
    """A static sibling states its own count instead of inheriting the adaptive model's last one.

    The team size lives on the process's shared CPU execution context and
    every CPU forward applies its own count through it, so the switch always
    follows that context.
    """
    calls: list[int] = []
    monkeypatch.setattr(adaptive.torch, "set_num_threads", calls.append)
    monkeypatch.setattr(adaptive.CpuTeam, "_active", None)
    adaptive.CpuTeam.apply(1)  # the adaptive model exploring, picked 1
    adaptive.CpuTeam.apply(16)  # a static sibling's forward
    adaptive.CpuTeam.apply(16)  # again: a repeated count is a no-op
    adaptive.CpuTeam.apply(2)  # the adaptive model picks another count
    assert calls == [1, 16, 2]


def test_a_count_that_changes_answers_never_reaches_the_table():
    """A shadow pass that fails to reproduce the served answer disqualifies its count.

    The served answer always comes from the configured count; a count whose
    shadow pass drifts on this host is discarded per bucket, so the table can
    only adopt counts that kept the answer identical and switching never
    turns into a different exact answer.
    """
    resolver = adaptive.AdaptiveThreads(base=8, explore=48)
    resolver.start()
    for _ in range(24):
        small = resolver.pick(16)
        resolver.record(16, 10.0, exact=small == 8)  # every count drifts
        large = resolver.pick(2000)
        resolver.record(2000, 100.0, exact=large == 8)
    assert resolver.pick(16) == 8  # only the configured count survived
    assert resolver.pick(2000) == 8


def test_answers_bit_identical_across_thread_counts(qwen3_package, monkeypatch):
    questions = dict(QUESTIONS)

    def answers(env_on: bool, threads: int) -> list[dict]:
        if env_on:
            monkeypatch.setenv(adaptive.ENV, "1")
        else:
            monkeypatch.delenv(adaptive.ENV, raising=False)
        runtime = start_runtime(qwen3_package, threads=threads)
        try:
            client = TestClient(create_app(runtime))
            out = []
            for i in range(8):
                r = client.post(
                    "/v1/decisions",
                    json={"state": f"state {i}", "questions": questions},
                )
                assert r.status_code == 200
                out.append(r.json())
            return out
        finally:
            runtime.stop()

    base = answers(env_on=False, threads=1)
    adaptive_on = answers(env_on=True, threads=4)
    assert adaptive_on == base
