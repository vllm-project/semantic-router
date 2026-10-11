"""Adaptive CPU intra-op threads: the resolver learns, and answers never change.

The record (``cpu-forward-thread-scaling-arm``) proved responses bit-identical
across intra-op thread counts on the real models; here the same guarantee is
checked on a fixture, through the real request path, with the resolver
exploring live.
"""

import pytest
import torch
import vllm_srun.engines.native.threads as adaptive
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app
from vllm_srun.engines.native.engine import NativeEngineModel

from .conftest import QUESTIONS, start_runtime


@pytest.fixture
def default_cpu_threads():
    """Leave the process's CPU thread count as these tests found it.

    The runtime tests really switch the team size; without restoring it,
    every test filed after these runs on whatever count exploration left
    behind.
    """
    default = torch.get_num_threads()
    yield
    torch.set_num_threads(default)
    adaptive.CpuTeam._active = None


def test_bucket_edges():
    assert adaptive._bucket(0) == 0
    assert adaptive._bucket(31) == 0
    assert adaptive._bucket(32) == 1
    assert adaptive._bucket(2000) == 6


def test_bit_equal_reaches_into_dicts_of_tensors():
    """Encoder outputs carry per-layer tensors in dicts; they compare by value.

    A dict branch must recurse like lists and dataclasses do — falling back
    to ``==`` raises on a multi-element tensor and took ModernBERT's shadow
    verification down with it.
    """
    same = {"0": torch.tensor([1.0, 2.0]), "1": torch.tensor([3.0])}
    assert NativeEngineModel._bit_equal(same, dict(same))
    moved = dict(same, **{"1": torch.tensor([9.0])})
    assert not NativeEngineModel._bit_equal(same, moved)
    assert not NativeEngineModel._bit_equal(same, {"0": same["0"]})


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


def test_a_drift_retires_adaptation_during_exploration():
    """One drifting shadow retires the resolver instead of disqualifying a pair.

    The served answer always comes from the configured count, and a count
    that changes it on any input must never serve: the first observed drift
    means this host's answers move across counts, so adaptation turns off
    for good and the configured count serves for the model's life.
    """
    resolver = adaptive.AdaptiveThreads(base=8, explore=48)
    resolver.start()
    for _ in range(24):
        small = resolver.pick(16)
        resolver.record(16, 10.0, exact=small == 8)  # every count drifts
        large = resolver.pick(2000)
        resolver.record(2000, 100.0, exact=large == 8)
    assert resolver.pick(16) == 8  # retired: the configured count forever
    assert resolver.pick(2000) == 8
    assert resolver._table is None


def test_drift_after_the_table_builds_retires_adaptation():
    """The learned counts keep verifying after the table builds; drift retires.

    Verification samples are inputs: a count that only agreed on the
    exploration window must not serve. The table's counts shadow the served
    answers for a few more requests, and one observed drift there retires
    adaptation exactly as during exploration.
    """
    resolver = adaptive.AdaptiveThreads(base=8, explore=12)
    resolver.start()
    for _ in range(12):
        resolver.pick(16)
        resolver.record(16, 10.0)
    assert resolver._table is not None and not resolver.adopted  # transitioning
    assert resolver._transition > 0
    resolver.record(16, 10.0, exact=False)  # a post-table shadow drifted
    assert not resolver.adopted and resolver._table is None
    assert resolver.pick(16) == 8 and resolver.pick(2000) == 8


def test_exploration_starts_once_the_runtime_is_ready(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """The runtime starts exploration once the startup golden check has passed.

    Through the real runtime path: the served model is ready — the golden
    check passed — and it samples the requests it serves.
    """
    monkeypatch.setenv(adaptive.ENV, "1")
    runtime = start_runtime(qwen3_package, threads=4)
    try:
        client = TestClient(create_app(runtime))
        resolver = runtime.served[0].model.engine_model.threads_resolver
        assert resolver is not None and resolver._started
        r = client.post(
            "/v1/decisions", json={"state": "state 0", "questions": QUESTIONS}
        )
        assert r.status_code == 200
        assert resolver._seen > 0
    finally:
        runtime.stop()


def test_the_learned_count_serves_after_adoption(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """Once exploration builds the table, the learned count serves requests.

    Through the real runtime path: past the exploration window the serving
    pass states the learned count on the shared CPU context, so the learned
    setting reaches execution instead of only feeding a table.
    """
    monkeypatch.setenv(adaptive.ENV, "1")
    runtime = start_runtime(qwen3_package, threads=4)
    try:
        client = TestClient(create_app(runtime))
        resolver = runtime.served[0].model.engine_model.threads_resolver
        for i in range(80):
            r = client.post(
                "/v1/decisions", json={"state": f"state {i}", "questions": QUESTIONS}
            )
            assert r.status_code == 200
        assert resolver.adopted  # exploration and the transition are both over
        sampled = [tokens for per in resolver._samples.values() for tokens, _ in per]
        count = resolver.pick(sampled[0])
        r = client.post(
            "/v1/decisions", json={"state": "state 80", "questions": QUESTIONS}
        )
        assert r.status_code == 200
        assert adaptive.CpuTeam._active == count
    finally:
        runtime.stop()


def test_opting_in_never_changes_an_answer(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """The served answers never change while the setting is on.

    The configured count serves every request through exploration and the
    post-table transition — sixty requests reach deep into both — so the
    assertion holds on any host, including hosts whose thread counts
    disagree bitwise: the shadow verification is exactly what retires
    adaptation there instead of ever letting a drifting count serve.
    """
    questions = dict(QUESTIONS)

    def answers(env_on: bool) -> list[dict]:
        if env_on:
            monkeypatch.setenv(adaptive.ENV, "1")
        else:
            monkeypatch.delenv(adaptive.ENV, raising=False)
        runtime = start_runtime(qwen3_package, threads=4)
        try:
            client = TestClient(create_app(runtime))
            out = []
            for i in range(60):
                r = client.post(
                    "/v1/decisions",
                    json={"state": f"state {i}", "questions": questions},
                )
                assert r.status_code == 200
                out.append(r.json())
            return out
        finally:
            runtime.stop()

    base = answers(env_on=False)
    adaptive_on = answers(env_on=True)
    assert adaptive_on == base
