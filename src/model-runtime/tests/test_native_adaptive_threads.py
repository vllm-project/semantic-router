"""Adaptive CPU intra-op threads: the resolver learns, and answers never change.

The record (``cpu-forward-thread-scaling-arm``) proved responses bit-identical
across intra-op thread counts on the measured Arm models; x86 hosts can move
within a bucket. The level-1 tests here hold on any host — the configured
count serves everything — and the level-2 serving test asserts whichever
path the host's numerics take: serving on a bit-identical host, retired on
one that drifts.
"""

import logging
import types

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


def test_env_levels(monkeypatch):
    monkeypatch.delenv(adaptive.ENV, raising=False)
    assert adaptive.level() == 0 and adaptive.enabled() is False
    for raw in ("1", "true", "yes", "on"):
        monkeypatch.setenv(adaptive.ENV, raw)
        assert adaptive.level() == 1 and adaptive.enabled() is True
        assert adaptive.serving_enabled() is False
    monkeypatch.setenv(adaptive.ENV, "2")
    assert adaptive.level() == 2 and adaptive.serving_enabled() is True


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
    exploration window must not serve. At level 2 the table's counts shadow
    the served answers for a few more requests, and one observed drift there
    retires adaptation exactly as during exploration.
    """
    resolver = adaptive.AdaptiveThreads(base=8, explore=12, serve=True)
    resolver.start()
    for _ in range(12):
        resolver.pick(16)
        resolver.record(16, 10.0)
    assert resolver._table is not None and not resolver.adopted  # transitioning
    assert resolver._transition > 0
    resolver.record(16, 10.0, exact=False)  # a post-table shadow drifted
    assert not resolver.adopted and resolver._table is None
    assert resolver.pick(16) == 8 and resolver.pick(2000) == 8


def test_the_warm_base_pass_pairs_with_the_shadow_timing():
    """A warm configured-count pass files under base, beside the warm shadows.

    The cold configured pass that serves the request pays cold caches; the
    shadow pays warm ones. Timing a second, warm configured pass and filing
    it under base keeps the caches out of which count looks fastest — on
    every request, including the ones whose shadow count is base itself, so
    base's samples never mix cold with warm.
    """
    resolver = adaptive.AdaptiveThreads(base=4, explore=6)
    resolver.start()
    for _ in range(3):
        picked = resolver.pick(64)  # cycles 1, 2, then base itself
        resolver.record_base(64, 8.0)  # the warm configured pass, always
        if picked != 4:
            resolver.record(64, 5.0)  # shadow pass: warm caches, faster
    assert resolver._samples[4] == [(64, 8.0)] * 3  # no cold sample ever
    assert resolver._seen == 2  # only shadow timings advance the window
    assert resolver._table is None


def test_level_1_runs_one_pass_per_request_after_adoption():
    """Past the table, level 1 stops shadowing: one forward per request.

    Once the table builds, level 1's report is done and pick()'s learned
    count has nothing left to measure — so the configured count serves in
    a single pass, even when the learned count differs from it. Before the
    fix, every request paid the served pass plus a dead shadow forever.
    """
    resolver = adaptive.AdaptiveThreads(base=4, explore=9, serve=False)
    fake = types.SimpleNamespace(
        threads_resolver=resolver,
        device=types.SimpleNamespace(type="cpu"),
        _cpu_threads=4,
        _bit_equal=NativeEngineModel._bit_equal,
    )
    resolver.start()
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        return "ok"

    for _ in range(14):  # exploration: nine shadow samples over ~14 requests
        resolver.pick(96)
        NativeEngineModel._with_threads(fake, 96, run)
    assert resolver.adopted
    calls["n"] = 0
    for _ in range(10):
        assert NativeEngineModel._with_threads(fake, 96, run) == "ok"
    assert calls["n"] == 10  # one pass per request, no dead shadow
    assert adaptive.CpuTeam._active == 4  # the configured count serves


def test_a_single_count_budget_never_explorers():
    """A budget that allows only the configured count has nothing to learn.

    With --threads 1 the allowed counts are just base: an exploration
    window could never close (no shadow advances it), so the resolver
    starts adopted and every request is one pass from the start.
    """
    resolver = adaptive.AdaptiveThreads(base=1, explore=9, serve=False)
    assert resolver.adopted and resolver.pick(96) == 1
    fake = types.SimpleNamespace(
        threads_resolver=resolver,
        device=types.SimpleNamespace(type="cpu"),
        _cpu_threads=1,
    )
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        return "ok"

    for _ in range(5):
        assert NativeEngineModel._with_threads(fake, 96, run) == "ok"
    assert calls["n"] == 5


def test_level_2_transition_only_when_serving():
    """The post-table transition exists to verify learned counts that serve.

    At level 1 the table never serves, so there is nothing to transition:
    adoption turns immediately and the exploration cost stops at its window.
    """
    reporter = adaptive.AdaptiveThreads(base=8, explore=12, serve=False)
    reporter.start()
    for _ in range(12):
        reporter.pick(16)
        reporter.record(16, 10.0)
    assert reporter.adopted and reporter._transition == 0


def test_the_table_log_names_the_state_the_resolver_is_in(caplog):
    """The learned-table log says what the learned counts now do.

    ``record`` files each sample under its count in ``counts`` order, so
    feeding a whole cycle per round times every count. A table that picked
    only the configured count verifies nothing new: level 2 serves it at
    once ("serving", no transition), level 1 keeps reporting ("report
    only"), and a count new to execution first announces its transition.
    """
    with caplog.at_level(logging.INFO, logger=adaptive.log.name):
        server = adaptive.AdaptiveThreads(base=8, explore=12, serve=True)
        server.start()
        for _ in range(3):
            for count in server.counts:  # one record cycle: [1, 2, 4, 8]
                server.record(16, 1.0 if count == server.base else 5.0)
        assert server.adopted and server._transition == 0
        assert "(serving): 0-31: 8 threads" in caplog.text

        caplog.clear()
        reporter = adaptive.AdaptiveThreads(base=8, explore=12, serve=False)
        reporter.start()
        for _ in range(3):
            for count in reporter.counts:
                reporter.record(16, 1.0 if count == reporter.base else 5.0)
        assert "(report only): 0-31: 8 threads" in caplog.text
        assert "; set --threads to apply them" in caplog.text

        caplog.clear()
        transitioning = adaptive.AdaptiveThreads(base=8, explore=12, serve=True)
        transitioning.start()
        for _ in range(3):
            for count in transitioning.counts:
                transitioning.record(16, 1.0 if count == 4 else 5.0)
        assert transitioning._transition == adaptive._TRANSITION
        assert "(serving after transition): 0-31: 4 threads" in caplog.text


def test_exploration_starts_once_the_runtime_is_ready(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """The runtime starts exploration once the startup golden check has passed.

    Through the real runtime path: the served model is ready — the golden
    check passed — and the first request either gets sampled or retires the
    learning on a host whose answers move across counts.
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
        assert resolver._seen > 0 or resolver.retired
    finally:
        runtime.stop()


def test_level_1_reports_the_learned_counts_but_never_serves_them(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """The default level learns and reports; the configured count still serves.

    Through the real runtime path: past the exploration window the table is
    built and logged, and every request — sampled or not — still runs and
    answers at the configured count, on any host.
    """
    monkeypatch.setenv(adaptive.ENV, "1")
    runtime = start_runtime(qwen3_package, threads=4)
    try:
        client = TestClient(create_app(runtime))
        resolver = runtime.served[0].model.engine_model.threads_resolver
        for i in range(100):
            r = client.post(
                "/v1/decisions", json={"state": f"state {i}", "questions": QUESTIONS}
            )
            assert r.status_code == 200
        if resolver.retired:
            assert adaptive.CpuTeam._active == 4  # a drifting host: base serves
            return
        assert resolver.adopted and not resolver.serving
        assert adaptive.CpuTeam._active == 4  # the configured count, always
    finally:
        runtime.stop()


def test_level_2_serves_the_learned_count_after_adoption(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """At level 2 the learned count serves once its verification completes.

    Through the real runtime path: exploration, then the transition, then
    the serving pass states the learned count on the shared CPU context. On
    a host whose answers move across counts the first drifting shadow
    retires the resolver instead — that branch is the level-2 safety net.
    """
    monkeypatch.setenv(adaptive.ENV, "2")
    runtime = start_runtime(qwen3_package, threads=4)
    try:
        client = TestClient(create_app(runtime))
        resolver = runtime.served[0].model.engine_model.threads_resolver
        for i in range(130):
            r = client.post(
                "/v1/decisions", json={"state": f"state {i}", "questions": QUESTIONS}
            )
            assert r.status_code == 200
        if resolver.retired:
            assert resolver.pick(64) == 4  # a drifting host keeps its count
            return
        assert resolver.adopted and resolver.serving
        sampled = [tokens for per in resolver._samples.values() for tokens, _ in per]
        count = resolver.pick(sampled[0])
        r = client.post(
            "/v1/decisions", json={"state": "state 130", "questions": QUESTIONS}
        )
        assert r.status_code == 200
        assert adaptive.CpuTeam._active == count
    finally:
        runtime.stop()


def test_opting_in_never_changes_an_answer(
    qwen3_package, monkeypatch, default_cpu_threads
):
    """The served answers never change while the setting is on.

    At the default level the configured count serves every request —
    exploration, report-only steady state, all sixty — so the answers are
    those of the plain runtime on any host, however its thread counts
    disagree bitwise; verification retires the learning there instead of
    ever letting a drifting count near an answer.
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
