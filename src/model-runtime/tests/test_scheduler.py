import dataclasses
import threading
import time

import pytest
from vllm_sr_runtime.errors import RuntimeServiceError
from vllm_sr_runtime.plugins.base import Job, LoadedModel, RenderedItem
from vllm_sr_runtime.profiles.batching import BatchingProfile
from vllm_sr_runtime.profiles.exact import ExactProfile, merged
from vllm_sr_runtime.profiles.shared_context import SharedContextProfile
from vllm_sr_runtime.scheduler import scheduler as scheduler_module
from vllm_sr_runtime.scheduler.planner import micro_batches
from vllm_sr_runtime.scheduler.scheduler import DEADLINE, Scheduler, SchedulerLimits


def item(name, length):
    return RenderedItem(
        name,
        "noul",
        list(range(length)),
        [length - 3, length - 2],
        length - 1,
        ["false", "true"],
        ["No", "Yes"],
    )


class FakeModel(LoadedModel):
    def __init__(self, budget=None, gate=None, fail=False):
        self.calls = []
        self.budget = budget
        self.gate = gate
        self.fail = fail

    def plan(self, state, questions):
        raise NotImplementedError

    def run(self, items):
        if self.gate is not None:
            self.gate.wait()
        if self.fail:
            raise RuntimeError("HIP error: device fault")
        self.calls.append([i.question_id for i in items])
        return [[0.0, float(len(i.ids))] for i in items]

    def answer(self, item, logits):
        return {}

    def forward_token_budget(self):
        return self.budget


def test_micro_batches_split_longest_first_and_keep_request_order():
    assert micro_batches([10, 20, 30], None) == [[0, 1, 2]]
    assert micro_batches([10, 20, 30], 96) == [[0, 1, 2]]
    assert micro_batches([10, 20, 30], 64) == [[1, 2], [0]]
    with pytest.raises(ValueError):
        micro_batches([100], 64)


def test_exact_profile_never_mixes_requests():
    jobs = [
        Job([item("a", 9), item("b", 17)], None, 0.0, "exact"),
        Job([item("c", 5)], None, 0.0, "exact"),
    ]
    batches = ExactProfile().plan(jobs, None)
    assert [[(id(job), idx) for job, idx in b.parts] for b in batches] == [
        [(id(jobs[0]), [0, 1])],
        [(id(jobs[1]), [0])],
    ]


def test_batching_profile_covers_every_item_once_within_the_budget():
    jobs = [
        Job([item(f"{j}{i}", 8 * (i + j + 1)) for i in range(5)], None, 0.0, "batching")
        for j in range(4)
    ]
    batches = BatchingProfile(max_batch_tokens=256).plan(jobs, None)
    seen = [
        (id(job), index)
        for batch in batches
        for job, indices in batch.parts
        for index in indices
    ]
    assert sorted(seen) == sorted((id(job), i) for job in jobs for i in range(5))
    for batch in batches:
        rows = batch.items()
        assert max(len(r.ids) for r in rows) * len(rows) <= 256 or len(rows) == 1


def test_scheduler_answers_in_item_order():
    model = FakeModel()
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        future = scheduler.submit(
            [item("a", 4), item("b", 6)], deadline=None, profile="exact"
        )
        assert future.result(timeout=5) == [[0.0, 4.0], [0.0, 6.0]]
    finally:
        scheduler.stop()


def test_expired_jobs_are_not_run():
    model = FakeModel()
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        future = scheduler.submit(
            [item("a", 4)], deadline=time.monotonic() - 1, profile="exact"
        )
        assert future.result(timeout=5) is DEADLINE
        assert model.calls == []
    finally:
        scheduler.stop()


def test_admission_refuses_beyond_the_queue_bound():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    scheduler = Scheduler(
        model, {"exact": ExactProfile()}, SchedulerLimits(max_queue=1)
    )
    scheduler.start()
    try:
        first = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        time.sleep(0.05)  # the worker holds the first job inside run()
        scheduler.submit([item("b", 4)], deadline=None, profile="exact")
        with pytest.raises(RuntimeServiceError) as error:
            scheduler.submit([item("c", 4)], deadline=None, profile="exact")
        assert error.value.code == "overloaded" and error.value.status == 429
        gate.set()
        assert first.result(timeout=5)
    finally:
        gate.set()
        scheduler.stop()


def test_unknown_profile_is_refused():
    scheduler = Scheduler(FakeModel(), {"exact": ExactProfile()})
    with pytest.raises(RuntimeServiceError):
        scheduler.submit([item("a", 4)], deadline=None, profile="batching")


def test_device_faults_fail_the_request_and_are_recorded():
    scheduler = Scheduler(FakeModel(fail=True), {"exact": ExactProfile()})
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        with pytest.raises(RuntimeError, match="device fault"):
            future.result(timeout=5)
        assert scheduler.failure is not None
    finally:
        scheduler.stop()


def test_batching_window_coalesces_concurrent_requests():
    model = FakeModel()
    scheduler = Scheduler(
        model, {"batching": BatchingProfile()}, SchedulerLimits(batch_window_ms=50)
    )
    scheduler.start()
    try:
        futures = [
            scheduler.submit([item(f"r{i}", 8)], deadline=None, profile="batching")
            for i in range(3)
        ]
        for future in futures:
            future.result(timeout=5)
        assert len(model.calls) == 1 and sorted(model.calls[0]) == ["r0", "r1", "r2"]
    finally:
        scheduler.stop()


def test_only_coalescing_profiles_wait_for_the_batching_window():
    model = FakeModel()
    scheduler = Scheduler(
        model,
        {"shared_context": SharedContextProfile()},
        SchedulerLimits(batch_window_ms=2_000),
    )
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit(
            [item("a", 8)], deadline=None, profile="shared_context"
        ).result(timeout=5)
        assert time.monotonic() - started < 1.0
    finally:
        scheduler.stop()


class FusingModel(FakeModel):
    fuse_bundled_jobs = True


def test_exact_runs_queued_requests_together_only_on_a_batch_invariant_model():
    jobs = [
        Job([item("a", 9)], None, 0.0, "exact"),
        Job([item("b", 13)], None, 0.0, "exact"),
    ]
    profile = ExactProfile()
    profile.available(FakeModel())
    assert len(profile.plan(jobs, None)) == 2
    invariant = FakeModel()
    invariant.batch_invariant = True
    profile.available(invariant)
    (batch,) = profile.plan(jobs, None)
    assert [job for job, _ in batch.parts] == jobs and batch.exact


def names(batch):
    return sorted(row.question_id for row in batch.items())


def window(name, length, start):
    return dataclasses.replace(
        item(name, length), ids=list(range(start, start + length))
    )


def test_shared_batches_hold_one_length_class_within_the_cap():
    short = Job([item("s1", 5), item("s2", 7)], None, 0.0, "exact")
    windows = Job(
        [window(f"w{i}", 512, 1000 * i) for i in range(3)] + [item("tail", 6)],
        None,
        0.0,
        "exact",
    )
    batches = merged([windows, short], cap=512)
    assert [names(batch) for batch in batches] == [
        ["s1", "s2", "tail"],
        ["w0"],
        ["w1"],
        ["w2"],
    ]


def test_shared_batches_keep_identical_sequences_together():
    first = Job([item("a", 512)], None, 0.0, "exact")
    other = Job([item("b", 500)], None, 0.0, "exact")
    twin = Job([item("c", 512)], None, 0.0, "exact")
    batches = merged([first, other, twin], cap=512)
    assert [names(batch) for batch in batches] == [["b"], ["a", "c"]]


class GatedModel(FakeModel):
    """Blocks a forward that holds a gated item until its gate opens."""

    def __init__(self, budget=None, gates=None):
        super().__init__(budget=budget)
        self.gates = gates or {}
        self.entered = threading.Event()

    def run(self, items):
        self.calls.append([i.question_id for i in items])
        for row in items:
            if row.question_id in self.gates:
                self.entered.set()
                self.gates[row.question_id].wait(5)
        return [[0.0, float(len(i.ids))] for i in items]


def started(model, limits=None, token_cost=None):
    scheduler = Scheduler(model, {"exact": ExactProfile()}, limits)
    scheduler._token_cost = token_cost
    scheduler.start()
    return scheduler


def test_a_job_is_answered_as_soon_as_its_own_batches_ran():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate, "long": gate})
    scheduler = started(model)
    try:
        scheduler.submit([item("blocker", 8)], deadline=None, profile="exact")
        assert model.entered.wait(5)
        long = scheduler.submit([item("long", 64)], deadline=None, profile="exact")
        short = scheduler.submit([item("short", 8)], deadline=None, profile="exact")
        model.entered.clear()
        gate.clear()
        threading.Timer(0.05, gate.set).start()
        assert short.result(timeout=5) == [[0.0, 8.0]]
        long.result(timeout=5)
        assert model.calls == [["blocker"], ["short"], ["long"]]
    finally:
        gate.set()
        scheduler.stop()


def test_short_requests_run_between_the_batches_of_a_long_one(monkeypatch):
    monkeypatch.setattr(scheduler_module, "COST_SMOOTHING", 0.0)
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = started(model, token_cost=1e-2)
    try:
        long = scheduler.submit(
            [item("long0", 64), item("long1", 64)], deadline=None, profile="exact"
        )
        assert model.entered.wait(5)
        short = scheduler.submit([item("short", 8)], deadline=None, profile="exact")
        time.sleep(0.02)
        gate.set()
        short.result(timeout=5)
        long.result(timeout=5)
        assert model.calls == [["long0"], ["short"], ["long1"]]
    finally:
        gate.set()
        scheduler.stop()


def test_a_long_job_runs_before_work_queued_after_its_expected_finish(monkeypatch):
    monkeypatch.setattr(scheduler_module, "COST_SMOOTHING", 0.0)
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = started(model, token_cost=1e-6)
    try:
        long = scheduler.submit(
            [item("long0", 64), item("long1", 64)], deadline=None, profile="exact"
        )
        assert model.entered.wait(5)
        time.sleep(0.02)
        late = scheduler.submit([item("late", 8)], deadline=None, profile="exact")
        gate.set()
        late.result(timeout=5)
        long.result(timeout=5)
        assert model.calls == [["long0"], ["long1"], ["late"]]
    finally:
        gate.set()
        scheduler.stop()


def test_a_job_past_its_deadline_skips_its_remaining_batches():
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = started(model)
    try:
        future = scheduler.submit(
            [item("long0", 64), item("long1", 64)],
            deadline=time.monotonic() + 0.05,
            profile="exact",
        )
        assert model.entered.wait(5)
        time.sleep(0.1)
        gate.set()
        assert future.result(timeout=5) is DEADLINE
        assert model.calls == [["long0"]]
    finally:
        gate.set()
        scheduler.stop()


def test_stopping_answers_every_planned_and_queued_job():
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = started(model)
    long = scheduler.submit(
        [item("long0", 64), item("long1", 64)], deadline=None, profile="exact"
    )
    assert model.entered.wait(5)
    queued = scheduler.submit([item("queued", 8)], deadline=None, profile="exact")
    stopper = threading.Thread(target=scheduler.stop)
    stopper.start()
    time.sleep(0.05)
    gate.set()
    stopper.join(5)
    for future in (long, queued):
        with pytest.raises(RuntimeServiceError) as error:
            future.result(timeout=5)
        assert error.value.code == "not_ready"


def test_exact_runs_a_bundle_group_as_one_batch_only_when_the_model_fuses():
    jobs = [
        Job([item("a", 9)], None, 0.0, "exact", group=7),
        Job([item("b", 17)], None, 0.0, "exact", group=7),
        Job([item("c", 5)], None, 0.0, "exact"),
    ]
    separate = ExactProfile()
    separate.available(FakeModel())
    assert len(separate.plan(jobs, None)) == 3
    fused = ExactProfile()
    fused.available(FusingModel())
    batches = fused.plan(jobs, None)
    assert [
        [job.items[0].question_id for job, _ in batch.parts] for batch in batches
    ] == [
        ["a", "b"],
        ["c"],
    ]


def test_a_group_is_queued_at_once_and_answered_per_job():
    model = FusingModel()
    exact = ExactProfile()
    exact.available(model)
    scheduler = Scheduler(model, {"exact": exact})
    scheduler.start()
    try:
        first, empty, second = scheduler.submit_group(
            [[item("a", 4)], [], [item("b", 6)]], deadline=None, profile="exact"
        )
        assert first.result(timeout=5) == [[0.0, 4.0]]
        assert empty.result(timeout=5) == []
        assert second.result(timeout=5) == [[0.0, 6.0]]
        assert model.calls == [["a", "b"]]
    finally:
        scheduler.stop()


def test_a_group_beyond_the_queue_bound_is_refused_whole():
    scheduler = Scheduler(
        FakeModel(), {"exact": ExactProfile()}, SchedulerLimits(max_queue=1)
    )
    with pytest.raises(RuntimeServiceError) as error:
        scheduler.submit_group(
            [[item("a", 4)], [item("b", 4)]], deadline=None, profile="exact"
        )
    assert error.value.code == "overloaded"
