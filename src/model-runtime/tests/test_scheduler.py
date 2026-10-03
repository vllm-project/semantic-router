import threading
import time

import pytest
from vllm_sr_runtime.errors import RuntimeServiceError
from vllm_sr_runtime.plugins.base import Job, LoadedModel, RenderedItem
from vllm_sr_runtime.profiles.batching import BatchingProfile
from vllm_sr_runtime.profiles.exact import ExactProfile
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


def test_jobs_cancelled_while_queued_are_not_run():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        first = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        time.sleep(0.05)  # the worker holds the first job inside run()
        second = scheduler.submit([item("b", 4)], deadline=None, profile="exact")
        assert second.cancel()
        gate.set()
        assert first.result(timeout=5)
        third = scheduler.submit([item("c", 4)], deadline=None, profile="exact")
        assert third.result(timeout=5)
        assert model.calls == [["a"], ["c"]]
    finally:
        gate.set()
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
