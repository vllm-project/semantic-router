import dataclasses
import threading
import time

import pytest
from vllm_srun.errors import RuntimeServiceError
from vllm_srun.plugins.base import Job
from vllm_srun.plugins.decisions import DecisionModel, RenderedItem
from vllm_srun.profiles.batching import BatchingProfile
from vllm_srun.profiles.exact import ExactProfile, merged
from vllm_srun.profiles.shared_context import SharedContextProfile
from vllm_srun.scheduler import scheduler as scheduler_module
from vllm_srun.scheduler.planner import cost, micro_batches
from vllm_srun.scheduler.scheduler import DEADLINE, Scheduler, SchedulerLimits
from vllm_srun.timing import RunTiming


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


class FakeModel(DecisionModel):
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


def wait_until(condition, timeout=5.0):
    stop = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < stop, "condition not reached"
        time.sleep(0.001)


def test_admission_counts_every_job_until_it_is_answered(monkeypatch):
    monkeypatch.setattr(scheduler_module, "COST_SMOOTHING", 0.0)
    first, second = threading.Event(), threading.Event()
    model = GatedModel(budget=64, gates={"long0": first, "short": second})
    scheduler = started(model, SchedulerLimits(max_queue=2), token_cost=1e-2)

    def refused():
        with pytest.raises(RuntimeServiceError) as error:
            scheduler.submit([item("late", 4)], deadline=None, profile="exact")
        assert error.value.code == "overloaded" and error.value.status == 429

    try:
        long = scheduler.submit(
            [item("long0", 64), item("long1", 64)], deadline=None, profile="exact"
        )
        assert model.entered.wait(5)
        model.entered.clear()
        short = scheduler.submit([item("short", 8)], deadline=None, profile="exact")
        refused()
        first.set()
        assert model.entered.wait(5)
        # "short" runs and "long1" is planned: nothing is queued, both are pending.
        refused()
        second.set()
        short.result(timeout=5)
        long.result(timeout=5)
        wait_until(lambda: scheduler._pending_jobs == 0)
        late = scheduler.submit([item("late", 4)], deadline=None, profile="exact")
        assert late.result(timeout=5) == [[0.0, 4.0]]
    finally:
        first.set()
        second.set()
        scheduler.stop()


def test_unknown_profile_is_refused():
    scheduler = Scheduler(FakeModel(), {"exact": ExactProfile()})
    with pytest.raises(RuntimeServiceError):
        scheduler.submit([item("a", 4)], deadline=None, profile="batching")


def test_device_faults_fail_the_request_and_are_recorded():
    scheduler = Scheduler(
        FakeModel(fail=True),
        {"exact": ExactProfile()},
        device_fault=lambda error: "device fault" in str(error),
    )
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        with pytest.raises(RuntimeError, match="device fault"):
            future.result(timeout=5)
        assert scheduler.failure is not None
    finally:
        scheduler.stop()


def test_other_forward_errors_fail_only_their_batch():
    model = FakeModel(fail=True)
    scheduler = Scheduler(
        model, {"exact": ExactProfile()}, device_fault=lambda error: False
    )
    scheduler.start()
    try:
        failed = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        with pytest.raises(RuntimeError):
            failed.result(timeout=5)
        model.fail = False
        served = scheduler.submit([item("b", 4)], deadline=None, profile="exact")
        assert served.result(timeout=5) == [[0.0, 4.0]]
        assert scheduler.failure is None
    finally:
        scheduler.stop()


class WorkerModel(FakeModel):
    """Runs on any thread; records the threads and the most batches in flight at once."""

    device_thread = False

    def __init__(self, gate=None, hold=0.0):
        super().__init__(gate=gate)
        self.hold = hold
        self.threads = []
        self.inside = 0
        self.most = 0
        self.count_lock = threading.Lock()

    def run(self, items):
        with self.count_lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
            self.threads.append(threading.current_thread().name)
        try:
            time.sleep(self.hold)
            return super().run(items)
        finally:
            with self.count_lock:
                self.inside -= 1


def test_an_idle_scheduler_runs_the_group_on_the_calling_thread():
    model = WorkerModel()
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        futures = scheduler.run_now(
            [[item("a", 4)], [item("b", 6)]], deadlines=[None, None], profile="exact"
        )
        assert futures is not None and all(future.done() for future in futures)
        assert [future.result() for future in futures] == [
            [[0.0, 4.0]],
            [[0.0, 6.0]],
        ]
        assert set(model.threads) == {threading.current_thread().name}
        expired = scheduler.run_now(
            [[item("c", 4)]], deadlines=[time.monotonic() - 1], profile="exact"
        )
        assert expired[0].result() is DEADLINE
    finally:
        scheduler.stop()


def test_run_now_declines_device_thread_models_and_a_busy_scheduler():
    assert (
        Scheduler(FakeModel(), {"exact": ExactProfile()}).run_now(
            [[item("a", 4)]], deadlines=[None], profile="exact"
        )
        is None
    )
    gate = threading.Event()
    model = WorkerModel(gate=gate)
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        running = scheduler.submit([item("a", 4)], deadline=None, profile="exact")
        while not model.threads:
            time.sleep(0.001)
        assert (
            scheduler.run_now([[item("b", 4)]], deadlines=[None], profile="exact")
            is None
        )
        queued = scheduler.submit([item("c", 4)], deadline=None, profile="exact")
        assert (
            scheduler.run_now([[item("d", 4)]], deadlines=[None], profile="exact")
            is None
        )
        gate.set()
        assert running.result(timeout=5) and queued.result(timeout=5)
        assert model.threads == ["vllm-srun-worker"] * 2
    finally:
        gate.set()
        scheduler.stop()


def test_callers_and_the_worker_never_run_one_model_at_once():
    model = WorkerModel(hold=0.0005)
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    answers = []

    def caller(index):
        for round_ in range(60):
            length = 4 + (index * 60 + round_) % 7
            items = [[item(f"{index}-{round_}", length)]]
            futures = scheduler.run_now(items, deadlines=[None], profile="exact")
            if futures is None:
                futures = [scheduler.submit(items[0], deadline=None, profile="exact")]
            answers.append((length, futures[0].result(timeout=10)))

    try:
        threads = [threading.Thread(target=caller, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        scheduler.stop()
    assert model.most == 1
    assert len(answers) == 180
    assert all(result == [[0.0, float(length)]] for length, result in answers)
    assert "vllm-srun-worker" in model.threads
    assert len(set(model.threads)) > 1


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
    profile.bind(FakeModel())
    assert len(profile.plan(jobs, None)) == 2
    invariant = FakeModel()
    invariant.batch_invariant = True
    profile.bind(invariant)
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


def test_packed_rows_share_batches_across_length_classes_within_the_cap():
    short = Job([item("s", 9)], None, 0.0, "exact")
    medium = Job([item("m", 20)], None, 0.0, "exact")
    longer = Job([item("l", 40)], None, 0.0, "exact")
    windows = Job(
        [window(f"w{i}", 512, 1000 * i) for i in range(2)], None, 0.0, "exact"
    )
    batches = merged([longer, windows, short, medium], cap=512, banded=False)
    assert [names(batch) for batch in batches] == [["l", "m", "s"], ["w0"], ["w1"]]


def test_exact_bands_lengths_only_for_models_that_pad():
    jobs = [
        Job([item(name, length)], None, 0.0, "exact")
        for name, length in (("a", 9), ("b", 40))
    ]
    padded_model = FakeModel()
    padded_model.batch_invariant = True
    packed_model = FakeModel()
    packed_model.batch_invariant = True
    packed_model.packs_rows = True
    profile = ExactProfile()
    profile.bind(padded_model)
    assert [names(batch) for batch in profile.plan(jobs, None)] == [["a"], ["b"]]
    profile.bind(packed_model)
    assert [names(batch) for batch in profile.plan(jobs, None)] == [["a", "b"]]
    uninvariant = FakeModel()
    uninvariant.packs_rows = True
    profile.bind(uninvariant)
    assert len(profile.plan(jobs, None)) == 2


def test_concurrent_requests_of_different_lengths_share_one_forward_on_a_packed_model():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate})
    model.batch_invariant = True
    model.packs_rows = True
    profile = ExactProfile()
    profile.bind(model)
    scheduler = Scheduler(model, {"exact": profile})
    scheduler.start()
    try:
        scheduler.submit([item("blocker", 8)], deadline=None, profile="exact")
        assert model.entered.wait(5)
        lengths = {"r9": 9, "r20": 20, "r40": 40, "r70": 70}
        futures = {
            name: scheduler.submit([item(name, length)], deadline=None, profile="exact")
            for name, length in lengths.items()
        }
        gate.set()
        for name, future in futures.items():
            assert future.result(timeout=5) == [[0.0, float(lengths[name])]]
        assert model.calls[0] == ["blocker"]
        assert [sorted(call) for call in model.calls[1:]] == [sorted(lengths)]
    finally:
        gate.set()
        scheduler.stop()


def test_shared_batches_keep_identical_sequences_together():
    first = Job([item("a", 512)], None, 0.0, "exact")
    other = Job([item("b", 500)], None, 0.0, "exact")
    twin = Job([item("c", 512)], None, 0.0, "exact")
    batches = merged([first, other, twin], cap=512)
    assert [names(batch) for batch in batches] == [["b"], ["a", "c"]]


@dataclasses.dataclass(frozen=True)
class Media:
    question_id: str
    ids: tuple = ()
    cost: int = 300


def test_inputs_without_token_ids_count_by_their_cost():
    images = Job([Media("i1"), Media("i2")], None, 0.0, "exact")
    text = Job([item("t", 8)], None, 0.0, "exact")
    batches = merged([images, text], cap=512)
    assert [names(batch) for batch in batches] == [["t"], ["i1"], ["i2"]]
    assert cost(Media("i1")) == 300 and cost(item("t", 8)) == 8


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


def test_cancelled_jobs_are_skipped_and_the_worker_keeps_serving():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate})
    scheduler = started(model)
    try:
        blocker = scheduler.submit([item("blocker", 8)], deadline=None, profile="exact")
        assert model.entered.wait(5)
        left = scheduler.submit([item("left", 8)], deadline=None, profile="exact")
        assert left.cancel() and blocker.cancel()
        assert not scheduler._queue and scheduler._pending_jobs == 0
        gate.set()
        after = scheduler.submit([item("after", 8)], deadline=None, profile="exact")
        assert after.result(timeout=5) == [[0.0, 8.0]]
        assert model.calls == [["blocker"], ["after"]]
        wait_until(lambda: scheduler._pending_jobs == 0)
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


class GatedCallerModel(GatedModel):
    """A gated model whose batches may run on the calling thread."""

    device_thread = False

    def __init__(self, budget=None, gates=None):
        super().__init__(budget=budget, gates=gates)
        self.threads = []

    def run(self, items):
        self.threads.append(threading.current_thread().name)
        return super().run(items)


def test_a_caller_leaves_its_remaining_batches_to_the_worker_when_work_arrives(
    monkeypatch,
):
    monkeypatch.setattr(scheduler_module, "COST_SMOOTHING", 0.0)
    gate = threading.Event()
    model = GatedCallerModel(budget=64, gates={"long0": gate})
    scheduler = started(model, token_cost=1e-2)
    ran = {}

    def caller():
        ran["futures"] = scheduler.run_now(
            [[item("long0", 64), item("long1", 64)]], deadlines=[None], profile="exact"
        )

    thread = threading.Thread(target=caller, name="caller")
    try:
        thread.start()
        assert model.entered.wait(5)
        short = scheduler.submit([item("short", 8)], deadline=None, profile="exact")
        gate.set()
        thread.join(5)
        assert short.result(timeout=5) == [[0.0, 8.0]]
        assert ran["futures"][0].result(timeout=5) == [[0.0, 64.0], [0.0, 64.0]]
        assert model.calls == [["long0"], ["short"], ["long1"]]
        assert model.threads == ["caller"] + ["vllm-srun-worker"] * 2
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


def test_coalescing_jobs_join_a_busy_worker_after_one_window():
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = Scheduler(
        model,
        {"exact": ExactProfile(), "batching": BatchingProfile()},
        SchedulerLimits(batch_window_ms=20),
    )
    scheduler.start()
    try:
        long = scheduler.submit(
            [item("long0", 64), item("long1", 64)], deadline=None, profile="exact"
        )
        assert model.entered.wait(5)
        batched = scheduler.submit([item("b", 8)], deadline=None, profile="batching")
        time.sleep(0.05)
        gate.set()
        batched.result(timeout=5)
        long.result(timeout=5)
        assert ["b"] in model.calls
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
    separate.bind(FakeModel())
    assert len(separate.plan(jobs, None)) == 3
    fused = ExactProfile()
    fused.bind(FusingModel())
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
    exact.bind(model)
    scheduler = Scheduler(model, {"exact": exact})
    scheduler.start()
    try:
        first, empty, second = scheduler.submit_group(
            [[item("a", 4)], [], [item("b", 6)]], deadlines=[None] * 3, profile="exact"
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
            [[item("a", 4)], [item("b", 4)]], deadlines=[None, None], profile="exact"
        )
    assert error.value.code == "overloaded"


def test_a_group_timing_counts_a_forward_once_however_many_of_its_jobs_it_holds():
    timing = RunTiming()
    timing.ran(3, 0.010, 100.0)
    timing.ran(3, 0.010, 100.0)
    timing.ran(4, 0.005, 100.5)
    assert timing.forward == pytest.approx(0.015) and timing.done == 100.5


class SlowModel(FakeModel):
    """Every forward takes ``seconds``."""

    def __init__(self, seconds, fuse=False):
        super().__init__()
        self.seconds = seconds
        self.fuse_bundled_jobs = fuse

    def run(self, items):
        time.sleep(self.seconds)
        return super().run(items)


@pytest.mark.parametrize("fuse, forwards", [(False, 2), (True, 1)])
def test_a_group_timing_holds_the_forwards_that_ran_its_jobs(fuse, forwards):
    model = SlowModel(0.03, fuse=fuse)
    exact = ExactProfile()
    exact.bind(model)
    scheduler = Scheduler(model, {"exact": exact})
    scheduler.start()
    timing = RunTiming()
    try:
        submitted = time.monotonic()
        futures = scheduler.submit_group(
            [[item("a", 4)], [item("b", 6)]],
            deadlines=[None, None],
            profile="exact",
            timing=timing,
        )
        for future in futures:
            future.result(timeout=5)
        answered = time.monotonic()
        assert len(model.calls) == forwards
        assert forwards * 0.03 <= timing.forward <= timing.done - submitted
        assert submitted < timing.done <= answered
    finally:
        scheduler.stop()


def test_a_group_timing_leaves_the_forwards_of_other_jobs_to_its_wait():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate})
    scheduler = started(model)
    timing = RunTiming()
    try:
        scheduler.submit([item("blocker", 8)], deadline=None, profile="exact")
        assert model.entered.wait(5)
        submitted = time.monotonic()
        (future,) = scheduler.submit_group(
            [[item("mine", 8)]], deadlines=[None], profile="exact", timing=timing
        )
        time.sleep(0.05)
        gate.set()
        future.result(timeout=5)
        waited = timing.done - submitted
        assert waited >= 0.05 and timing.forward < 0.025
    finally:
        gate.set()
        scheduler.stop()


def test_a_caller_records_its_group_timing_and_a_failed_forward_counts():
    model = WorkerModel(hold=0.01)
    scheduler = Scheduler(model, {"exact": ExactProfile()})
    scheduler.start()
    try:
        timing = RunTiming()
        futures = scheduler.run_now(
            [[item("a", 4)]], deadlines=[None], profile="exact", timing=timing
        )
        assert futures is not None and futures[0].done()
        assert model.threads == [threading.current_thread().name]
        assert timing.forward >= 0.01 and timing.done > 0
    finally:
        scheduler.stop()
    failing = Scheduler(FakeModel(fail=True), {"exact": ExactProfile()})
    failing.start()
    try:
        timing = RunTiming()
        (future,) = failing.submit_group(
            [[item("a", 4)]], deadlines=[None], profile="exact", timing=timing
        )
        with pytest.raises(RuntimeError):
            future.result(timeout=5)
        assert timing.done > 0
    finally:
        failing.stop()
