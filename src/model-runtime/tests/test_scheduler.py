import dataclasses
import threading
import time

import pytest
from vllm_srun.config import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    EngineArgs,
)
from vllm_srun.errors import RuntimeServiceError
from vllm_srun.plugins.base import Job
from vllm_srun.plugins.decisions import DecisionModel, RenderedItem
from vllm_srun.scheduler import scheduler as scheduler_module
from vllm_srun.scheduler.batching import (
    Batcher,
    SharedContext,
    SharePolicy,
    coalesced,
    merged,
    reference,
)
from vllm_srun.scheduler.planner import cost, micro_batches
from vllm_srun.scheduler.scheduler import (
    BURST_WINDOWS,
    DEADLINE,
    Scheduler,
    SchedulerLimits,
)
from vllm_srun.timing import RunTiming

ALONE = EngineArgs(max_num_seqs=1)


def batcher(model=None, args=ALONE, shared=None):
    """A batcher bound to ``model``; every row runs alone unless ``args`` lets forwards hold more."""
    planner = Batcher(args, shared)
    if model is not None:
        planner.bind(model)
    return planner


def job(items, group=None, reference=False):
    return Job(items, None, 0.0, group=group, reference=reference)


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


def test_one_seq_per_forward_runs_every_row_alone():
    jobs = [
        job([item("a", 9), item("b", 17)]),
        job([item("c", 5)]),
    ]
    planner = batcher()
    assert not planner.coalesces
    batches = planner.plan(jobs, None)
    assert [[(id(job), idx) for job, idx in b.parts] for b in batches] == [
        [(id(jobs[0]), [1])],
        [(id(jobs[0]), [0])],
        [(id(jobs[1]), [0])],
    ]


def test_a_reference_runs_each_job_in_its_released_forwards():
    jobs = [
        job([item("a", 9), item("b", 17)]),
        job([item("c", 5)]),
    ]
    batches = reference(FakeModel(), jobs, None)
    assert [[(id(job), idx) for job, idx in b.parts] for b in batches] == [
        [(id(jobs[0]), [0, 1])],
        [(id(jobs[1]), [0])],
    ]
    assert all(batch.exact for batch in batches)


def test_merged_forwards_cover_every_item_once_within_the_budget():
    jobs = [job([item(f"{j}{i}", 8 * (i + j + 1)) for i in range(5)]) for j in range(4)]
    planner = batcher(FakeModel(), EngineArgs(max_num_batched_tokens=256))
    assert planner.coalesces
    batches = planner.plan(jobs, None)
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
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4), item("b", 6)], deadline=None)
        assert future.result(timeout=5) == [[0.0, 4.0], [0.0, 6.0]]
    finally:
        scheduler.stop()


def test_expired_jobs_are_not_run():
    model = FakeModel()
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4)], deadline=time.monotonic() - 1)
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
            scheduler.submit([item("late", 4)], deadline=None)
        assert error.value.code == "overloaded" and error.value.status == 429

    try:
        long = scheduler.submit([item("long0", 64), item("long1", 64)], deadline=None)
        assert model.entered.wait(5)
        model.entered.clear()
        short = scheduler.submit([item("short", 8)], deadline=None)
        refused()
        first.set()
        assert model.entered.wait(5)
        # "short" runs and "long1" is planned: nothing is queued, both are pending.
        refused()
        second.set()
        short.result(timeout=5)
        long.result(timeout=5)
        wait_until(lambda: scheduler._pending_jobs == 0)
        late = scheduler.submit([item("late", 4)], deadline=None)
        assert late.result(timeout=5) == [[0.0, 4.0]]
    finally:
        first.set()
        second.set()
        scheduler.stop()


def test_the_cpu_row_cap_never_binds_at_the_cpu_token_budget():
    args = EngineArgs()
    assert (args.max_num_seqs, args.max_num_batched_tokens) == (None, None)
    assert args.batch_window_ms == 2.0 and args.dtype == "auto"
    limits = Batcher(args)
    assert (
        (limits.max_num_seqs, limits.max_num_batched_tokens)
        == (
            DEFAULT_MAX_NUM_SEQS,
            DEFAULT_MAX_NUM_BATCHED_TOKENS,
        )
        == (8_192, 65_536)
    )
    # The shortest row pads to 8 tokens, the most a 65,536-token forward holds.
    shortest = job([item(f"r{i}", 1) for i in range(9_000)])
    capped = coalesced([shortest], limits.max_num_batched_tokens, limits.max_num_seqs)
    uncapped = coalesced([shortest], limits.max_num_batched_tokens, 1 << 30)
    assert [len(b.items()) for b in capped] == [len(b.items()) for b in uncapped]
    assert [len(b.items()) for b in capped] == [8_192, 808]
    lengths = [1 + (i * 37) % 300 for i in range(4_000)]
    mixed = [job([item(f"m{i}", n)]) for i, n in enumerate(lengths)]
    assert [b.items() for b in coalesced(mixed, 65_536, limits.max_num_seqs)] == [
        b.items() for b in coalesced(mixed, 65_536, 1 << 30)
    ]


@pytest.mark.parametrize(
    "values, message",
    [
        ({"max_num_seqs": 0}, "max_num_seqs must be at least 1"),
        ({"max_num_batched_tokens": 4}, "max_num_batched_tokens must be at least 8"),
        ({"batch_window_ms": -1.0}, "batch_window_ms must be between 0 and 1000"),
        ({"dtype": "fp16"}, "dtype must be one of auto, float32, bfloat16, float16"),
        ({"enable_shared_context": "yes"}, "enable_shared_context must be a boolean"),
    ],
)
def test_engine_arguments_are_checked(values, message):
    with pytest.raises(ValueError, match=message):
        EngineArgs(**values)


def test_max_num_seqs_bounds_the_rows_of_one_forward():
    jobs = [job([item(f"r{i}", 8), item(f"s{i}", 8)]) for i in range(5)]
    batches = coalesced(jobs, 65_536, 3)
    assert [len(batch.items()) for batch in batches] == [3, 3, 3, 1]
    assert all(not batch.exact for batch in batches)
    seen = sorted(row.question_id for batch in batches for row in batch.items())
    assert seen == sorted(f"{p}{i}" for i in range(5) for p in "rs")
    # A request's own questions are rows too: ten questions take ten rows.
    ten = job([item(f"q{i}", 8) for i in range(10)])
    assert [len(b.items()) for b in coalesced([ten], 65_536, 4)] == [4, 4, 2]


class LaneModel(FakeModel):
    """Rows named ``img*`` read images: a lane of their own, whose forwards take ``media_seconds``."""

    def __init__(self, packs=False, media_seconds=0.0):
        super().__init__()
        self.packs_merged_rows = packs
        self.media_seconds = media_seconds

    def lane(self, item):
        return "media" if item.question_id.startswith("img") else "text"

    def run(self, items):
        if self.media_seconds and any(self.lane(i) == "media" for i in items):
            time.sleep(self.media_seconds)
        return super().run(items)


def test_merged_forwards_hold_one_lane_so_text_rows_are_not_padded_to_image_rows():
    model = LaneModel()
    planner = batcher(
        model, EngineArgs(max_num_seqs=256, max_num_batched_tokens=65_536)
    )
    jobs = [
        job([item("img0", 900), item("img1", 880)]),
        job([item("t0", 40), item("t1", 33)]),
        job([item("t2", 40)]),
    ]
    (turn,) = planner.plan(jobs, None)
    assert turn.lane == "media+text" and not turn.exact
    assert sorted(r.question_id for r in turn.items()) == [
        "img0",
        "img1",
        "t0",
        "t1",
        "t2",
    ]
    assert [(b.lane, sorted(r.question_id for r in b.items())) for b in turn.sub] == [
        ("media", ["img0", "img1"]),
        ("text", ["t0", "t1", "t2"]),
    ]
    assert all(not b.exact for b in turn.sub)
    text = turn.sub[1].items()
    assert len(text) * max(len(r.ids) for r in text) == 120


def test_packed_forwards_count_real_tokens_in_arrival_order():
    rows = [item("a", 1000), item("b", 8), item("c", 8), item("d", 1000)]
    padded_batches = coalesced([job(rows)], 2048, 256)
    assert [sorted(r.question_id for r in b.items()) for b in padded_batches] == [
        ["a", "d"],
        ["b", "c"],
    ]
    packed_batches = coalesced([job(rows)], 2048, 256, packed=True)
    assert [[r.question_id for r in b.items()] for b in packed_batches] == [
        ["a", "b", "c", "d"]
    ]
    # A lane's budget still bounds its packed forwards, and a row over it runs alone.
    split = coalesced([job([*rows, item("e", 3000)])], 2048, 256, packed=True)
    assert [[r.question_id for r in b.items()] for b in split] == [
        ["a", "b", "c", "d"],
        ["e"],
    ]


def test_each_lane_takes_its_own_budget():
    model = LaneModel(packs=True)
    model.coalesced_token_budget = lambda lane="text": {"text": 64, "media": 2048}[lane]
    planner = batcher(
        model, EngineArgs(max_num_seqs=256, max_num_batched_tokens=65_536)
    )
    jobs = [
        job(
            [item(f"img{i}", 500) for i in range(4)]
            + [item(f"t{i}", 32) for i in range(4)]
        )
    ]
    turns = planner.plan(jobs, None)
    # The n-th batches of the lanes share a turn: one turn per batch of the busiest lane.
    assert [(b.lane, len(b.items())) for b in turns] == [
        ("media+text", 6),
        ("text", 2),
    ]
    assert [(b.lane, len(b.items())) for b in turns[0].sub] == [
        ("media", 4),
        ("text", 2),
    ]


def test_a_turn_answers_its_text_rows_before_its_image_forward_ends():
    gate = threading.Event()

    class Gated(LaneModel):
        def run(self, items):
            if any(self.lane(i) == "media" for i in items):
                gate.wait(5)
            return super().run(items)

    model = Gated()
    args = EngineArgs(max_num_seqs=256, max_num_batched_tokens=65_536)
    scheduler = Scheduler(
        model, batcher(model, args), SchedulerLimits(batch_window_ms=50.0)
    )
    scheduler.start()
    try:
        image, text = scheduler.submit_group(
            [[item("img0", 900)], [item("t0", 40), item("t1", 40)]],
            deadlines=[None, None],
        )
        # One turn holds both lanes; the text forward runs first and is answered at once.
        assert text.result(timeout=5) == [[0.0, 40.0], [0.0, 40.0]]
        assert not image.done()
        gate.set()
        assert image.result(timeout=5) == [[0.0, 900.0]]
        assert model.calls == [["t0", "t1"], ["img0"]]
    finally:
        gate.set()
        scheduler.stop()


def test_image_forwards_do_not_raise_the_cost_of_text_rows():
    model = LaneModel(media_seconds=0.05)
    args = EngineArgs(max_num_seqs=256, max_num_batched_tokens=65_536)
    scheduler = Scheduler(
        model, batcher(model, args), SchedulerLimits(batch_window_ms=0.0)
    )
    scheduler.start()
    try:
        scheduler.submit([item("t0", 64)], deadline=None).result(timeout=5)
        text = scheduler._token_cost["text"]
        scheduler.submit([item("img0", 64)], deadline=None).result(timeout=5)
        assert scheduler._token_cost["text"] == text
        assert scheduler._token_cost["media"] >= 0.05 / 64
        assert scheduler._token_cost["media"] > 10 * text
    finally:
        scheduler.stop()


def test_a_reference_job_runs_alone_whatever_the_engine_arguments():
    model = FakeModel()
    scheduler = Scheduler(
        model, batcher(model, EngineArgs()), SchedulerLimits(batch_window_ms=2_000)
    )
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit(
            [item("a", 8), item("b", 12)], deadline=None, reference=True
        ).result(timeout=5)
        assert time.monotonic() - started < 1.0
        assert model.calls == [["a", "b"]]
    finally:
        scheduler.stop()


def test_device_faults_fail_the_request_and_are_recorded():
    scheduler = Scheduler(
        FakeModel(fail=True),
        batcher(),
        device_fault=lambda error: "device fault" in str(error),
    )
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4)], deadline=None)
        with pytest.raises(RuntimeError, match="device fault"):
            future.result(timeout=5)
        assert scheduler.failure is not None
    finally:
        scheduler.stop()


def test_other_forward_errors_fail_only_their_batch():
    model = FakeModel(fail=True)
    scheduler = Scheduler(model, batcher(model), device_fault=lambda error: False)
    scheduler.start()
    try:
        failed = scheduler.submit([item("a", 4)], deadline=None)
        with pytest.raises(RuntimeError):
            failed.result(timeout=5)
        model.fail = False
        served = scheduler.submit([item("b", 4)], deadline=None)
        assert served.result(timeout=5) == [[0.0, 4.0]]
        assert scheduler.failure is None
    finally:
        scheduler.stop()


@pytest.mark.parametrize("result_count", [0, 1, 3])
@pytest.mark.parametrize("inline", [False, True], ids=["worker", "inline"])
def test_wrong_result_count_fails_every_job_and_keeps_serving(
    monkeypatch, result_count, inline
):
    model = FakeModel()
    model.device_thread = not inline
    run = model.run

    def wrong_once(items):
        values = run(items)
        if len(model.calls) == 1:
            return (values + values)[:result_count]
        return values

    monkeypatch.setattr(model, "run", wrong_once)
    scheduler = Scheduler(model, batcher(model, EngineArgs(max_num_seqs=8)))
    scheduler.start()
    try:
        submit = scheduler.run_now if inline else scheduler.submit_group
        futures = submit([[item("a", 8)], [item("b", 8)]], deadlines=[None, None])
        assert futures is not None
        for future in futures:
            with pytest.raises(
                RuntimeError,
                match=f"model result count mismatch: expected 2, got {result_count}",
            ):
                future.result(timeout=5)
        assert model.calls == [["a", "b"]]
        assert scheduler.failure is None
        assert scheduler._thread.is_alive()
        wait_until(lambda: scheduler._pending_jobs == 0)
        served = scheduler.submit([item("later", 8)], deadline=None)
        assert served.result(timeout=5) == [[0.0, 8.0]]
        assert model.calls == [["a", "b"], ["later"]]
    finally:
        scheduler.stop()


class OutOfMemoryModel(FakeModel):
    def run(self, items):
        import torch

        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_a_forward_out_of_device_memory_answers_request_too_large():
    scheduler = Scheduler(
        OutOfMemoryModel(), batcher(), device_fault=lambda error: False
    )
    scheduler.start()
    try:
        future = scheduler.submit([item("a", 4)], deadline=None)
        with pytest.raises(RuntimeServiceError) as caught:
            future.result(timeout=5)
        assert caught.value.code == "request_too_large"
        assert scheduler.failure is None
    finally:
        scheduler.stop()


def test_a_shared_forward_out_of_device_memory_answers_overloaded():
    import torch

    error = scheduler_module._out_of_memory(torch.OutOfMemoryError("oom"), jobs=2)
    assert error is not None and error.code == "overloaded"
    assert scheduler_module._out_of_memory(RuntimeError("other"), jobs=1) is None


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
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        futures = scheduler.run_now(
            [[item("a", 4)], [item("b", 6)]], deadlines=[None, None]
        )
        assert futures is not None and all(future.done() for future in futures)
        assert [future.result() for future in futures] == [
            [[0.0, 4.0]],
            [[0.0, 6.0]],
        ]
        assert set(model.threads) == {threading.current_thread().name}
        expired = scheduler.run_now([[item("c", 4)]], deadlines=[time.monotonic() - 1])
        assert expired[0].result() is DEADLINE
    finally:
        scheduler.stop()


def test_run_now_declines_device_thread_models_and_a_busy_scheduler():
    assert (
        Scheduler(FakeModel(), batcher()).run_now([[item("a", 4)]], deadlines=[None])
        is None
    )
    gate = threading.Event()
    model = WorkerModel(gate=gate)
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        running = scheduler.submit([item("a", 4)], deadline=None)
        while not model.threads:
            time.sleep(0.001)
        assert scheduler.run_now([[item("b", 4)]], deadlines=[None]) is None
        queued = scheduler.submit([item("c", 4)], deadline=None)
        assert scheduler.run_now([[item("d", 4)]], deadlines=[None]) is None
        gate.set()
        assert running.result(timeout=5) and queued.result(timeout=5)
        assert model.threads == ["vllm-srun-worker"] * 2
    finally:
        gate.set()
        scheduler.stop()


def test_callers_and_the_worker_never_run_one_model_at_once():
    model = WorkerModel(hold=0.0005)
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    answers = []

    def caller(index):
        for round_ in range(60):
            length = 4 + (index * 60 + round_) % 7
            items = [[item(f"{index}-{round_}", length)]]
            futures = scheduler.run_now(items, deadlines=[None])
            if futures is None:
                futures = [scheduler.submit(items[0], deadline=None)]
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
        model, batcher(model, EngineArgs()), SchedulerLimits(batch_window_ms=50)
    )
    scheduler.start()
    try:
        futures = [
            scheduler.submit([item(f"r{i}", 8)], deadline=None) for i in range(3)
        ]
        for future in futures:
            future.result(timeout=5)
        assert len(model.calls) == 1 and sorted(model.calls[0]) == ["r0", "r1", "r2"]
    finally:
        scheduler.stop()


def test_a_job_served_alone_does_not_wait_for_the_batching_window():
    model = FakeModel()
    scheduler = Scheduler(
        model, batcher(model, EngineArgs()), SchedulerLimits(batch_window_ms=2_000)
    )
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit_group([[item("a", 8)]], deadlines=[None], alone=True)[
            0
        ].result(timeout=5)
        assert time.monotonic() - started < 1.0
    finally:
        scheduler.stop()


def test_a_window_waits_for_requests_still_being_planned_when_it_opened():
    model = FakeModel()
    planning = threading.Event()
    planning.set()
    opened: list[float] = []

    def still_planning(window_opened: float) -> bool:
        opened.append(window_opened)
        return planning.is_set()

    scheduler = Scheduler(
        model,
        batcher(model, EngineArgs()),
        SchedulerLimits(batch_window_ms=100),
        planning=still_planning,
    )
    scheduler.start()
    try:
        first = scheduler.submit([item("a", 8)], deadline=None)
        # Past the window, within BURST_WINDOWS of it: the second request is still on its way.
        time.sleep(0.15)
        second = scheduler.submit([item("b", 8)], deadline=None)
        planning.clear()
        first.result(timeout=5)
        second.result(timeout=5)
        assert len(model.calls) == 1 and opened
    finally:
        scheduler.stop()


def test_a_window_waits_for_a_burst_at_most_burst_windows():
    model = FakeModel()
    scheduler = Scheduler(
        model,
        batcher(model, EngineArgs()),
        SchedulerLimits(batch_window_ms=100),
        planning=lambda opened: True,
    )
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit([item("a", 8)], deadline=None).result(timeout=5)
        elapsed = time.monotonic() - started
        assert 0.1 * BURST_WINDOWS - 0.05 <= elapsed < 0.1 * BURST_WINDOWS + 0.7
    finally:
        scheduler.stop()


def test_only_merging_models_wait_for_the_batching_window():
    model = FakeModel()
    scheduler = Scheduler(
        model,
        batcher(model, ALONE, SharedContext(SharePolicy(min_shared_tokens=1))),
        SchedulerLimits(batch_window_ms=2_000),
    )
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit([item("a", 8)], deadline=None).result(timeout=5)
        assert time.monotonic() - started < 1.0
    finally:
        scheduler.stop()


def test_a_batch_invariant_model_merges_queued_requests_without_waiting():
    invariant = FakeModel()
    invariant.batch_invariant = True
    planner = batcher(invariant, EngineArgs())
    assert not planner.coalesces
    scheduler = Scheduler(invariant, planner, SchedulerLimits(batch_window_ms=2_000))
    scheduler.start()
    try:
        started = time.monotonic()
        scheduler.submit([item("a", 8)], deadline=None).result(timeout=5)
        assert time.monotonic() - started < 1.0
    finally:
        scheduler.stop()


def test_queued_requests_share_an_exact_forward_only_on_a_batch_invariant_model():
    jobs = [
        job([item("a", 9)]),
        job([item("b", 13)]),
    ]
    assert len(batcher(FakeModel()).plan(jobs, None)) == 2
    invariant = FakeModel()
    invariant.batch_invariant = True
    assert len(batcher(invariant).plan(jobs, None)) == 2
    (batch,) = batcher(invariant, EngineArgs()).plan(jobs, None)
    assert [job for job, _ in batch.parts] == jobs and batch.exact
    (merged_batch,) = batcher(FakeModel(), EngineArgs()).plan(jobs, None)
    assert len(merged_batch.parts) == 2 and not merged_batch.exact


def names(batch):
    return sorted(row.question_id for row in batch.items())


def window(name, length, start):
    return dataclasses.replace(
        item(name, length), ids=list(range(start, start + length))
    )


def test_shared_batches_hold_one_length_class_within_the_cap():
    short = job([item("s1", 5), item("s2", 7)])
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
    short = job([item("s", 9)])
    medium = job([item("m", 20)])
    longer = job([item("l", 40)])
    windows = Job(
        [window(f"w{i}", 512, 1000 * i) for i in range(2)], None, 0.0, "exact"
    )
    batches = merged([longer, windows, short, medium], cap=512, banded=False)
    assert [names(batch) for batch in batches] == [["l", "m", "s"], ["w0"], ["w1"]]


def test_batch_invariant_merging_bands_lengths_only_for_models_that_pad():
    jobs = [job([item(name, length)]) for name, length in (("a", 9), ("b", 40))]
    padded_model = FakeModel()
    padded_model.batch_invariant = True
    packed_model = FakeModel()
    packed_model.batch_invariant = True
    packed_model.packs_rows = True
    merging = EngineArgs()
    assert [
        names(batch) for batch in batcher(padded_model, merging).plan(jobs, None)
    ] == [["a"], ["b"]]
    assert [
        names(batch) for batch in batcher(packed_model, merging).plan(jobs, None)
    ] == [["a", "b"]]
    uninvariant = FakeModel()
    uninvariant.packs_rows = True
    assert len(batcher(uninvariant).plan(jobs, None)) == 2


def test_concurrent_requests_of_different_lengths_share_one_forward_on_a_packed_model():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate})
    model.batch_invariant = True
    model.packs_rows = True
    scheduler = Scheduler(model, batcher(model, EngineArgs()))
    scheduler.start()
    try:
        scheduler.submit([item("blocker", 8)], deadline=None)
        assert model.entered.wait(5)
        lengths = {"r9": 9, "r20": 20, "r40": 40, "r70": 70}
        futures = {
            name: scheduler.submit([item(name, length)], deadline=None)
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
    first = job([item("a", 512)])
    other = job([item("b", 500)])
    twin = job([item("c", 512)])
    batches = merged([first, other, twin], cap=512)
    assert [names(batch) for batch in batches] == [["b"], ["a", "c"]]


@dataclasses.dataclass(frozen=True)
class Media:
    question_id: str
    ids: tuple = ()
    cost: int = 300


def test_inputs_without_token_ids_count_by_their_cost():
    images = job([Media("i1"), Media("i2")])
    text = job([item("t", 8)])
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
    scheduler = Scheduler(model, batcher(model), limits)
    if token_cost is not None:
        scheduler._token_cost = {"text": token_cost}
    scheduler.start()
    return scheduler


def test_a_job_is_answered_as_soon_as_its_own_batches_ran():
    gate = threading.Event()
    model = GatedModel(gates={"blocker": gate, "long": gate})
    scheduler = started(model)
    try:
        scheduler.submit([item("blocker", 8)], deadline=None)
        assert model.entered.wait(5)
        long = scheduler.submit([item("long", 64)], deadline=None)
        short = scheduler.submit([item("short", 8)], deadline=None)
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
        blocker = scheduler.submit([item("blocker", 8)], deadline=None)
        assert model.entered.wait(5)
        left = scheduler.submit([item("left", 8)], deadline=None)
        assert left.cancel() and blocker.cancel()
        assert not scheduler._queue and scheduler._pending_jobs == 0
        gate.set()
        after = scheduler.submit([item("after", 8)], deadline=None)
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
        long = scheduler.submit([item("long0", 64), item("long1", 64)], deadline=None)
        assert model.entered.wait(5)
        short = scheduler.submit([item("short", 8)], deadline=None)
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
            [[item("long0", 64), item("long1", 64)]], deadlines=[None]
        )

    thread = threading.Thread(target=caller, name="caller")
    try:
        thread.start()
        assert model.entered.wait(5)
        short = scheduler.submit([item("short", 8)], deadline=None)
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
        long = scheduler.submit([item("long0", 64), item("long1", 64)], deadline=None)
        assert model.entered.wait(5)
        time.sleep(0.02)
        late = scheduler.submit([item("late", 8)], deadline=None)
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
        batcher(model, EngineArgs(max_num_batched_tokens=64)),
        SchedulerLimits(batch_window_ms=20),
    )
    scheduler.start()
    try:
        long = scheduler.submit([item("long0", 64), item("long1", 64)], deadline=None)
        assert model.entered.wait(5)
        batched = scheduler.submit([item("b", 8)], deadline=None)
        time.sleep(0.05)
        gate.set()
        batched.result(timeout=5)
        long.result(timeout=5)
        assert ["b"] in model.calls
    finally:
        gate.set()
        scheduler.stop()


def test_requests_that_arrive_while_a_batch_waits_share_its_forward():
    blocker, first = threading.Event(), threading.Event()
    model = GatedModel(gates={"blocker": blocker, "a": first})
    scheduler = Scheduler(
        model,
        batcher(model, EngineArgs(max_num_batched_tokens=16)),
        SchedulerLimits(batch_window_ms=20),
    )
    scheduler.start()
    try:
        scheduler.submit([item("blocker", 8)], deadline=None)
        assert model.entered.wait(5)
        model.entered.clear()
        waiting = [scheduler.submit([item(name, 8)], deadline=None) for name in "abc"]
        blocker.set()
        # [a, b] runs, [c] waits for the device.
        assert model.entered.wait(5)
        late = scheduler.submit([item("d", 8)], deadline=None)
        time.sleep(0.05)
        first.set()
        for future in [*waiting, late]:
            future.result(timeout=5)
        assert model.calls == [["blocker"], ["a", "b"], ["c", "d"]]
    finally:
        blocker.set()
        first.set()
        scheduler.stop()


def test_a_job_past_its_deadline_skips_its_remaining_batches():
    gate = threading.Event()
    model = GatedModel(budget=64, gates={"long0": gate})
    scheduler = started(model)
    try:
        future = scheduler.submit(
            [item("long0", 64), item("long1", 64)],
            deadline=time.monotonic() + 0.05,
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
    long = scheduler.submit([item("long0", 64), item("long1", 64)], deadline=None)
    assert model.entered.wait(5)
    queued = scheduler.submit([item("queued", 8)], deadline=None)
    stopper = threading.Thread(target=scheduler.stop)
    stopper.start()
    time.sleep(0.05)
    gate.set()
    stopper.join(5)
    for future in (long, queued):
        with pytest.raises(RuntimeServiceError) as error:
            future.result(timeout=5)
        assert error.value.code == "not_ready"


def test_a_bundles_jobs_share_forwards_with_other_requests():
    jobs = [
        job([item("a", 9)], group=7),
        job([item("b", 17)], group=7),
        job([item("c", 5)]),
    ]
    assert len(batcher(FakeModel()).plan(jobs, None)) == 3
    (batch,) = batcher(FakeModel(), EngineArgs()).plan(jobs, None)
    assert names(batch) == ["a", "b", "c"]


def test_a_group_is_queued_at_once_and_answered_per_job():
    model = FakeModel()
    scheduler = Scheduler(model, batcher(model, EngineArgs()))
    scheduler.start()
    try:
        first, empty, second = scheduler.submit_group(
            [[item("a", 4)], [], [item("b", 6)]], deadlines=[None] * 3
        )
        assert first.result(timeout=5) == [[0.0, 4.0]]
        assert empty.result(timeout=5) == []
        assert second.result(timeout=5) == [[0.0, 6.0]]
        assert [sorted(call) for call in model.calls] == [["a", "b"]]
    finally:
        scheduler.stop()


def test_a_group_beyond_the_queue_bound_is_refused_whole():
    scheduler = Scheduler(FakeModel(), batcher(), SchedulerLimits(max_queue=1))
    with pytest.raises(RuntimeServiceError) as error:
        scheduler.submit_group([[item("a", 4)], [item("b", 4)]], deadlines=[None, None])
    assert error.value.code == "overloaded"


def test_a_group_timing_counts_a_forward_once_however_many_of_its_jobs_it_holds():
    timing = RunTiming()
    timing.ran(3, 0.010, 100.0)
    timing.ran(3, 0.010, 100.0)
    timing.ran(4, 0.005, 100.5)
    assert timing.forward == pytest.approx(0.015) and timing.done == 100.5


class SlowModel(FakeModel):
    """Every forward takes ``seconds``."""

    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def run(self, items):
        time.sleep(self.seconds)
        return super().run(items)


@pytest.mark.parametrize("args, forwards", [(ALONE, 2), (EngineArgs(), 1)])
def test_a_group_timing_holds_the_forwards_that_ran_its_jobs(args, forwards):
    model = SlowModel(0.03)
    scheduler = Scheduler(model, batcher(model, args))
    scheduler.start()
    timing = RunTiming()
    try:
        submitted = time.monotonic()
        futures = scheduler.submit_group(
            [[item("a", 4)], [item("b", 6)]],
            deadlines=[None, None],
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
        scheduler.submit([item("blocker", 8)], deadline=None)
        assert model.entered.wait(5)
        submitted = time.monotonic()
        (future,) = scheduler.submit_group(
            [[item("mine", 8)]], deadlines=[None], timing=timing
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
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        timing = RunTiming()
        futures = scheduler.run_now([[item("a", 4)]], deadlines=[None], timing=timing)
        assert futures is not None and futures[0].done()
        assert model.threads == [threading.current_thread().name]
        assert timing.forward >= 0.01 and timing.done > 0
    finally:
        scheduler.stop()
    failing = Scheduler(FakeModel(fail=True), batcher())
    failing.start()
    try:
        timing = RunTiming()
        (future,) = failing.submit_group(
            [[item("a", 4)]], deadlines=[None], timing=timing
        )
        with pytest.raises(RuntimeError):
            future.result(timeout=5)
        assert timing.done > 0
    finally:
        failing.stop()
