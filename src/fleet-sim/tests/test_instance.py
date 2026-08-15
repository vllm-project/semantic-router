"""Regression tests for Instance preemption bookkeeping (issue #2890).

Covers:
- queue desync when admission requires preempting an active request
- stale completion events firing after a request is re-admitted
- advance_to not spinning when the KV budget cannot fit anything
"""

from fleet_sim.core.instance import Instance
from fleet_sim.core.request import Request, RequestState
from fleet_sim.gpu_profiles.manual import ManualProfile


def make_instance(total_kv_blks: int = 12) -> Instance:
    """Small deterministic instance: W=1, H=0 so latencies are trivial."""
    profile = ManualProfile(
        name="test",
        W=1.0,
        H=0.0,
        chunk=1000,
        blk_size=1,
        total_kv_blks=total_kv_blks,
        max_slots=100,
        cost_per_hr=1.0,
        calibration_ctx=6,
    )
    return Instance(instance_id=0, pool_id="p", gpu=profile, max_ctx=6)


def test_admitted_request_is_removed_from_queue_when_preemption_occurs():
    """The request admitted via preemption must leave the queue exactly once.

    Regression for the first half of #2890: ``_start_next`` peeked the
    queue head, then re-queued victims with ``appendleft``, then called
    ``popleft`` which removed the last victim instead of the candidate.
    The candidate stayed in the queue while also becoming active.
    """
    inst = make_instance()
    a = Request(req_id=1, arrival_time=0, l_in=1, l_out=9)  # 10 blocks
    b = Request(req_id=2, arrival_time=0, l_in=1, l_out=1)  # 2 blocks
    c = Request(req_id=3, arrival_time=0, l_in=1, l_out=9)  # 10 blocks

    for r in (a, b, c):
        assert inst.accept(r)

    inst._start_next(0.0)  # admits a
    inst._start_next(0.0)  # admits b (budget exactly full: 10 + 2 == 12)
    admitted = inst._start_next(1.0)  # c evicts a

    assert admitted is True
    # c must be active and gone from the queue.
    assert c not in list(inst._queue), "admitted request still queued"
    assert c in inst._active_reqs
    # a was preempted and re-queued; b stayed active.
    assert a in list(inst._queue)
    assert a not in inst._active_reqs
    assert b in inst._active_reqs
    assert inst.queue_depth == 1


def test_stale_completion_event_does_not_complete_request_twice():
    """A preempted-then-readmitted request must complete exactly once.

    Regression for the second half of #2890: the victim's original
    completion event stayed in the event heap.  Once the request was
    re-admitted (``preempted`` reset), the stale event could fire and
    double-decrement slots/blocks and double-count the completion.

    Scenario: a (10 blocks) is preempted by c (2 blocks); b finishes at
    t=1 freeing a slot; a is re-admitted at t=1 (admission 2) while its
    original completion event is still scheduled for t=5 (admission 1).
    """
    inst = make_instance()
    a = Request(req_id=1, arrival_time=0, l_in=1, l_out=9)  # 10 blocks
    b = Request(req_id=2, arrival_time=0, l_in=1, l_out=1)  # 2 blocks
    c = Request(req_id=3, arrival_time=0, l_in=1, l_out=1)  # 2 blocks

    for r in (a, b, c):
        inst.accept(r)

    inst._start_next(0.0)  # admit a (completion at t=5, admission 1)
    inst._start_next(0.0)  # admit b (completion at t=1)
    inst._start_next(1.0)  # c evicts a; a queued, preempted=True
    assert a.preempted is True
    assert a not in inst._active_reqs

    # b completes at t=1, freeing a slot; a is re-admitted (admission 2).
    completed = inst.advance_to(1.0)
    assert [r.req_id for r in completed] == [2]
    assert a.admission_seq == 2

    # Run past a's stale completion (t=5, admission 1) and its real one.
    completed += inst.advance_to(100.0)
    ids = [r.req_id for r in completed]
    assert ids.count(1) == 1, f"request 1 completed {ids.count(1)} times"
    assert ids.count(2) == 1
    assert ids.count(3) == 1
    assert len(completed) == 3
    assert inst.total_requests == 3
    assert a.state == RequestState.DONE


def test_advance_to_does_not_spin_when_budget_cannot_fit_anything():
    """If the KV budget cannot fit even the head request, advance_to must
    fall through to the next event instead of looping on _start_next.

    Regression for the infinite-loop symptom in #2890: with no active
    request to preempt and no room in the budget, _start_next used to
    return without progress and the advance loop kept retrying the same
    request forever.
    """
    inst = make_instance(total_kv_blks=3)
    big = Request(req_id=1, arrival_time=0, l_in=1, l_out=9)  # 10 blocks
    assert inst.accept(big)

    completed = inst.advance_to(10.0)  # must return, not hang
    assert completed == []
    assert big in list(inst._queue)  # still waiting, not lost
    assert big not in inst._active_reqs
    assert inst.active_count == 0


def test_next_event_time_waits_when_head_cannot_be_admitted():
    """next_event_time must not claim immediate service when the head of
    the queue cannot fit the KV budget even after preemption."""
    inst = make_instance(total_kv_blks=3)
    big = Request(req_id=1, arrival_time=0, l_in=1, l_out=9)  # 10 blocks
    inst.accept(big)

    assert inst.next_event_time() == float("inf")


def test_full_run_completes_every_request_exactly_once():
    """Full run with a preemption drives every request to DONE exactly
    once, with consistent queue/active bookkeeping the whole way."""
    inst = make_instance()
    a = Request(req_id=1, arrival_time=0, l_in=1, l_out=9)  # 10 blocks
    b = Request(req_id=2, arrival_time=0, l_in=1, l_out=1)  # 2 blocks
    c = Request(req_id=3, arrival_time=0, l_in=1, l_out=1)  # 2 blocks
    for r in (a, b, c):
        inst.accept(r)

    completed = inst.advance_to(100.0)

    ids = [r.req_id for r in completed]
    assert sorted(ids) == [1, 2, 3]
    assert a.state == RequestState.DONE
    assert b.state == RequestState.DONE
    assert c.state == RequestState.DONE
    assert inst.queue_depth == 0
    assert inst.active_count == 0
    assert inst.total_requests == 3
    # b finishes first, so admitting c (2 blocks) needs no preemption.
    assert inst.total_preempted == 0
