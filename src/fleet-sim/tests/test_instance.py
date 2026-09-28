"""Regression tests for Instance preemption bookkeeping.

These drive `_start_next` directly (rather than through `advance_to`) so
each admission/eviction step can be asserted in isolation, independent of
how the surrounding simulation happens to schedule events.
"""

import heapq

from fleet_sim.core.instance import Instance, _Event
from fleet_sim.core.request import Request, RequestState
from fleet_sim.gpu_profiles.manual import ManualProfile


def _profile():
    # W=1, H=0 makes iter_latency == W regardless of active count, so
    # completion times are exactly predictable: s_eff = (1 + l_out) / n_slots.
    return ManualProfile(
        name="tiny",
        W=1.0,
        H=0.0,
        chunk=1000,
        blk_size=1,
        total_kv_blks=12,
        max_slots=100,
        cost_per_hr=1.0,
        calibration_ctx=6,
    )


def _instance():
    return Instance(instance_id=0, pool_id="pool", gpu=_profile(), max_ctx=6)


def _req(req_id, l_out):
    return Request(req_id=req_id, arrival_time=0.0, l_in=1, l_out=l_out)


class TestPreemptionQueueBookkeeping:
    def test_admission_that_evicts_removes_the_admitted_request_from_queue(self):
        inst = _instance()
        a = _req(1, l_out=9)  # 10 blocks
        b = _req(2, l_out=1)  # 2 blocks
        c = _req(3, l_out=9)  # 10 blocks; only fits once a is evicted

        for r in (a, b, c):
            inst.accept(r)

        inst._start_next(0.0)  # admits a
        inst._start_next(0.0)  # admits b; budget now exactly full (10 + 2 == 12)
        assert list(inst._queue) == [c]

        inst._start_next(1.0)  # c's own admission must evict a

        assert c not in inst._queue
        assert c in inst._active_reqs
        assert list(inst._queue) == [a]
        assert a not in inst._active_reqs
        assert a.preempted is True

    def test_stale_event_from_a_preempted_admission_is_ignored_after_readmission(self):
        inst = _instance()
        a = _req(1, l_out=9)

        # Admit `a`, matching the bookkeeping _start_next performs.
        a.start_time = 0.0
        a.state = RequestState.DECODING
        a.admission_epoch = 1
        inst._active_slots = 1
        inst._used_kv_blocks = 10
        inst._active_reqs.append(a)
        heapq.heappush(inst._events, _Event(time=5.0, req=a, epoch=1))

        # Preempt `a`, as the eviction loop in _start_next does.
        a.preempted = True
        a.admission_epoch += 1
        inst._active_reqs.remove(a)
        inst._active_slots -= 1
        inst._used_kv_blocks -= 10

        # Re-admit before the stale event's time is reached.
        inst._queue.appendleft(a)
        inst._start_next(1.0)

        assert a.preempted is False
        assert a.admission_epoch == 3  # bumped at preemption, then at re-admission

        completed = inst.advance_to(100.0)

        assert [r.req_id for r in completed] == [1]
        assert inst.total_requests == 1
        assert inst._active_slots == 0
        assert inst._used_kv_blocks == 0
