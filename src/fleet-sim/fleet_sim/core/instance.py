"""Single GPU instance simulation — request-level event model.

Design
------
Each GPU instance is modeled as an M/G/c-style queue where the "c" servers
are the KV-cache slots (n_slots concurrent sequences).

When a request enters service (a slot opens up), its service times are
computed using the *current* active-slot count and mean sequence length:

    H_eff         = H * (mean_seq_len / calibration_ctx)  # attention ∝ seq_len
    decode_iter_t = W + H_eff * n_active    # memory-bandwidth-bound

Prefill chunks are roofline-checked for compute vs memory boundedness
(ComputedProfile) or fall back to decode cost (ManualProfile):

    prefill_iter_t = max(compute_bound, memory_bound)
    compute_bound  = 4 × n_heads × head_dim × chunk × (chunk/2 + kv_history)
                     × n_layers / tp / fp16_tflops

    S_raw = prefill_iters * prefill_iter_t + L_out * decode_iter_t
    S_eff = S_raw_at_full_slots / n_slots   # for queue consistency with M/G/c

TTFT is:
    TTFT = queue_wait + ceil(L_in/CHUNK) * prefill_iter_t(n_active)

Physical completion (for TPOT):
    physical_end = start_time + S_raw

KV-cache accounting:
    Each request occupies ceil((l_in + l_out) / blk_size) blocks while
    active.  Admission is gated by BOTH n_slots AND the KV block budget.
    When a new request would overflow the block budget, the longest active
    sequence is preempted (re-queued at its head position).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..gpu_profiles.profiles import GpuProfile
from .request import Request, RequestState


@dataclass
class _Event:
    """A scheduled service completion event.

    ``admission_seq`` records the request's admission generation this event
    belongs to.  When a request is preempted and later re-admitted its
    admission_seq advances, so a stale completion event from the earlier
    admission can be recognized and dropped.
    """

    time: float
    req: Request
    admission_seq: int

    def __lt__(self, other: _Event) -> bool:
        return self.time < other.time


class Instance:
    """Simulate one GPU instance using a fast request-level event model.

    Each request occupies one logical server slot from arrival to completion.
    The effective service time accounts for all n_slots running in parallel,
    matching the M/G/c analytical model.

    Parameters
    ----------
    instance_id  : unique ID within its pool
    pool_id      : ID of the parent pool
    gpu          : GpuProfile for this GPU type
    max_ctx      : maximum context length (tokens) for this pool
    chunk_mode   : "shared" (one chunk budget shared across batch) or "independent" (default; each sequence gets its own chunk budget)
    max_queue    : maximum pending request queue depth
    on_ttft      : optional callback(Request) at TTFT
    on_complete  : optional callback(Request) at completion
    """

    def __init__(
        self,
        instance_id: int,
        pool_id: str,
        gpu: GpuProfile,
        max_ctx: int,
        chunk_mode: str = "independent",
        max_queue: int = 1024,
        on_ttft: Callable[[Request], None] | None = None,
        on_complete: Callable[[Request], None] | None = None,
    ):
        self.instance_id = instance_id
        self.pool_id = pool_id
        self.gpu = gpu
        self.max_ctx = max_ctx
        self.chunk_mode = chunk_mode
        self.max_queue = max_queue
        self.on_ttft = on_ttft
        self.on_complete = on_complete

        self.n_slots: int = gpu.n_slots(max_ctx)

        # KV-cache block budget (PagedAttention)
        self._total_kv_blocks: int = gpu.total_kv_blks
        self._used_kv_blocks: int = 0

        # simulation state
        self._queue: deque[Request] = deque()
        self._active_slots: int = 0  # currently occupied server slots
        self._active_reqs: list[Request] = []  # for preemption (longest-first)
        self._events: list[_Event] = []  # min-heap of completion events
        self._now: float = 0.0

        # metrics
        self.total_requests: int = 0
        self.total_preempted: int = 0
        self.total_busy_time: float = 0.0
        self.total_idle_time: float = 0.0
        self._last_active_change: float = 0.0
        self._was_busy: bool = False

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return self._active_slots

    @property
    def is_full(self) -> bool:
        """True when no further requests can be admitted (slot OR KV budget)."""
        if self._active_slots >= self.n_slots:
            return True
        # Approximate: assume next request uses blk_size blocks (minimum)
        return self._used_kv_blocks + 1 > self._total_kv_blocks

    @property
    def current_time(self) -> float:
        return self._now

    def accept(self, req: Request) -> bool:
        """Enqueue a request for service.  Returns False if queue is full."""
        if len(self._queue) >= self.max_queue:
            return False
        req.state = RequestState.QUEUED
        req.queue_time = self._now
        req.pool_id = self.pool_id
        req.instance_id = self.instance_id
        self._queue.append(req)
        return True

    def advance_to(self, target_time: float) -> list[Request]:
        """Process all events up to target_time; dequeue and start waiting requests.

        Returns list of requests that completed (DONE state) in this window.
        """
        completed: list[Request] = []

        while True:
            # Determine the next thing to do
            next_event_t = self._events[0].time if self._events else float("inf")

            # If we can start a queued request right now (free slot), do it
            if self._queue and self._active_slots < self.n_slots:
                qlen = len(self._queue)
                if self._start_next(self._now) and len(self._queue) < qlen:
                    continue
                # Either the head could not be admitted (budget too small
                # even after preemption), or admitting it required evicting
                # another request, so the queue did not shrink and another
                # attempt would just swap victims again.  Fall through so
                # the clock advances to the next event instead of spinning.

            # Jump to next event or target_time, whichever comes first
            step = min(next_event_t, target_time)
            if step > target_time:
                break
            if step == float("inf"):
                break

            # Advance time
            prev = self._now
            self._now = step

            # Update utilisation tracking
            if self._was_busy:
                self.total_busy_time += step - prev
            else:
                self.total_idle_time += step - prev
            self._was_busy = self._active_slots > 0 or len(self._queue) > 0

            if step >= next_event_t and self._events:
                # Process all events at this time
                while self._events and self._events[0].time <= self._now:
                    ev = heapq.heappop(self._events)
                    req = ev.req
                    if req.preempted or ev.admission_seq != req.admission_seq:
                        # This completion event is stale: either the request is
                        # currently preempted (slot/blocks were released at
                        # preemption time), or it was re-admitted since this
                        # event was scheduled, in which case only the event from
                        # the latest admission may complete it.
                        continue
                    req.end_time = self._now
                    req.state = RequestState.DONE
                    self._active_slots -= 1
                    req_blocks = math.ceil((req.l_in + req.l_out) / self.gpu.blk_size)
                    self._used_kv_blocks -= req_blocks
                    if req in self._active_reqs:
                        self._active_reqs.remove(req)
                    self.total_requests += 1
                    completed.append(req)
                    if self.on_complete:
                        self.on_complete(req)

                # Fill freed slots from queue
                while self._queue and self._active_slots < self.n_slots:
                    qlen = len(self._queue)
                    if not self._start_next(self._now):
                        break
                    if len(self._queue) >= qlen:
                        # Preemption exchange: admitting this request pushed
                        # another one back to the queue.  Stop here; the
                        # main loop will advance the clock to the next event.
                        break

            if step >= target_time:
                break

        return completed

    def _start_next(self, now: float) -> bool:
        """Pull the next request from queue into active service.

        Checks KV block budget before admitting.  If the budget would be
        exceeded, preempts the longest currently-active request instead of
        admitting the new one.

        Returns True when a request was admitted, False when the head of the
        queue could not be admitted (queue empty, or budget too small for
        even a single request).  Callers must not retry in a tight loop
        when this returns False.
        """
        if not self._queue:
            return False

        # Take the candidate off the queue up front.  Preemption below
        # re-queues victims at the head, so popping after that loop would
        # remove the wrong request (the last victim) instead of this one.
        req = self._queue.popleft()
        req_blocks = math.ceil((req.l_in + req.l_out) / self.gpu.blk_size)

        n_victims = self._preempt_for(req_blocks)

        # If still no room (budget too small for even 1 req), put the
        # candidate back right after the victims that were just re-queued,
        # preserving their retry-first ordering, and skip for now.
        if self._used_kv_blocks + req_blocks > self._total_kv_blocks:
            self._queue.insert(n_victims, req)
            return False

        req.start_time = now
        req.state = RequestState.PREFILLING
        req.preempted = False  # reset preemption flag on re-admission
        req.admission_seq += 1  # invalidate completion events from before
        self._active_slots += 1
        self._used_kv_blocks += req_blocks
        self._active_reqs.append(req)

        self._schedule_completion(req, now)
        return True

    def _preempt_for(self, req_blocks: int) -> int:
        """Free KV blocks by evicting the longest active request.

        Repeatedly preempts the longest currently-active sequence until
        ``req_blocks`` fits in the budget or nothing is left to evict.
        Victims are re-queued at the head so they are retried first.

        Returns the number of requests preempted.
        """
        n_victims = 0
        while (
            self._used_kv_blocks + req_blocks > self._total_kv_blocks
            and self._active_reqs
        ):
            victim = max(self._active_reqs, key=lambda r: r.l_in + r.l_out)
            victim_blocks = math.ceil((victim.l_in + victim.l_out) / self.gpu.blk_size)
            # Invalidate the victim's pending completion event via the preempted flag
            victim.preempted = True
            self._active_reqs.remove(victim)
            self._active_slots -= 1
            self._used_kv_blocks -= victim_blocks
            # Re-queue the victim at head so it is retried first
            self._queue.appendleft(victim)
            self.total_preempted += 1
            n_victims += 1
        return n_victims

    def _schedule_completion(self, req: Request, now: float) -> None:
        """Compute service times for an admitted request and schedule its
        completion event on the event heap."""
        # ── Seq-len-aware iter_t: scale H by mean active sequence length ─────
        # Attention cost ∝ seq_len, so H_eff = H * (mean_seq_len / calibration_ctx).
        # This correctly models pools serving short requests as faster than
        # pools serving long requests, even at the same n_slots.
        if self._active_reqs:
            mean_seq_len = sum(r.l_in + r.l_out for r in self._active_reqs) / len(
                self._active_reqs
            )
        else:
            mean_seq_len = float(req.l_in + req.l_out)

        l_in = req.l_in
        l_out = req.l_out
        prefill_iters = math.ceil(l_in / self.gpu.chunk)

        # ── Prefill time: roofline check (compute-bound vs memory-bound) ──────
        # FlashAttention FLOPs scale as chunk × kv_history, which can be
        # compute-bound for large chunks on modern GPUs (H100/B200).
        # Average KV history during prefill ≈ l_in / 2 (linearly grows from 0).
        kv_history_avg = l_in / 2.0
        prefill_iter_t = self.gpu.prefill_iter_latency(
            self.gpu.chunk, kv_history_avg, self._active_slots, mean_seq_len
        )
        prefill_time = prefill_iters * prefill_iter_t
        req.first_token_time = now + prefill_time
        req.state = RequestState.DECODING
        if self.on_ttft:
            self.on_ttft(req)

        # Physical completion time (for TPOT = (finish-TTFT) / (l_out-1))
        decode_iter_t = self.gpu.iter_latency(self._active_slots, mean_seq_len)
        s_raw = prefill_time + l_out * decode_iter_t
        req.physical_end_time = now + s_raw

        # Effective service time uses n_slots (full batch) for queue scheduling
        # consistency with the M/G/c analytical model, but with the actual
        # mean sequence length so throughput estimates are accurate.
        prefill_iter_t_full = self.gpu.prefill_iter_latency(
            self.gpu.chunk, kv_history_avg, self.n_slots, mean_seq_len
        )
        decode_iter_t_full = self.gpu.iter_latency(self.n_slots, mean_seq_len)
        s_raw_full = prefill_iters * prefill_iter_t_full + l_out * decode_iter_t_full
        s_eff = s_raw_full / self.n_slots

        completion_time = now + s_eff
        heapq.heappush(self._events, _Event(completion_time, req, req.admission_seq))

    def next_event_time(self) -> float:
        """Time of the next completion event (or now if the queue can be served)."""
        if self._queue and self._active_slots < self.n_slots:
            req = self._queue[0]
            req_blocks = math.ceil((req.l_in + req.l_out) / self.gpu.blk_size)
            # A request can start now only if it fits the KV budget, either
            # directly or after preempting active requests.  Otherwise the
            # clock must wait for the next completion to free blocks.
            if (
                self._used_kv_blocks + req_blocks <= self._total_kv_blocks
                or self._active_reqs
            ):
                return self._now
        if self._events:
            return self._events[0].time
        return float("inf")

    def utilisation(self) -> float:
        total = self.total_busy_time + self.total_idle_time
        return self.total_busy_time / total if total > 0 else 0.0
