"""Input-adaptive intra-op thread counts for CPU forwards (opt-in).

Record (``cpu-forward-thread-scaling-arm``): a short classify is ~99%
``forward``, and that forward is per-op dispatch, allocation and barrier
overhead rather than GEMM, so the fastest intra-op thread count for a small
input can be 1 while a long input wants every thread — and the crossover
moves with the host generation. Responses are bit-identical across intra-op
thread counts (GEMM parallelizes over output rows; the attention and
elementwise kernels over independent elements), so adapting the count per
request cannot change an ``exact`` answer.

Opt in with ``VLLM_SRUN_CPU_ADAPTIVE_THREADS=1``. Exploration is locked until
the startup golden check passes (its two runs must see one thread count,
whatever the host's answer semantics across counts), and then the resolver
cycles the allowed counts over the first requests while recording (tokens,
forward milliseconds), before serving each token-size bucket with its fastest
count; a bucket without enough samples keeps the configured count. Forwards
run serialized on the process's single CPU device thread (``accel/cpu.py``),
so ``torch.set_num_threads`` at a forward's start races with nothing. The
learned table is one-shot: it reflects the load the exploration window saw.
Every CPU forward states its own count on the shared :class:`CpuTeam` — an
adaptive model the resolver's pick, a static sibling its configured one — so
a batch never inherits the previous model's team size.
"""

from __future__ import annotations

import math
import os
import statistics
import threading

import torch

ENV = "VLLM_SRUN_CPU_ADAPTIVE_THREADS"
_EXPLORE = 48  # forwards sampled before the learned table takes over
_MIN = 3  # samples per (bucket, count) before a bucket trusts a count


def enabled() -> bool:
    """Whether the environment opts a CPU model into adaptive threads."""
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _bucket(tokens: int) -> int:
    """Token counts in logarithmic buckets: <32, <64, <128, ..., >=2048."""
    if tokens <= 0:
        return 0
    return min(max(int(math.log2(tokens)) - 4, 0), 6)


class CpuTeam:
    """The process's shared CPU execution context: one OpenMP team size for all models.

    Every CPU forward applies its own count here and a repeated count is a
    no-op, so steady-state single-model traffic never touches
    ``torch.set_num_threads`` while mixed traffic still lands on the size its
    model asked for.
    """

    _active: int | None = None

    @classmethod
    def apply(cls, count: int) -> None:
        if count != cls._active:
            torch.set_num_threads(count)
            cls._active = count


class AdaptiveThreads:
    """Learns the fastest intra-op thread count per input-size bucket.

    ``base`` is the configured count (``EngineOptions.threads``, or
    PyTorch's own default when unset); the resolver never returns more than
    it. Until :meth:`start` (the startup golden check has passed) it is
    locked to ``base``, and until ``explore`` forwards have been sampled it
    cycles the allowed counts, so the learning traffic spreads over all of
    them.
    """

    def __init__(
        self, base: int, counts: list[int] | None = None, explore: int = _EXPLORE
    ):
        self.base = max(1, base)
        pool = counts or [2**k for k in range(8)]
        self.counts = sorted({c for c in pool if 1 <= c <= self.base})
        if self.base not in self.counts:
            self.counts.append(self.base)
        self.explore = max(explore, len(self.counts) * _MIN)
        self._lock = threading.Lock()
        self._started = False
        self._samples: dict[int, list[tuple[int, float]]] = {
            count: [] for count in self.counts
        }
        self._rejected: set[tuple[int, int]] = set()
        self._seen = 0
        self._table: dict[int, int] | None = None

    def start(self) -> None:
        """Begin exploring: the startup golden check has passed."""
        self._started = True

    def pick(self, tokens: int) -> int:
        """The count this request's forward should run with."""
        with self._lock:
            if not self._started:
                return self.base
            if self._table is None:
                return self.counts[self._seen % len(self.counts)]
            return self._table.get(_bucket(tokens), self.base)

    def record(self, tokens: int, forward_ms: float, exact: bool = True) -> None:
        """File one forward's size and duration under the count it ran with.

        The served answer always comes from ``base``; a shadow pass under
        another count that fails to reproduce it bit for bit
        (``exact=False``) disqualifies that (count, bucket) — its timing is
        discarded and the resolver can only ever adopt a count that kept the
        answer identical on this host.
        """
        with self._lock:
            if not self._started or self._table is not None:
                return
            count = self.counts[self._seen % len(self.counts)]
            if not exact:
                self._rejected.add((count, _bucket(tokens)))
            else:
                self._samples[count].append((tokens, forward_ms))
            self._seen += 1
            if self._seen >= self.explore:
                self._build()

    def _build(self) -> None:
        medians: dict[tuple[int, int], float] = {}
        for count, samples in self._samples.items():
            by_bucket: dict[int, list[float]] = {}
            for tokens, ms in samples:
                by_bucket.setdefault(_bucket(tokens), []).append(ms)
            for bucket, values in by_bucket.items():
                if len(values) >= _MIN and (count, bucket) not in self._rejected:
                    medians[(bucket, count)] = statistics.median(values)
        table: dict[int, int] = {}
        for bucket in {bucket for bucket, _ in medians}:
            counts = [count for b, count in medians if b == bucket]
            table[bucket] = min(counts, key=lambda count: medians[(bucket, count)])
        # A bucket without enough samples on any count keeps the configured
        # count (``pick`` falls back to ``base``); an empty table means the
        # traffic never repeated a size enough to rank any count.
        self._table = table
