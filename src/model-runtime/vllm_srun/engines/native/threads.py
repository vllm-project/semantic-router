"""Input-adaptive intra-op thread counts for CPU forwards (opt-in).

Record (``cpu-forward-thread-scaling-arm``): a short classify is ~99%
``forward``, and that forward is per-op dispatch, allocation and barrier
overhead rather than GEMM, so the fastest intra-op thread count for a small
input can be 1 while a long input wants every thread — and the crossover
moves with the host generation. On the measured hosts the responses are
bit-identical across intra-op thread counts (GEMM parallelizes over output
rows; the attention and elementwise kernels over independent elements), and
the resolver verifies that live instead of assuming it: every pass under a
non-configured count must reproduce the served answer bit for bit, and one
observed drift retires adaptation for good, leaving the configured count.

Opt in with ``VLLM_SRUN_CPU_ADAPTIVE_THREADS=1``. Exploration is locked until
the startup golden check passes (its two runs must see one thread count,
whatever the host's answer semantics across counts), and then the resolver
cycles the allowed counts over the first requests while recording (tokens,
forward milliseconds), before serving each token-size bucket with its fastest
count; a bucket without enough samples keeps the configured count. The
learned counts keep shadow-verifying for the first requests after the table
builds too — verification samples are inputs, and a count that only agreed
on the exploration window must not serve — before ``adopted`` turns and they
serve themselves. Forwards run serialized on the process's single CPU device
thread (``accel/cpu.py``), so ``torch.set_num_threads`` at a forward's start
races with nothing. The learned table is one-shot: it reflects the load the
exploration window saw. Every CPU forward states its own count on the shared
:class:`CpuTeam` — an adaptive model the resolver's pick, a static sibling
its configured one — so a batch never inherits the previous model's team
size.
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
# Requests the learned counts keep shadow-verifying after the table builds.
_TRANSITION = 16


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
    them. A shadow pass that drifts retires the resolver — adaptation only
    fits a host whose answers never move across counts. After the table
    builds, the learned counts keep verifying against the served answers for
    ``_TRANSITION`` more requests before ``adopted`` turns and they serve.
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
        self._seen = 0
        self._table: dict[int, int] | None = None
        self._transition = 0

    def start(self) -> None:
        """Begin exploring: the startup golden check has passed."""
        self._started = True

    @property
    def adopted(self) -> bool:
        """Whether the learned counts verified themselves and now serve."""
        return self._table is not None and self._transition == 0

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

        The served answer always comes from ``base``. A shadow pass under
        another count that fails to reproduce it bit for bit (``exact=False``)
        retires the resolver: one observed drift means this host's answers
        move across counts, so no count can serve without risking the exact
        profile. Records after the table builds count down the transition
        that keeps the learned counts verifying against served answers.
        """
        with self._lock:
            if not self._started:
                return
            if self._table is None:
                if not exact:
                    self._retire()
                    return
                count = self.counts[self._seen % len(self.counts)]
                self._samples[count].append((tokens, forward_ms))
                self._seen += 1
                if self._seen >= self.explore:
                    self._build()
                return
            if self._transition:
                if not exact:
                    self._retire()
                    return
                self._transition -= 1

    def _retire(self) -> None:
        """Give up adaptation: keep the configured count for the model's life."""
        self._started = False
        self._table = None
        self._transition = 0
        self._samples = {count: [] for count in self.counts}

    def _build(self) -> None:
        medians: dict[tuple[int, int], float] = {}
        for count, samples in self._samples.items():
            by_bucket: dict[int, list[float]] = {}
            for tokens, ms in samples:
                by_bucket.setdefault(_bucket(tokens), []).append(ms)
            for bucket, values in by_bucket.items():
                if len(values) >= _MIN:
                    medians[(bucket, count)] = statistics.median(values)
        table: dict[int, int] = {}
        for bucket in {bucket for bucket, _ in medians}:
            counts = [count for b, count in medians if b == bucket]
            table[bucket] = min(counts, key=lambda count: medians[(bucket, count)])
        # A bucket without enough samples on any count keeps the configured
        # count (``pick`` falls back to ``base``); an empty table means the
        # traffic never repeated a size enough to rank any count.
        self._table = table
        self._transition = _TRANSITION
