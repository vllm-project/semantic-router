"""Input-adaptive intra-op thread counts for CPU forwards (opt-in).

Record (``cpu-forward-thread-scaling-arm``): a short classify is ~99%
``forward``, and that forward is per-op dispatch, allocation and barrier
overhead rather than GEMM, so the fastest intra-op thread count for a small
input can be 1 while a long input wants every thread — and the crossover
moves with the host generation. Responses are bit-identical across intra-op
thread counts (GEMM parallelizes over output rows; the attention and
elementwise kernels over independent elements), so adapting the count per
request cannot change an ``exact`` answer.

Opt in with ``VLLM_SRUN_CPU_ADAPTIVE_THREADS=1``. The resolver cycles the
allowed counts over the first requests while recording (tokens, forward
milliseconds), then serves each token-size bucket with its fastest count; a
bucket without enough samples keeps the configured count. Forwards run
serialized on the process's single CPU device thread (``accel/cpu.py``), so
``torch.set_num_threads`` at a forward's start races with nothing. The
learned table is one-shot: it reflects the load the exploration window saw.
A process serving several CPU models should enable this for all of them or
none — an adaptive model's count persists on the shared device thread and
would otherwise become a non-adaptive sibling's next team size (answers
stay identical either way; only speed is shared).
"""

from __future__ import annotations

import math
import os
import statistics
import threading

ENV = "VLLM_SRUN_CPU_ADAPTIVE_THREADS"
_EXPLORE = 48   # forwards sampled before the learned table takes over
_MIN = 3        # samples per (bucket, count) before a bucket trusts a count


def enabled() -> bool:
    """Whether the environment opts a CPU model into adaptive threads."""
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _bucket(tokens: int) -> int:
    """Token counts in logarithmic buckets: <32, <64, <128, ..., >=2048."""
    if tokens <= 0:
        return 0
    return min(max(int(math.log2(tokens)) - 4, 0), 6)


class AdaptiveThreads:
    """Learns the fastest intra-op thread count per input-size bucket.

    ``base`` is the configured count (``EngineOptions.threads``, or
    PyTorch's own default when unset); the resolver never returns more than
    it. Until ``explore`` forwards have been sampled it cycles the allowed
    counts, so the learning traffic spreads over all of them.
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
        self._samples: dict[int, list[tuple[int, float]]] = {
            count: [] for count in self.counts
        }
        self._seen = 0
        self._table: dict[int, int] | None = None

    def pick(self, tokens: int) -> int:
        """The count this request's forward should run with."""
        with self._lock:
            if self._table is None:
                return self.counts[self._seen % len(self.counts)]
            return self._table.get(_bucket(tokens), self.base)

    def record(self, tokens: int, forward_ms: float) -> None:
        """File one forward's size and duration under the count it ran with."""
        with self._lock:
            if self._table is not None:
                return
            count = self.counts[self._seen % len(self.counts)]
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
