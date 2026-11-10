"""Input-adaptive intra-op thread counts for CPU forwards (opt-in).

Record (``cpu-forward-thread-scaling-arm``): a short classify is ~99%
``forward``, and that forward is per-op dispatch, allocation and barrier
overhead rather than GEMM, so the fastest intra-op thread count for a small
input can be 1 while a long input wants every thread — and the crossover
moves with the host generation. Whether the answers survive a change of
count is a property of the host and its kernels, not a law of the thread
API: on the measured Arm hosts they are bit-identical across counts, on
x86 they can move within a bucket, input-dependently. The resolver never
assumes either way: it verifies live, and one observed drift retires the
learning for good.

``VLLM_SRUN_CPU_ADAPTIVE_THREADS`` levels:

- ``1``: learn and report. Exploration runs after the startup golden check
  passes — the configured count serves every request, the allowed counts
  run as timed shadow passes that must reproduce the served answer bit for
  bit — and the fastest count per token-size bucket is logged when the
  table builds. The answers never leave the configured count; point
  ``--threads`` at the learned optimum to apply it (one count is input-safe
  by definition; only a change of count can move an answer).
- ``2``: also serve the learned counts. Only meaningful on hosts where
  bit-identity across counts holds — the Arm record — because x86 drifts
  input-dependently within a bucket, so no verification window can vouch
  for the inputs it did not see. The learned counts keep shadow-verifying
  for a transition after the table builds, and one drift retires
  everything, but a host that drifts only beyond the window can still move
  an exact answer; that risk is what level 2 accepts.

Forwards run serialized on the process's single CPU device thread
(``accel/cpu.py``), so ``torch.set_num_threads`` at a forward's start
races with nothing. Every CPU forward states its own count on the shared
:class:`CpuTeam` — an adaptive model its pick, a static sibling its
configured one — so a batch never inherits the previous model's team size.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import threading

import torch

ENV = "VLLM_SRUN_CPU_ADAPTIVE_THREADS"
_LEARN_LEVEL = 1  # explore, verify, report the learned counts
_SERVE_LEVEL = 2  # also serve the learned counts once verified
_EXPLORE = 48  # forwards sampled before the learned table takes over
_MIN = 3  # samples per (bucket, count) before a bucket trusts a count
# Requests the learned counts keep shadow-verifying after the table builds.
_TRANSITION = 16

log = logging.getLogger(__name__)


def level() -> int:
    """The opt-in level: 0 off, 1 learn and report, 2 also serve learned counts."""
    raw = os.environ.get(ENV, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return 0
    if raw == "2":
        return _SERVE_LEVEL
    return _LEARN_LEVEL


def enabled() -> bool:
    """Whether the environment opts a CPU model into adaptive threads."""
    return level() >= _LEARN_LEVEL


def serving_enabled() -> bool:
    """Whether learned counts may serve requests, not only shadow them."""
    return level() >= _SERVE_LEVEL


def _bucket(tokens: int) -> int:
    """Token counts in logarithmic buckets: <32, <64, <128, ..., >=1024."""
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
    them. A shadow pass that drifts retires the learning — adaptation only
    fits a host whose answers never move across counts. ``serve`` turns on
    the level-2 behaviour: after the table builds, the learned counts keep
    verifying against the served answers for ``_TRANSITION`` requests before
    ``adopted`` turns and they serve.
    """

    def __init__(
        self,
        base: int,
        counts: list[int] | None = None,
        explore: int = _EXPLORE,
        serve: bool = False,
    ):
        self.base = max(1, base)
        pool = counts or [2**k for k in range(8)]
        self.counts = sorted({c for c in pool if 1 <= c <= self.base})
        if self.base not in self.counts:
            self.counts.append(self.base)
        if self.counts == [self.base]:
            # The budget allows only the configured count: nothing to
            # explore, and an exploration window could never close.
            self._table: dict[int, int] | None = {}
        else:
            self._table = None
        self.explore = max(explore, len(self.counts) * _MIN)
        self.serving = serve
        self._lock = threading.Lock()
        self._started = False
        self._retired = False
        self._samples: dict[int, list[tuple[int, float]]] = {
            count: [] for count in self.counts
        }
        self._seen = 0
        self._cycled = 0
        self._transition = 0

    def start(self) -> None:
        """Begin exploring: the startup golden check has passed."""
        self._started = True

    @property
    def retired(self) -> bool:
        """Whether an observed drift has permanently shut the learning down."""
        return self._retired

    @property
    def adopted(self) -> bool:
        """Whether the learned counts verified themselves and may serve."""
        return self._table is not None and self._transition == 0

    def pick(self, tokens: int) -> int:
        """The count this request's forward should run with."""
        with self._lock:
            if not self._started:
                return self.base
            if self._table is None:
                count = self.counts[self._cycled % len(self.counts)]
                self._cycled += 1
                return count
            return self._table.get(_bucket(tokens), self.base)

    def record(self, tokens: int, forward_ms: float, exact: bool = True) -> None:
        """File one shadow pass's duration under the count it ran with.

        The served answer always comes from ``base``. A shadow pass under
        another count that fails to reproduce it bit for bit (``exact=False``)
        retires the resolver: one observed drift means this host's answers
        move across counts. Records after the table builds count down the
        transition that keeps the learned counts verifying.
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

    def record_base(self, tokens: int, forward_ms: float) -> None:
        """File a warm configured-count pass, timed beside the shadow passes.

        A model's first forward on fresh input pays cold caches; pairing the
        cold configured-count pass with warm shadow passes would credit the
        caches to the other counts. The warm pass only archives: it advances
        no window, and base's own samples sit in the round-robin like any
        other count's.
        """
        with self._lock:
            if self._started and self._table is None:
                self._samples[self.base].append((tokens, forward_ms))

    def _retire(self) -> None:
        """Give up adaptation: keep the configured count for the model's life."""
        self._retired = True
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
        self._table = table
        # A transition verifies counts that are new to execution: a table
        # that picked the configured count everywhere introduces nothing,
        # so there is nothing to verify and serving can start at once.
        new_counts = set(table.values()) - {self.base}
        self._transition = _TRANSITION if self.serving and new_counts else 0
        if table:
            edges = (
                "0-31",
                "32-63",
                "64-127",
                "128-255",
                "256-511",
                "512-1023",
                "1024+",
            )
            learned = ", ".join(
                f"{edges[bucket]}: {count} threads"
                for bucket, count in sorted(table.items())
            )
            # A table that picked only the configured count verifies nothing
            # new: level 2 serves it at once, with no transition to await.
            state = (
                "serving after transition"
                if self._transition
                else "serving" if self.serving else "report only"
            )
            log.info(
                "learned intra-op thread counts (%s): %s%s",
                state,
                learned,
                "" if self.serving else "; set --threads to apply them",
            )
        # A bucket without enough samples on any count keeps the configured
        # count (``pick`` falls back to ``base``); an empty table means the
        # traffic never repeated a size enough to rank any count.
