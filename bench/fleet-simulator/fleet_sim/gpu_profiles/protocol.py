"""GpuProfile Protocol — the only interface the simulation engine needs.

Any object that implements these four members can be used as a GPU profile
throughout the fleet simulator. Both ManualProfile (hand-calibrated constants)
and ComputedProfile (derived from HardwareSpec + ModelSpec) satisfy this
protocol.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class GpuProfile(Protocol):
    """Structural interface for a GPU performance profile.

    The DES engine (Instance, Pool) and the M/G/c optimizer (FleetOptimizer)
    only ever call these methods. Implementation details — whether constants
    are hand-calibrated or derived from first principles — are irrelevant to
    the simulation core.
    """

    name: str
    cost_per_hr: float

    def iter_latency(self, n_active: int,
                     mean_seq_len: Optional[float] = None) -> float:
        """Wall-clock time for one decode iteration (seconds).

        Parameters
        ----------
        n_active     : number of sequences currently in-flight on this GPU
        mean_seq_len : mean total token length (prompt + output) of the active
                       sequences.  H is scaled by mean_seq_len / calibration_ctx
                       so that attention cost reflects actual sequence lengths.
                       Defaults to calibration_ctx (no scaling) when None.
        """
        ...

    def prefill_iter_latency(self, chunk_tokens: int,
                              kv_history_tokens: float,
                              n_active: int,
                              mean_seq_len: Optional[float] = None) -> float:
        """Wall-clock time for one prefill-chunk iteration (seconds).

        Prefill attention is potentially compute-bound (FlashAttention FLOPs
        scale as chunk × kv_history, which can exceed the memory-bandwidth
        ceiling for large chunks).  Implementations should return
        ``max(compute_bound, memory_bound)`` to correctly handle both regimes.

        ManualProfile falls back to ``iter_latency`` (no compute spec available).
        ComputedProfile applies the roofline check using ``hw.fp16_tc_flops``.

        Parameters
        ----------
        chunk_tokens      : prefill tokens processed in this chunk
        kv_history_tokens : tokens already in KV cache when this chunk runs
                            (average over all active sequences)
        n_active          : total in-flight sequences (prefill + decode)
        mean_seq_len      : mean sequence length for H scaling (same as
                            ``iter_latency``)
        """
        ...

    def n_slots(self, max_ctx: int) -> int:
        """Maximum concurrent in-flight sequences for given context window.

        Parameters
        ----------
        max_ctx : maximum token length a request in this pool can have
        """
        ...

    def service_time(self, l_in: int, l_out: int, max_ctx: int) -> float:
        """Total service time (seconds) for one request.

        Parameters
        ----------
        l_in    : prompt length (tokens)
        l_out   : output length (tokens)
        max_ctx : pool's maximum context length
        """
        ...

    def throughput(self, max_ctx: int, mean_l_in: float,
                   mean_l_out: float) -> float:
        """Steady-state request throughput (req/s) at full concurrency.

        Parameters
        ----------
        max_ctx    : pool's maximum context length
        mean_l_in  : mean prompt length over the workload distribution
        mean_l_out : mean output length over the workload distribution
        """
        ...
