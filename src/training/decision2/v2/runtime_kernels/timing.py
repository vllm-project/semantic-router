"""GPU timing helpers shared by the kernel benchmarks.

``time_call`` is the per-kernel number the reports use: after warm-up, every
call is bracketed by a device synchronisation and a pair of HIP events, and
the median over at least 100 calls is kept. ``time_graph`` replays the same
call captured ``reps`` times in one HIP graph, which removes the host launch
gaps a multi-kernel reference pays in eager mode (the cost the study track's
HIP graphs remove); it is the fair comparison for kernel-count reductions.
``rotating`` cycles through copies of the operands so that weight-bound GEMMs
are not served from the 256 MB Infinity Cache, which a real forward (a new
weight matrix for every layer) never hits.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

MI325X = {
    "hbm_bytes_per_s": 6.0e12,
    "bf16_dense_flops": 1307.4e12,
    "fp32_vector_flops": 163.4e12,
    "cus": 304,
}


def time_call(
    fn: Callable[[], Any], torch: Any, warmup: int = 20, iters: int = 200
) -> dict[str, float]:
    """Median / p10 / p90 microseconds of ``fn`` with a synchronisation around every call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        torch.cuda.synchronize()
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    samples = sorted(1000.0 * s.elapsed_time(e) for s, e in zip(start, end))
    return summary(samples)


def time_graph(
    fn: Callable[[], Any], torch: Any, reps: int = 20, replays: int = 100
) -> dict[str, float] | None:
    """Per-call microseconds of ``fn`` captured ``reps`` times in one graph (None if capture fails)."""
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
    except (
        Exception
    ):  # noqa: BLE001 - capture is best effort (e.g. a host sync inside fn)
        torch.cuda.synchronize()
        return None
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(replays):
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(1000.0 * start.elapsed_time(end) / reps)
    del graph
    return summary(sorted(samples))


def summary(samples: list[float]) -> dict[str, float]:
    n = len(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": samples[max(0, int(0.1 * n) - 1)],
        "p90_us": samples[min(n - 1, int(0.9 * n))],
        "n": n,
    }


def rotating(make: Callable[[int], Any], copies: int) -> Callable[[], Any]:
    """A zero-argument callable returning the next of ``copies`` operand sets, round robin."""
    sets = [make(i) for i in range(copies)]
    state = {"i": 0}

    def next_set() -> Any:
        state["i"] = (state["i"] + 1) % copies
        return sets[state["i"]]

    return next_set


def bandwidth_fraction(bytes_moved: float, microseconds: float) -> float:
    return bytes_moved / (microseconds * 1e-6) / MI325X["hbm_bytes_per_s"]


def flop_fraction(flops: float, microseconds: float) -> float:
    return flops / (microseconds * 1e-6) / MI325X["bf16_dense_flops"]
