"""Max-speed profile: numerics-changing kernels on top of cross-request batching.

Allows kernels that are not bit-exact against the reference (non-exact fused
norms, tuned GEMM selection) and, where an engine supports it, a merged LoRA.
Its decision changes and accuracy are recorded per size in ``docs/records/``.
"""

from __future__ import annotations

from dataclasses import replace

from ..plugins.base import EngineOptions
from .batching import BatchingProfile


class MaxSpeedProfile(BatchingProfile):
    name = "max_speed"
    numerics = "approximate"
    description = "Approximate kernels and cross-request batching for the lowest latency and highest throughput."

    def engine_options(self, base: EngineOptions) -> EngineOptions:
        return replace(base, exact_kernels_only=False)
