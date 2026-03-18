"""Spillover router: length-based primary routing with load-aware overflow.

Short requests (≤ threshold) go to the short pool by default.
When the short pool utilisation exceeds ``spill_threshold`` (0–1), they
overflow to the long pool — which also serves its own long requests.
Long requests (> threshold) always go to the long pool.

This models the common deployment pattern where the long pool acts as a
general-purpose fallback while the short pool is optimised for low-latency
serving of the majority traffic.
"""
from __future__ import annotations
from typing import Dict, Optional
from .base import BaseRouter
from ..core.request import Request
from ..core.fleet import PoolConfig


class SpilloverRouter(BaseRouter):
    """Length-based routing with overflow from short → long pool.

    Constructor kwargs
    ------------------
    threshold      : int   short/long boundary (tokens)
    spill_threshold: float utilisation fraction above which short requests
                           spill to the long pool (default 0.85)
    """

    def __init__(self, pools: Dict[str, PoolConfig],
                 threshold: Optional[int] = None,
                 spill_threshold: float = 0.85, **kwargs):
        super().__init__(pools, **kwargs)
        self._sorted = sorted(pools.values(), key=lambda p: p.max_ctx)
        self.threshold = threshold or self._sorted[0].max_ctx
        self.spill_threshold = spill_threshold
        self._live_pools = None   # injected by Fleet

    def set_pools(self, live_pools) -> None:
        self._live_pools = live_pools

    def _pool_pressure(self, pool_id: str) -> float:
        """Pressure = (active + queued) / n_gpus.

        Measures mean requests per GPU — a value > spill_threshold means
        the pool is filling up.  This is robust to large n_slots values
        that make raw slot-utilisation near zero even under heavy load.
        """
        if self._live_pools is None:
            return 0.0
        pool = self._live_pools.get(pool_id)
        if pool is None:
            return 0.0
        total_reqs = sum(i.active_count + i.queue_depth for i in pool.instances)
        n_gpus     = len(pool.instances)
        return total_reqs / max(1, n_gpus)

    def route(self, req: Request) -> Optional[str]:
        total = req.l_in + req.l_out
        short_id = self._sorted[0].pool_id
        long_id  = self._sorted[-1].pool_id

        if total > self.threshold:
            # Long request — only the long pool has enough context capacity
            return long_id

        # Short request: prefer short pool, spill to long if overloaded.
        # spill_threshold is expressed as mean requests per GPU in the short pool.
        if self._pool_pressure(short_id) >= self.spill_threshold:
            return long_id
        return short_id
