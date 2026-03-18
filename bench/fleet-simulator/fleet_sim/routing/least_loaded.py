"""Least-loaded router: sends all requests to the pool with the lowest
current utilisation (active + queued requests as a fraction of capacity).

Useful as a baseline for homogeneous fleets or for benchmarking.
"""
from __future__ import annotations
from typing import Dict, Optional
from .base import BaseRouter
from ..core.request import Request
from ..core.fleet import PoolConfig


class LeastLoadedRouter(BaseRouter):
    """Route every request to the least-loaded pool (by queue depth fraction).

    Requires access to pool objects; set via ``set_pools(pools_dict)`` after
    the Fleet is constructed.
    """

    def __init__(self, pools: Dict[str, PoolConfig], **kwargs):
        super().__init__(pools, **kwargs)
        self._live_pools = None  # injected by Fleet after pool construction

    def set_pools(self, live_pools) -> None:
        self._live_pools = live_pools

    def route(self, req: Request) -> Optional[str]:
        if self._live_pools is None:
            # Fall back to first pool if live state not available
            return self.pool_ids[0]
        # Pick pool with lowest (queued + active) / capacity ratio
        best_id, best_load = None, float("inf")
        for pid, pool in self._live_pools.items():
            total = sum(i.active_count + i.queue_depth
                        for i in pool.instances)
            cap = sum(i.n_slots for i in pool.instances)
            load = total / max(1, cap)
            if load < best_load:
                best_load = load
                best_id = pid
        return best_id
