"""Length-based router: routes requests to pools by token count.

A request with l_in + l_out ≤ threshold → short pool.
Otherwise → long pool.

If multiple pools are configured, the router applies thresholds in order:
requests go to the first pool whose max_ctx ≥ request length (shortest-fit).
This is the standard pool-routing algorithm: route each request to the
smallest pool whose context capacity can accommodate it.
"""

from __future__ import annotations

from ..core.fleet import PoolConfig
from ..core.request import Request
from .base import BaseRouter


class LengthRouter(BaseRouter):
    """Route by total token length to the smallest fitting pool.

    Constructor kwargs
    ------------------
    threshold : int, optional
        Explicit short/long boundary (tokens).  If omitted, uses the
        first pool's max_ctx as the threshold.
    """

    def __init__(
        self, pools: dict[str, PoolConfig], threshold: int | None = None, **kwargs
    ):
        super().__init__(pools, **kwargs)
        # Sort pools by max_ctx ascending so shortest-fit comes first
        self._sorted = sorted(pools.values(), key=lambda p: p.max_ctx)
        self.threshold = threshold

    def route(self, req: Request) -> str | None:
        total = req.l_in + req.l_out

        if self.threshold is not None:
            # Binary split: short or long
            for pc in self._sorted:
                if total <= self.threshold:
                    return pc.pool_id  # first (shortest) pool
            # Falls through to last pool
            return self._sorted[-1].pool_id

        # Shortest-fit: route to smallest pool that can handle this request
        for pc in self._sorted:
            if total <= pc.max_ctx:
                return pc.pool_id

        # Request exceeds all pool capacities: send to largest pool
        return self._sorted[-1].pool_id
