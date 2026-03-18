"""Random router: uniformly assigns each request to a random pool.

Useful as a baseline that demonstrates the cost of ignoring request length.
"""
from __future__ import annotations
import random
from typing import Dict, Optional
from .base import BaseRouter
from ..core.request import Request
from ..core.fleet import PoolConfig


class RandomRouter(BaseRouter):
    def __init__(self, pools: Dict[str, PoolConfig],
                 seed: int = 0, **kwargs):
        super().__init__(pools, **kwargs)
        self._rng = random.Random(seed)

    def route(self, req: Request) -> Optional[str]:
        return self._rng.choice(self.pool_ids)
