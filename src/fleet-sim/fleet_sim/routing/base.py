"""Abstract base class for request routers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.fleet import PoolConfig
from ..core.request import Request


class BaseRouter(ABC):
    """Route incoming requests to a pool.

    Subclasses implement ``route(req) -> pool_id``.

    Parameters
    ----------
    pools : dict mapping pool_id → PoolConfig, in priority order
    """

    def __init__(self, pools: dict[str, PoolConfig], **kwargs):
        self.pools = pools
        self.pool_ids = list(pools.keys())

    @abstractmethod
    def route(self, req: Request) -> str | None:
        """Return the pool_id this request should be sent to, or None to drop."""
        ...

    def describe(self) -> str:
        return f"{self.__class__.__name__}(pools={self.pool_ids})"
