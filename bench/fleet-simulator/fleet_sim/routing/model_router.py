"""Model-based router: directs each request to the pool that serves its model.

Use this when the fleet has multiple pools running *different models* (or
different serving configurations of the same model) and routing is determined
by which model a request needs, not by prompt length.

Example fleet layout
--------------------
pools:
  llama70b  – 20× A100-80GB running Llama-3-70B
  llama8b   – 8×  A10G      running Llama-3-8B
  codellama – 4×  A100-80GB running CodeLlama-34B

Each request carries ``Request.model_id = "llama70b"`` (or whichever model it
targets), and ModelRouter dispatches accordingly.

Fallback behaviour
------------------
If a request has ``model_id=None`` or names an unknown pool, ModelRouter falls
back to ``default_pool`` (first pool in the fleet if not specified).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fleet_sim.routing.base import BaseRouter
from fleet_sim.core.request import Request


class ModelRouter(BaseRouter):
    """Route each request to the pool matching ``request.model_id``.

    Parameters
    ----------
    pools        : mapping of pool_id → PoolConfig (standard fleet dict)
    default_pool : pool_id to use when a request has no model_id or an
                   unknown one.  Defaults to the first pool in the fleet.
    """

    def __init__(self, pools: Dict[str, Any],
                 default_pool: Optional[str] = None, **kwargs):
        super().__init__(pools, **kwargs)
        if default_pool is not None and default_pool not in pools:
            raise ValueError(
                f"default_pool '{default_pool}' is not in the fleet "
                f"(available: {list(pools.keys())})"
            )
        self._default = default_pool or self.pool_ids[0]

    def route(self, req: Request) -> Optional[str]:
        if req.model_id and req.model_id in self.pools:
            return req.model_id
        return self._default
