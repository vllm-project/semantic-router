"""Semantic-classification-based router.

Routes each request to a pool based on the *semantic content* of the prompt
rather than token length or an explicit model tag.  A user-supplied
classifier function examines the request and returns a pool_id.

Typical use case
----------------
A semantic router (e.g., a lightweight classifier, an embedding-based
nearest-neighbour lookup, or a rule-based intent classifier) sits at the
gateway and decides which model/pool is most appropriate for each request:

  - Simple factual queries → small model pool (fast, cheap)
  - Complex reasoning tasks → large model pool (accurate)
  - Code generation → code-specialist pool
  - Anything else → default pool

Integration with vLLM semantic routing
---------------------------------------
If you run vLLM with a semantic router that pre-classifies requests at the
gateway (assigning a ``model_id`` or ``route_label`` per request), you can
replay those routing decisions in the simulator by setting
``Request.model_id`` from the trace and using ``ModelRouter``.

For simulation without a live classifier, supply a ``classify_fn`` that
maps a ``Request`` to a ``pool_id``.  This lets you benchmark routing
policies offline before deploying them in production.

Example
-------
::

    from fleet_sim.routing.semantic_router import SemanticRouter

    def my_classifier(req):
        # Simple heuristic: short requests to llama8b, long to llama70b
        if req.l_total < 2048:
            return "llama8b"
        return "llama70b"

    router = SemanticRouter(pools, classify_fn=my_classifier)

    # Or embed into FleetConfig:
    fc = FleetConfig(
        pools=[PoolConfig("llama70b", A100_80GB, 20, 8192),
               PoolConfig("llama8b",  A10G,      8, 4096)],
        router_type="SemanticRouter",
        router_kwargs={"classify_fn": my_classifier},
    )
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fleet_sim.routing.base import BaseRouter
from fleet_sim.core.request import Request


class SemanticRouter(BaseRouter):
    """Route requests using an arbitrary classifier function.

    Parameters
    ----------
    pools        : mapping of pool_id → PoolConfig
    classify_fn  : callable ``(Request) -> Optional[str]`` that returns a
                   pool_id.  If it returns ``None`` or an unknown pool_id,
                   the request is sent to ``default_pool``.
    default_pool : pool_id to use as fallback.  Defaults to first pool.

    Notes
    -----
    ``classify_fn`` is called once per request at routing time.  Keep it
    lightweight (< 1 ms) to avoid distorting the simulated queue dynamics.
    For classifiers that are themselves expensive (e.g., embedding lookup),
    model their latency as part of the TTFT budget by subtracting from the
    SLO before calling ``FleetOptimizer``.
    """

    def __init__(self,
                 pools: Dict[str, Any],
                 classify_fn: Optional[Callable[[Request], Optional[str]]] = None,
                 default_pool: Optional[str] = None,
                 **kwargs):
        super().__init__(pools, **kwargs)
        if classify_fn is None:
            # Default: fall through to model_id if set, else first pool
            def classify_fn(req: Request) -> Optional[str]:
                return req.model_id
        self._classify = classify_fn
        self._default = default_pool or self.pool_ids[0]

    def route(self, req: Request) -> Optional[str]:
        pool_id = self._classify(req)
        if pool_id and pool_id in self.pools:
            return pool_id
        return self._default
