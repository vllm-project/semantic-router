"""Pool of homogeneous GPU instances with load balancing.

A Pool groups N identical GPU instances sharing a common:
  - GPU hardware profile
  - Maximum context length
  - Routing policy (which instance to assign new requests to)

Load balancing strategies
-------------------------
  least_queue   : route to instance with shortest queue (default)
  round_robin   : cycle through instances
  least_loaded  : route to instance with lowest (active + queued) count
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional

from .instance import Instance
from .request import Request, RequestState
from ..gpu_profiles.profiles import GpuProfile


class Pool:
    """A homogeneous group of GPU instances.

    Parameters
    ----------
    pool_id      : unique identifier for this pool
    gpu          : GPU type for all instances in this pool
    n_gpus       : number of GPU instances
    max_ctx      : maximum context length served by this pool (tokens)
    chunk_mode   : prefill scheduling for instances
    lb_strategy  : load balancing ("least_queue", "round_robin", "least_loaded")
    max_queue    : per-instance queue depth cap
    """

    def __init__(
        self,
        pool_id: str,
        gpu: GpuProfile,
        n_gpus: int,
        max_ctx: int,
        chunk_mode: str = "independent",
        lb_strategy: str = "least_queue",
        max_queue: int = 512,
        on_ttft: Optional[Callable[[Request], None]] = None,
        on_complete: Optional[Callable[[Request], None]] = None,
    ):
        self.pool_id = pool_id
        self.gpu = gpu
        self.n_gpus = n_gpus
        self.max_ctx = max_ctx
        self.lb_strategy = lb_strategy

        self.instances: list[Instance] = [
            Instance(
                instance_id=i,
                pool_id=pool_id,
                gpu=gpu,
                max_ctx=max_ctx,
                chunk_mode=chunk_mode,
                max_queue=max_queue,
                on_ttft=on_ttft,
                on_complete=on_complete,
            )
            for i in range(n_gpus)
        ]
        self._rr_counter = itertools.cycle(range(n_gpus))
        self.rejected: int = 0   # requests rejected due to queue overflow

    # ── routing ───────────────────────────────────────────────────────────────

    def route(self, req: Request) -> bool:
        """Route req to the best instance.  Returns False if all queues full."""
        inst = self._pick_instance()
        if inst is None:
            self.rejected += 1
            return False
        return inst.accept(req)

    def _pick_instance(self) -> Optional[Instance]:
        if not self.instances:
            return None

        if self.lb_strategy == "round_robin":
            for _ in range(self.n_gpus):
                inst = self.instances[next(self._rr_counter)]
                if not inst.is_full or inst.queue_depth < inst.max_queue:
                    return inst
            return None

        if self.lb_strategy == "least_queue":
            return min(self.instances, key=lambda i: i.queue_depth)

        if self.lb_strategy == "least_loaded":
            return min(self.instances,
                       key=lambda i: i.active_count + i.queue_depth)

        return self.instances[0]

    # ── simulation advancement ────────────────────────────────────────────────

    def advance_to(self, target_time: float) -> list[Request]:
        """Advance all instances to target_time; return completed requests."""
        completed: list[Request] = []
        for inst in self.instances:
            completed.extend(inst.advance_to(target_time))
        return completed

    def next_event_time(self) -> float:
        """Earliest event time across all instances."""
        return min((i.next_event_time() for i in self.instances),
                   default=float("inf"))

    # ── metrics ───────────────────────────────────────────────────────────────

    def mean_utilisation(self) -> float:
        """Mean GPU utilisation across instances."""
        utils = [i.utilisation() for i in self.instances]
        return sum(utils) / len(utils) if utils else 0.0

    def total_completed(self) -> int:
        return sum(i.total_requests for i in self.instances)

    def cost_per_hr(self) -> float:
        return self.gpu.cost_per_hr * self.n_gpus

    def annualised_cost_usd(self) -> float:
        return self.cost_per_hr() * 8760
