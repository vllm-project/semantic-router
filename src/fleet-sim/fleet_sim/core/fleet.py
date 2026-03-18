"""Fleet: top-level multi-pool coordinator.

FleetConfig defines the static configuration; Fleet executes the simulation
event loop, dispatching requests through the router and advancing instance
time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..gpu_profiles.profiles import GpuProfile
from .pool import Pool
from .request import Request, RequestState

# ── Configuration dataclasses ─────────────────────────────────────────────────


@dataclass
class PoolConfig:
    """Static specification for one pool in the fleet.

    Parameters
    ----------
    pool_id      : unique name (e.g. "short", "long")
    gpu          : GpuProfile for GPUs in this pool
    n_gpus       : number of GPU instances
    max_ctx      : maximum context length this pool handles
    chunk_mode   : "independent" (default, matches M/G/c model) or "shared"
    lb_strategy  : load-balancing ("least_queue", "round_robin", "least_loaded")
    """

    pool_id: str
    gpu: GpuProfile
    n_gpus: int
    max_ctx: int
    chunk_mode: str = "independent"
    lb_strategy: str = "least_queue"


@dataclass
class FleetConfig:
    """Full fleet configuration.

    Parameters
    ----------
    pools        : ordered list of pool specs (router sees them in this order)
    router_type  : routing algorithm class name (registered in routing module)
    router_kwargs: extra kwargs passed to router constructor
    """

    pools: list[PoolConfig]
    router_type: str = "LengthRouter"
    router_kwargs: dict = field(default_factory=dict)

    def total_gpus(self) -> int:
        return sum(p.n_gpus for p in self.pools)

    def total_cost_per_hr(self) -> float:
        return sum(p.gpu.cost_per_hr * p.n_gpus for p in self.pools)

    def annualised_cost_usd(self) -> float:
        return self.total_cost_per_hr() * 8760


# ── Fleet simulation engine ───────────────────────────────────────────────────


class Fleet:
    """Event-driven fleet simulation.

    Usage
    -----
    ::

        cfg = FleetConfig(pools=[PoolConfig(...)], router_type="LengthRouter",
                          router_kwargs={"threshold": 4096})
        fleet = Fleet(cfg)
        fleet.run(requests)   # list of (arrival_time, Request) tuples
        results = fleet.collect_metrics()
    """

    def __init__(self, config: FleetConfig):
        self.config = config
        self._pools: dict[str, Pool] = {}
        self._completed: list[Request] = []
        self._router = None  # set in run()

    def _build(self) -> None:
        self._pools = {}
        self._completed = []

        def on_ttft(req: Request) -> None:
            pass  # could hook metrics here

        def on_complete(req: Request) -> None:
            self._completed.append(req)

        for pc in self.config.pools:
            self._pools[pc.pool_id] = Pool(
                pool_id=pc.pool_id,
                gpu=pc.gpu,
                n_gpus=pc.n_gpus,
                max_ctx=pc.max_ctx,
                chunk_mode=pc.chunk_mode,
                lb_strategy=pc.lb_strategy,
                on_complete=on_complete,
            )

        # Build router
        from .. import routing as _routing

        router_cls = getattr(_routing, self.config.router_type)
        self._router = router_cls(
            pools={pc.pool_id: pc for pc in self.config.pools},
            **self.config.router_kwargs,
        )
        # Inject live pool references for routers that need runtime state
        if hasattr(self._router, "set_pools"):
            self._router.set_pools(self._pools)

    def run(
        self,
        arrivals: list[tuple[float, Request]],
        verbose: bool = False,
    ) -> FleetSimResult:
        """Simulate the fleet processing a list of requests.

        Parameters
        ----------
        arrivals : list of (arrival_time_s, Request) sorted by time
        verbose  : print progress every 10k requests
        """
        self._build()

        # Sort by arrival time (defensive)
        arrivals = sorted(arrivals, key=lambda x: x[0])

        now = 0.0
        idx = 0  # next arrival index
        n_total = len(arrivals)

        while idx < n_total or any(
            p.total_completed()
            + sum(i.active_count + i.queue_depth for i in p.instances)
            > 0
            for p in self._pools.values()
        ):
            # Determine next event time
            next_arrival = arrivals[idx][0] if idx < n_total else float("inf")
            next_completion = min(
                (p.next_event_time() for p in self._pools.values()),
                default=float("inf"),
            )

            step_to = min(next_arrival, next_completion)
            if step_to == float("inf"):
                break

            # Advance all pools to step_to
            for pool in self._pools.values():
                pool.advance_to(step_to)

            now = step_to

            # Process all arrivals at or before now
            while idx < n_total and arrivals[idx][0] <= now:
                _, req = arrivals[idx]
                req.arrival_time = arrivals[idx][0]
                pool_id = self._router.route(req)
                if pool_id is not None and pool_id in self._pools:
                    self._pools[pool_id].route(req)
                else:
                    req.state = RequestState.DONE  # dropped
                idx += 1

                if verbose and idx % 10_000 == 0:
                    pct = idx / n_total * 100
                    done = len(self._completed)
                    print(
                        f"  [{pct:5.1f}%]  arrivals={idx:,}  "
                        f"completed={done:,}  t={now:.2f}s"
                    )

        # Final drain
        drain_time = now + 600.0  # give up to 10 min to flush
        for pool in self._pools.values():
            pool.advance_to(drain_time)

        return FleetSimResult(
            config=self.config,
            completed=list(self._completed),
            pools=self._pools,
        )

    def collect_metrics(self) -> FleetSimResult:
        """Return result object (valid after run())."""
        return FleetSimResult(
            config=self.config,
            completed=list(self._completed),
            pools=self._pools,
        )


# ── Result object ─────────────────────────────────────────────────────────────


class FleetSimResult:
    """Aggregated metrics from a completed fleet simulation."""

    def __init__(
        self, config: FleetConfig, completed: list[Request], pools: dict[str, Pool]
    ):
        self.config = config
        self.completed = completed
        self.pools = pools

    # ── fleet-level ───────────────────────────────────────────────────────────

    def percentile(self, metric: str, p: float, pool_id: str | None = None) -> float:
        """Compute a percentile of a metric across completed requests.

        metric : "ttft" | "e2e" | "queue_wait"
        p      : percentile in [0, 100]
        pool_id: filter to a specific pool (None = all)
        """

        reqs = [r for r in self.completed if pool_id is None or r.pool_id == pool_id]
        vals = [getattr(r, metric) for r in reqs if getattr(r, metric) is not None]
        if not vals:
            return float("nan")
        vals.sort()
        idx = max(0, int(len(vals) * p / 100) - 1)
        return vals[idx]

    def p99_ttft_ms(self, pool_id: str | None = None) -> float:
        return self.percentile("ttft", 99, pool_id) * 1000

    def p50_ttft_ms(self, pool_id: str | None = None) -> float:
        return self.percentile("ttft", 50, pool_id) * 1000

    def p99_queue_wait_ms(self, pool_id: str | None = None) -> float:
        return self.percentile("queue_wait", 99, pool_id) * 1000

    def throughput(self) -> float:
        """Completed requests per second."""
        times = [r.end_time for r in self.completed if r.end_time]
        if not times:
            return 0.0
        return len(times) / (max(times) - min(r.arrival_time for r in self.completed))

    def total_gpus(self) -> int:
        return sum(p.n_gpus for p in self.pools.values())

    def cost_per_hr(self) -> float:
        return sum(p.cost_per_hr() for p in self.pools.values())

    def annualised_cost_usd(self) -> float:
        return self.cost_per_hr() * 8760

    def mean_utilisation(self, pool_id: str | None = None) -> float:
        ps = [self.pools[pool_id]] if pool_id else list(self.pools.values())
        utils = [p.mean_utilisation() for p in ps]
        return sum(utils) / len(utils) if utils else 0.0

    def slo_compliance(self, t_slo_ms: float, pool_id: str | None = None) -> float:
        """Fraction of requests with TTFT ≤ t_slo_ms."""
        reqs = [r for r in self.completed if pool_id is None or r.pool_id == pool_id]
        ttfts = [r.ttft for r in reqs if r.ttft is not None]
        if not ttfts:
            return float("nan")
        return sum(1 for t in ttfts if t * 1000 <= t_slo_ms) / len(ttfts)

    def summary(self, t_slo_ms: float = 500.0) -> dict:
        """Return a flat dict of key fleet-level metrics."""
        result: dict = {
            "total_gpus": self.total_gpus(),
            "total_completed": len(self.completed),
            "cost_per_hr_usd": round(self.cost_per_hr(), 4),
            "annualised_cost_kusd": round(self.annualised_cost_usd() / 1000, 1),
            "fleet_p99_ttft_ms": round(self.p99_ttft_ms(), 1),
            "fleet_p50_ttft_ms": round(self.p50_ttft_ms(), 1),
            "fleet_p99_qwait_ms": round(self.p99_queue_wait_ms(), 1),
            "fleet_slo_compliance": round(self.slo_compliance(t_slo_ms), 4),
            "mean_utilisation": round(self.mean_utilisation(), 4),
        }
        for pool_id, pool in self.pools.items():
            result[f"{pool_id}_n_gpus"] = pool.n_gpus
            result[f"{pool_id}_p99_ttft_ms"] = round(self.p99_ttft_ms(pool_id), 1)
            result[f"{pool_id}_p99_qwait_ms"] = round(
                self.p99_queue_wait_ms(pool_id), 1
            )
            result[f"{pool_id}_util"] = round(pool.mean_utilisation(), 4)
            result[f"{pool_id}_slo"] = round(self.slo_compliance(t_slo_ms, pool_id), 4)
        return result

    def print_summary(self, t_slo_ms: float = 500.0) -> None:
        d = self.summary(t_slo_ms)
        print(f"\n  Fleet summary  (SLO = {t_slo_ms:.0f}ms P99 TTFT)")
        print(f"  {'Total GPUs':30s}: {d['total_gpus']}")
        print(f"  {'Annualised cost':30s}: ${d['annualised_cost_kusd']:.1f}K/yr")
        print(f"  {'Fleet P99 TTFT':30s}: {d['fleet_p99_ttft_ms']:.1f}ms")
        print(f"  {'Fleet P50 TTFT':30s}: {d['fleet_p50_ttft_ms']:.1f}ms")
        print(f"  {'SLO compliance':30s}: {d['fleet_slo_compliance']*100:.2f}%")
        print(f"  {'Mean utilisation':30s}: {d['mean_utilisation']*100:.1f}%")
        for pool_id in self.pools:
            print(f"\n  Pool '{pool_id}':")
            print(f"    {'GPUs':26s}: {d[f'{pool_id}_n_gpus']}")
            print(f"    {'P99 TTFT':26s}: {d[f'{pool_id}_p99_ttft_ms']:.1f}ms")
            print(f"    {'P99 queue wait':26s}: {d[f'{pool_id}_p99_qwait_ms']:.1f}ms")
            print(f"    {'Utilisation':26s}: {d[f'{pool_id}_util']*100:.1f}%")
            print(f"    {'SLO compliance':26s}: {d[f'{pool_id}_slo']*100:.2f}%")
