"""Compress-and-Route (C&R) router.

Implements the C&R routing mechanism:

  1. Requests with total_tokens ≤ B_short  → short pool (no change)
  2. Requests with B_short < total_tokens ≤ gamma * B_short  → BORDERLINE
       If safe_to_compress(req): greedy-compress to exactly B_short tokens → short pool
       Else: → long pool (no compression)
  3. Requests with total_tokens > gamma * B_short → long pool (too long to compress)

Compression safety is determined by the request's ``category`` field:
  "code"  → unsafe (structural semantics are not preserved by sentence extraction)
  "prose" | "rag" | "mixed" → always safe (p_c = 1.0)

The vLLM Semantic Router compressor is a greedy sentence-selection algorithm
(TextRank + TF-IDF + position + novelty scoring) that stops when the token
budget B_short - l_out is filled.  It always hits the budget for safe-category
prompts with >= min_keep=5 sentences — trivially satisfied for borderline
requests (>= B_short tokens at ~18 tok/sentence means hundreds of sentences).
Therefore p_c = 1.0 for safe categories; the ``p_compress`` parameter is
retained for backwards compatibility but defaults to 1.0.

Parameters
----------
B_short      : short-pool context threshold (tokens)
gamma        : compression bandwidth ratio (>= 1.0); defines the borderline band
               as (B_short, gamma * B_short]
p_compress   : probability a "safe" request is successfully compressed.
               Defaults to 1.0 (greedy compressor always hits budget for safe cats).
               Set < 1.0 only to model partial-failure scenarios.
safe_cats    : set of categories treated as compressible
               (default {"prose", "rag", "mixed"})
compress_lat : extra gateway latency added to compressed requests (s)
               (default 0.003 = 3ms; measured on modern CPU for TextRank extraction)
"""
from __future__ import annotations
import random
import math
from typing import Dict, Optional, Set
from .base import BaseRouter
from ..core.request import Request, RequestState
from ..core.fleet import PoolConfig


class CompressAndRouteRouter(BaseRouter):
    """C&R router modelling the vLLM Semantic Router greedy compressor.

    Expects exactly two pools with pool_ids "short" and "long" (or the
    first and last pool in sorted max_ctx order).
    """

    def __init__(
        self,
        pools: Dict[str, PoolConfig],
        B_short: int,
        gamma: float = 1.5,
        p_compress: float = 1.0,
        safe_cats: Optional[Set[str]] = None,
        compress_lat: float = 0.003,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(pools, **kwargs)
        self.B_short = B_short
        self.gamma = gamma
        self.p_compress = p_compress
        self.safe_cats = safe_cats or {"prose", "rag", "mixed"}
        self.compress_lat = compress_lat
        self._rng = random.Random(seed)

        sorted_pools = sorted(pools.values(), key=lambda p: p.max_ctx)
        self._short_id = sorted_pools[0].pool_id
        self._long_id  = sorted_pools[-1].pool_id

        # Counters for analysis
        self.n_short = 0
        self.n_long = 0
        self.n_compressed = 0
        self.n_borderline_unsafe = 0

    def route(self, req: Request) -> Optional[str]:
        total = req.l_in + req.l_out

        if total <= self.B_short:
            # Definitely short
            self.n_short += 1
            return self._short_id

        elif total <= self.gamma * self.B_short:
            # Borderline: attempt greedy compression for safe categories.
            # p_compress defaults to 1.0 (greedy always succeeds); set < 1.0
            # only to model partial-failure / code-heavy workload scenarios.
            if (req.category in self.safe_cats
                    and (self.p_compress >= 1.0
                         or self._rng.random() < self.p_compress)):
                # Greedy compressor fills exactly B_short - l_out input tokens.
                target_l_in = self.B_short - req.l_out
                if target_l_in <= 0:
                    # Degenerate: output alone fills slot → long pool
                    self.n_borderline_unsafe += 1
                    self.n_long += 1
                    return self._long_id
                req.orig_l_in = req.l_in
                req.l_in = target_l_in
                req.compressed = True
                self.n_compressed += 1
                self.n_short += 1
                return self._short_id
            else:
                # Unsafe category or p_compress < 1.0 failure draw
                self.n_borderline_unsafe += 1
                self.n_long += 1
                return self._long_id

        else:
            # Definitely long
            self.n_long += 1
            return self._long_id

    def alpha(self) -> float:
        """Fraction of routed requests that went to short pool."""
        total = self.n_short + self.n_long
        return self.n_short / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "B_short": self.B_short,
            "gamma": self.gamma,
            "n_short": self.n_short,
            "n_long": self.n_long,
            "n_compressed": self.n_compressed,
            "n_borderline_unsafe": self.n_borderline_unsafe,
            "alpha": round(self.alpha(), 4),
        }
