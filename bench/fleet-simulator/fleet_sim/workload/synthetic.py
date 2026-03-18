"""Synthetic workload generators.

CdfWorkload  : generate requests by sampling from an empirical CDF
               (compatible with the JSON CDFs produced by preprocess.py)
PoissonWorkload : Poisson arrival process wrapping any length distribution
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..core.request import Request


class CdfWorkload:
    """Sample request lengths from an empirical CDF.

    Supports a JSON file containing either:
        {"cdf": [[token_threshold, cumulative_frac], ...]}
    or a plain list of [threshold, frac] pairs.

    Parameters
    ----------
    cdf_source : path to JSON file, or list of (threshold, frac) pairs
    l_out_frac : output tokens as fraction of total (default 0.20)
    l_in_frac  : input tokens as fraction of total (default 0.80)
    category_mix : dict mapping category → fraction (must sum to 1.0)
                   default {"prose": 0.60, "code": 0.25, "rag": 0.15}
    seed       : RNG seed
    """

    def __init__(
        self,
        cdf_source: Union[str, Path, list],
        l_out_frac: float = 0.20,
        l_in_frac: float = 0.80,
        category_mix: Optional[dict] = None,
        seed: int = 42,
    ):
        if isinstance(cdf_source, (str, Path)):
            raw = json.load(open(cdf_source))
            cdf = raw["cdf"] if isinstance(raw, dict) else raw
        else:
            cdf = cdf_source
        self._cdf: list[tuple[int, float]] = [
            (int(t), float(f)) for t, f in cdf
        ]
        self.l_out_frac = l_out_frac
        self.l_in_frac = l_in_frac
        self.category_mix = category_mix or {
            "prose": 0.60, "code": 0.25, "rag": 0.15,
        }
        self._rng = random.Random(seed)
        self._categories = list(self.category_mix.keys())
        self._cat_weights = [self.category_mix[c] for c in self._categories]

    def sample_length(self) -> int:
        """Draw one total token length from the empirical CDF."""
        u = self._rng.random()
        prev_thresh = 1
        for thresh, frac in self._cdf:
            if u <= frac:
                lo = max(1, prev_thresh)
                hi = max(lo, thresh)
                return self._rng.randint(lo, hi)
            prev_thresh = thresh
        return self._cdf[-1][0]

    def sample_request(self, req_id: int, arrival: float) -> Request:
        total = self.sample_length()
        l_in  = max(1, int(total * self.l_in_frac))
        l_out = max(1, total - l_in)
        cat = self._rng.choices(self._categories, self._cat_weights)[0]
        return Request(
            req_id=req_id,
            arrival_time=arrival,
            l_in=l_in,
            l_out=l_out,
            category=cat,
        )

    def p_quantile(self, p: float) -> int:
        """Return the p-th percentile token length (0 < p < 1)."""
        for thresh, frac in self._cdf:
            if frac >= p:
                return thresh
        return self._cdf[-1][0]


class PoissonWorkload:
    """Generate a Poisson arrival stream of requests.

    Parameters
    ----------
    lam        : arrival rate (requests per second)
    length_gen : CdfWorkload or any object with .sample_request(id, t)
    n_requests : number of requests to generate
    seed       : RNG seed for arrival times
    warm_up    : fraction of requests to treat as warm-up (excluded from metrics)
    """

    def __init__(
        self,
        lam: float,
        length_gen: CdfWorkload,
        n_requests: int = 50_000,
        seed: int = 42,
        warm_up: float = 0.10,
    ):
        self.lam = lam
        self.length_gen = length_gen
        self.n_requests = n_requests
        self.warm_up = warm_up
        self._rng = random.Random(seed)

    def generate(self) -> List[Tuple[float, Request]]:
        """Return list of (arrival_time, Request) sorted by arrival time."""
        arrivals: list[tuple[float, Request]] = []
        t = 0.0
        for i in range(self.n_requests):
            t += self._rng.expovariate(self.lam)
            req = self.length_gen.sample_request(req_id=i, arrival=t)
            req.arrival_time = t
            arrivals.append((t, req))
        return arrivals

    def warm_up_index(self) -> int:
        """Index of the first non-warm-up request."""
        return int(self.n_requests * self.warm_up)
