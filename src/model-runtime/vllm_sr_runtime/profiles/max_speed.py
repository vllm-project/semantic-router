"""Max-speed profile: shared-context trees and cross-request batching together, plus approximate kernels.

A multi-question request whose shared state pays off (the shared_context
profile's policy) runs as one shared-context tree; every other request's
questions are coalesced with those of concurrent requests (the batching
profile). Kernels that are not bit-exact against the reference may be
selected. Answers can differ from the exact path by rounding; the decision
changes and accuracy of each part are recorded per size in ``docs/records/``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..plugins.base import Batch, EngineOptions, Job, LoadedModel
from .batching import DEFAULT_MAX_BATCH_TOKENS, BatchingProfile
from .shared_context import SharedContextProfile


class MaxSpeedProfile(BatchingProfile):
    name = "max_speed"
    numerics = "approximate"
    description = (
        "Shared-context trees for multi-question requests, cross-request batching for the rest and approximate "
        "kernels: the lowest latency and highest throughput."
    )

    def __init__(self, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS):
        super().__init__(max_batch_tokens=max_batch_tokens)
        self.shared = SharedContextProfile()
        self.trees = False

    def engine_options(self, base: EngineOptions) -> EngineOptions:
        return replace(base, exact_kernels_only=False, reduced_precision=True)

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        super().bind(model)
        self.trees = self.shared.available(model) is None
        if self.trees:
            self.shared.bind(model)

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        trees: list[Batch] = []
        rest: list[Job] = []
        for job in jobs:
            prefix = self.shared.share(job.items, token_budget) if self.trees else 0
            if prefix:
                trees.append(
                    Batch(
                        parts=[(job, list(range(len(job.items))))], shared_prefix=prefix
                    )
                )
            else:
                rest.append(job)
        return trees + super().plan(rest, token_budget)
