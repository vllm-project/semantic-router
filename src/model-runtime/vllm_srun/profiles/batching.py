"""Cross-request batching: questions of concurrent requests share padded forwards.

Rows of different requests run in one batch, so padding and GEMM shapes
differ from the exact path and answers can change by rounding. The
scheduler collects requests for at most ``batch_window_ms``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..plugins.base import Batch, Job, Profile
from ..scheduler.planner import cost, padded

if TYPE_CHECKING:
    from ..config import ServeConfig

DEFAULT_MAX_BATCH_TOKENS = 65_536


class BatchingProfile(Profile):
    name = "batching"
    numerics = "approximate"
    coalesces = True
    description = "Coalesce questions from concurrent requests into shared padded batches (higher throughput)."

    def __init__(self, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS):
        self.max_batch_tokens = max_batch_tokens

    @classmethod
    def from_config(cls, config: ServeConfig) -> BatchingProfile:
        return cls(max_batch_tokens=config.max_batch_tokens)

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        budget = min(token_budget or self.max_batch_tokens, self.max_batch_tokens)
        rows = sorted(
            ((job, index) for job in jobs for index in range(len(job.items))),
            key=lambda pair: -cost(pair[0].items[pair[1]]),
        )
        batches: list[Batch] = []
        current: dict[int, tuple[Job, list[int]]] = {}
        width = 0
        count = 0
        for job, index in rows:
            length = padded(cost(job.items[index]))
            new_width = max(width, length)
            if current and new_width * (count + 1) > budget:
                batches.append(Batch(parts=list(current.values()), exact=False))
                current, width, count = {}, 0, 0
                new_width = length
            current.setdefault(id(job), (job, []))[1].append(index)
            width, count = new_width, count + 1
        if current:
            batches.append(Batch(parts=list(current.values()), exact=False))
        for batch in batches:
            for _, indices in batch.parts:
                indices.sort()
        return batches
