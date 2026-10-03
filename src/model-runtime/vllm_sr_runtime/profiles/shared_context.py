"""Shared-context profile: run the common state of a multi-question request once.

Tree mode packs the shared prefix and every question suffix into one row;
cache mode runs the prefix with a cache and the suffixes as a batch. A
request with a near-tie answer can fall back to the exact path (τ guard),
and requests below the backbone's measured break-even run exactly. Answers
can differ from the exact path by rounding; the accuracy record lives in
``docs/records/``. The engine-side tree forward is provided by the inference
track; until an engine model advertises ``shared_context``, the profile
reports itself unavailable instead of silently running the exact path.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plugins.base import Batch, Job, LoadedModel, Profile
from ..scheduler.planner import micro_batches


@dataclass(frozen=True)
class SharePolicy:
    mode: str = "tree"
    min_questions: int = 2
    min_shared_tokens: int | None = None
    align: int = 1
    tau: float = 0.0
    fallback: str = "request"


class SharedContextProfile(Profile):
    name = "shared_context"
    numerics = "approximate"
    description = "Compute the shared state of a multi-question request once (tree mode, τ fallback, break-even)."

    def __init__(self, policy: SharePolicy | None = None):
        self.policy = policy or SharePolicy()

    def available(self, model: LoadedModel) -> str | None:
        if not getattr(model.engine_model, "supports_shared_context", False):
            return "the loaded engine model has no shared-context forward"
        return None

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        batches = []
        for job in jobs:
            if len(job.items) >= self.policy.min_questions:
                batches.append(Batch(parts=[(job, list(range(len(job.items))))]))
                continue
            for indices in micro_batches(
                [len(item.ids) for item in job.items], token_budget
            ):
                batches.append(Batch(parts=[(job, indices)]))
        return batches
