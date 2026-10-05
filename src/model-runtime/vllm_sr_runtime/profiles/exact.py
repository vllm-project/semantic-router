"""The default profile: the released runtime's numerics and batching, bit for bit."""

from __future__ import annotations

from ..plugins.base import Batch, Job, Profile
from ..scheduler.planner import micro_batches


class ExactProfile(Profile):
    name = "exact"
    numerics = "exact"
    description = (
        "One request's questions as one padded batch, split only by the forward token budget; "
        "byte-identical to the released packages' runtime."
    )

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        batches = []
        for job in jobs:
            for indices in micro_batches(
                [len(item.ids) for item in job.items], token_budget
            ):
                batches.append(Batch(parts=[(job, indices)]))
        return batches
