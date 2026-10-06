"""A one-by-one profile: the smallest complete profile plugin.

Every job runs in its own forwards, in arrival order, split as the released
runtime splits one request, so its numerics are ``exact``. A model serves it
with ``--profile example_one_by_one``, and a request picks it with
``options.profile``.
"""

from __future__ import annotations

from typing import Any

from vllm_srun.plugins.base import Batch, Job, LoadedModel, Profile
from vllm_srun.scheduler.planner import exact_split


class OneByOneProfile(Profile):
    name = "example_one_by_one"
    numerics = "exact"
    description = "Every job alone, in arrival order, split as the released runtime splits a request."

    def __init__(self) -> None:
        self.model: LoadedModel[Any, Any] | None = None

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        self.model = model

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        return [
            Batch(parts=[(job, indices)])
            for job in jobs
            for indices in exact_split(self.model, job.items, token_budget)
        ]
