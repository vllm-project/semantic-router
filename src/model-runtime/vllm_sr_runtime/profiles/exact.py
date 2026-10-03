"""The default profile: the released runtime's numerics and batching, bit for bit."""

from __future__ import annotations

from ..plugins.base import Batch, Job, LoadedModel, Profile
from ..scheduler.planner import micro_batches


class ExactProfile(Profile):
    name = "exact"
    numerics = "exact"
    description = (
        "One request's questions as one padded batch, split only by the forward token budget; "
        "byte-identical to the released packages' runtime."
    )

    def __init__(self) -> None:
        self.fuse = False

    def available(self, model: LoadedModel) -> str | None:
        self.fuse = bool(getattr(model, "fuse_bundled_jobs", False))
        return None

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """One request (or, for models that fuse bundles, one bundle group) per batch."""
        units: list[list[Job]] = []
        groups: dict[int, list[Job]] = {}
        for job in jobs:
            if self.fuse and job.group is not None:
                if job.group not in groups:
                    groups[job.group] = []
                    units.append(groups[job.group])
                groups[job.group].append(job)
            else:
                units.append([job])
        batches = []
        for unit in units:
            rows = [(job, index) for job in unit for index in range(len(job.items))]
            for indices in micro_batches(
                [len(job.items[index].ids) for job, index in rows], token_budget
            ):
                parts: dict[int, tuple[Job, list[int]]] = {}
                for position in indices:
                    job, index = rows[position]
                    parts.setdefault(id(job), (job, []))[1].append(index)
                batches.append(Batch(parts=list(parts.values())))
        return batches
