"""The default profile: the released runtime's numerics and batching, bit for bit."""

from __future__ import annotations

from ..plugins.base import Batch, Job, LoadedModel, Profile
from ..scheduler.planner import exact_split


class ExactProfile(Profile):
    name = "exact"
    numerics = "exact"
    description = (
        "One request's questions as the released runtime batches them (by default one padded batch, "
        "split only by the forward token budget); byte-identical to the released packages' runtime."
    )

    def __init__(self) -> None:
        self.fuse = False
        self.merge = False
        self.model: LoadedModel | None = None

    def available(self, model: LoadedModel) -> str | None:
        self.fuse = bool(getattr(model, "fuse_bundled_jobs", False))
        self.merge = bool(getattr(model, "batch_invariant", False))
        self.model = model
        return None

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """One request (or, for models that fuse bundles, one bundle group) per batch.

        A batch-invariant model runs every queued job in shared batches instead.
        """
        units = ([list(jobs)] if jobs else []) if self.merge else self._units(jobs)
        batches = []
        for unit in units:
            rows = [(job, index) for job in unit for index in range(len(job.items))]
            items = [job.items[index] for job, index in rows]
            for indices in exact_split(self.model, items, token_budget):
                parts: dict[int, tuple[Job, list[int]]] = {}
                for position in indices:
                    job, index = rows[position]
                    parts.setdefault(id(job), (job, []))[1].append(index)
                batches.append(Batch(parts=list(parts.values())))
        return batches

    def _units(self, jobs: list[Job]) -> list[list[Job]]:
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
        return units
