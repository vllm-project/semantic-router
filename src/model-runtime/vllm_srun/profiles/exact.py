"""The default profile: the released runtime's numerics and batching, bit for bit."""

from __future__ import annotations

from typing import Any

from ..plugins.base import Batch, Job, LoadedModel, Profile
from ..scheduler.planner import cost, exact_split, length_class, padded

# Padded tokens of one shared forward on a batch-invariant model. It bounds how
# long one forward holds the device, so a short request queued behind the
# windows of a long one waits for one window, not for all of them.
MERGED_BATCH_TOKENS = 512


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
        self.banded = True
        self.model: LoadedModel[Any, Any] | None = None

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        self.fuse = model.fuse_bundled_jobs
        self.merge = model.batch_invariant
        self.banded = not model.packs_rows
        self.model = model

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """One request (or, for models that fuse bundles, one bundle group) per batch.

        A batch-invariant model runs every queued job in shared batches instead.
        """
        if self.merge:
            return merged(
                jobs,
                min(token_budget or MERGED_BATCH_TOKENS, MERGED_BATCH_TOKENS),
                self.banded,
            )
        batches = []
        for unit in self._units(jobs):
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


def merged(jobs: list[Job], cap: int, banded: bool = True) -> list[Batch]:
    """Every job's rows in shared batches, each within ``cap`` padded tokens.

    Only for batch-invariant models, where a row's answer does not depend on
    the batch it runs in. Rows with the same token IDs stay in one batch, where
    the family computes them once, and count once against ``cap``; inputs
    without token IDs (images, audio) count by their ``cost``. A row costlier
    than ``cap`` runs alone. ``banded`` keeps each batch to one length class,
    so padded rows pad little; a model that packs its rows needs no bands, and
    closed-loop callers' short texts, spread over several classes, then still
    share forwards.
    """
    units: dict[object, tuple[int, list[tuple[Job, int]]]] = {}
    for job in jobs:
        for index, item in enumerate(job.items):
            key = tuple(item.ids) if item.ids else (id(job), index)
            units.setdefault(key, (padded(cost(item)), []))[1].append((job, index))
    batches: list[Batch] = []
    current: dict[int, tuple[Job, list[int]]] = {}
    band = width = count = 0
    for length, rows in sorted(units.values(), key=lambda unit: unit[0]):
        if current and (
            (banded and length_class(length) != band)
            or max(width, length) * (count + 1) > cap
        ):
            batches.append(Batch(parts=list(current.values())))
            current, width, count = {}, 0, 0
        band = length_class(length)
        for job, index in rows:
            current.setdefault(id(job), (job, []))[1].append(index)
        width, count = max(width, length), count + 1
    if current:
        batches.append(Batch(parts=list(current.values())))
    for batch in batches:
        for _, indices in batch.parts:
            indices.sort()
    return batches
