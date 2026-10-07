"""Shared-context profile: run the common state of a multi-question request once.

Tree mode packs the shared prefix and every question suffix into one row (the
engine's ``shared_prefix`` forward). The shared prefix is the longest token
prefix of all of a request's questions, cut before the first option endpoint
and rounded down to ``align``. A request shares it when it has at least
``min_questions`` questions and saves at least ``min_shared_tokens`` prefix
tokens, ``(questions - 1) * prefix`` (None: the backbone's measured break-even),
and its packed row fits the forward token budget; every other request runs
exactly. This is the released runtime's shared-context switch with its
recommended policy (tree mode, τ 0, automatic break-even), and the engine runs
the same operations, so the switch's accuracy record applies. Answers can
differ from the exact path by rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ..plugins.base import Batch, Job, LoadedModel, Profile, WorkItem
from ..plugins.decisions import RenderedItem
from ..scheduler.planner import exact_split, padded

# Shared-prefix tokens a request must save before tree mode pays off, as
# (with HIP graphs, eager): dense backbones, then Gated DeltaNet hybrids by
# hidden size. Wider hybrids break even at WIDE_HYBRID_BREAK_EVEN.
DENSE_BREAK_EVEN = (3072, 1024)
HYBRID_BREAK_EVEN = ((1024, 6144, 2048), (2048, 3072, 1536), (3072, 1536, 1024))
WIDE_HYBRID_BREAK_EVEN = 512
MIN_SHARED_PREFIX = 4


@dataclass(frozen=True)
class SharePolicy:
    mode: str = "tree"
    min_questions: int = 2
    min_shared_tokens: int | None = None
    align: int = 1
    tau: float = 0.0
    fallback: str = "request"


def shared_prefix(items: list[RenderedItem], align: int = 1) -> int:
    """Tokens every item starts with, before any option endpoint, rounded down to ``align``."""
    ids = [item.ids for item in items]
    low, high = min(ids), max(ids)
    common = next(
        (i for i, (a, b) in enumerate(zip(low, high, strict=False)) if a != b),
        min(len(low), len(high)),
    )
    limit = min(min(item.gather) for item in items)
    return min(common, limit) // align * align


def auto_shared_tokens(config: dict[str, Any], graphs: bool) -> int:
    """Prefix tokens a request must save before sharing pays off on this backbone.

    Sharing adds launches (a packed row, two attention or gated-delta kernels per
    layer) that a small backbone recovers only on larger requests, later still when
    the exact path replays graphs. The values are the break-evens measured on one
    MI325X for the released runtime's switch.
    """
    hidden = config.get("hidden_size", 0)
    if "linear_attention" not in set(config.get("layer_types") or ()):
        with_graphs, eager = DENSE_BREAK_EVEN
        return with_graphs if graphs else eager
    for max_hidden, with_graphs, eager in HYBRID_BREAK_EVEN:
        if hidden <= max_hidden:
            return with_graphs if graphs else eager
    return WIDE_HYBRID_BREAK_EVEN


class SharedContextProfile(Profile):
    name = "shared_context"
    numerics = "approximate"
    description = "Compute the shared state of a multi-question request once (tree mode, automatic break-even)."

    def __init__(self, policy: SharePolicy | None = None):
        self.policy = policy or SharePolicy()
        self.threshold = self.policy.min_shared_tokens
        self.model: LoadedModel[Any, Any] | None = None

    def available(self, model: LoadedModel[Any, Any]) -> str | None:
        if not model.engine_model.supports_shared_context:
            return "the loaded engine model has no shared-context forward"
        if self.policy.mode != "tree" or self.policy.tau > 0:
            return "only tree mode with tau 0 is implemented"
        return None

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        self.model = model
        engine = model.engine_model
        if self.threshold is None:
            config = engine.spec.backbone.config if engine.spec is not None else {}
            self.threshold = auto_shared_tokens(config, engine.replays_graphs)

    def share(self, items: list[WorkItem], token_budget: int | None) -> int:
        """The prefix this job shares, or 0 when it runs exactly."""
        decided = self.model.shared_context(items, token_budget) if self.model else None
        if decided is not None:
            return decided
        if len(items) < self.policy.min_questions:
            return 0
        prefix = shared_prefix(cast("list[RenderedItem]", items), self.policy.align)
        if prefix < MIN_SHARED_PREFIX or (len(items) - 1) * prefix < (
            self.threshold or 0
        ):
            return 0
        packed = prefix + sum(len(item.ids) - prefix for item in items)
        if token_budget is not None and padded(packed) > token_budget:
            return 0
        return prefix

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        batches = []
        for job in jobs:
            prefix = self.share(job.items, token_budget)
            if prefix:
                batches.append(
                    Batch(
                        parts=[(job, list(range(len(job.items))))], shared_prefix=prefix
                    )
                )
                continue
            for indices in exact_split(self.model, job.items, token_budget):
                batches.append(Batch(parts=[(job, indices)]))
        return batches
