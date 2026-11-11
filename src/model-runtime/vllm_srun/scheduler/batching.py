"""How queued requests become forwards: the batch planner a model's engine arguments configure.

Every row of every queued request (one per question of a decision request,
one per input or window of the other surfaces) is planned together: rows of
concurrent requests share forwards of at most ``max_num_batched_tokens``
tokens and ``max_num_seqs`` rows (``coalesced``). Each lane of the model
(``LoadedModel.lane``: text rows, rows that read images) is planned on its own,
so no forward mixes lanes. Padded forwards take rows longest first and count
their rows times their longest row; a model that packs merged rows back to
back counts their sum and takes them in arrival order. Padding, neighbours and
GEMM shapes then depend on the rows beside a row, so answers can move by
rounding; ``max_num_seqs`` 1 runs every row alone. A batch-invariant
model, whose rows give the same answer alone or in any batch, merges the
requests already queued without waiting and keeps its forwards short
(``merged``), so its answers never change. With shared context, a
multi-question request whose common state pays off runs as one packed
shared-context tree instead (``SharedContext``); on a model whose items each
name their own context (``LoadedModel.merges_shared_batches``: state-once
models), whole requests share those forwards (``shared_batches``).

The golden check plans its requests as references: each alone, in the
forwards the model's released runtime runs for one request (``reference``),
so readiness compares like with like against the recorded answers.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..config import DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS
from ..plugins.base import Batch, Job, LoadedModel, WorkItem
from .planner import cost, exact_split, length_class, padded

if TYPE_CHECKING:
    from ..config import EngineArgs
    from ..plugins.decisions import RenderedItem

# Padded tokens of one merged forward on a batch-invariant model. It bounds how
# long one forward holds the device, so a short request queued behind the
# windows of a long one waits for one window, not for all of them.
MERGED_BATCH_TOKENS = 512
# Shared-prefix tokens a request must save before tree mode pays off, as
# (with HIP graphs, eager): dense backbones, then Gated DeltaNet hybrids by
# hidden size. Wider hybrids break even at WIDE_HYBRID_BREAK_EVEN.
DENSE_BREAK_EVEN = (3072, 1024)
HYBRID_BREAK_EVEN = ((1024, 6144, 2048), (2048, 3072, 1536), (3072, 1536, 1024))
WIDE_HYBRID_BREAK_EVEN = 512
MIN_SHARED_PREFIX = 4


def _batch(
    parts: dict[int, tuple[Job, list[int]]], exact: bool = True, lane: str = "text"
) -> Batch:
    for _, indices in parts.values():
        indices.sort()
    return Batch(parts=list(parts.values()), exact=exact, lane=lane)


def reference(
    model: LoadedModel[Any, Any] | None,
    jobs: list[Job],
    token_budget: int | None,
) -> list[Batch]:
    """Every job alone, in the forwards its model's released runtime runs for one request (the golden check).

    Each forward keeps the released split's row order, which a family's
    physical batches set (``LoadedModel.exact_batches``).
    """
    return [
        Batch(parts=[(job, list(indices))])
        for job in jobs
        for indices in exact_split(model, job.items, token_budget)
    ]


def merged(
    jobs: list[Job], cap: int, banded: bool = True, max_rows: int | None = None
) -> list[Batch]:
    """Every job's rows in shared batches, each within ``cap`` padded tokens and ``max_rows`` rows.

    Only for batch-invariant models, where a row's answer does not depend on
    the batch it runs in. Rows with the same token IDs stay in one batch, where
    the family computes them once, and count once against ``cap`` and
    ``max_rows``; inputs without token IDs (images, audio) count by their
    ``cost``. A row costlier than ``cap`` runs alone. ``banded`` keeps each
    batch to one length class, so padded rows pad little; a model that packs
    its rows needs no bands, and closed-loop callers' short texts, spread over
    several classes, then still share forwards.
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
            or (max_rows is not None and count + 1 > max_rows)
        ):
            batches.append(_batch(current))
            current, width, count = {}, 0, 0
        band = length_class(length)
        for job, index in rows:
            current.setdefault(id(job), (job, []))[1].append(index)
        width, count = max(width, length), count + 1
    if current:
        batches.append(_batch(current))
    return batches


def coalesced(
    jobs: list[Job],
    budget: int | Callable[[str], int],
    max_rows: int,
    lane: Callable[[WorkItem], str] | None = None,
    packed: bool = False,
) -> list[Batch]:
    """Rows of every queued request in shared batches, each holding the rows of one lane.

    ``lane`` names each row's lane (one lane when None) and ``budget`` is the
    token budget of every lane, or of each lane by name. A padded batch holds
    at most its budget in rows times its longest row, longest first, so rows
    of like lengths share a forward; a ``packed`` batch (a model that lays
    merged rows back to back) holds at most its budget in the sum of its rows,
    each rounded up to 8, in arrival order. Every batch holds at most
    ``max_rows`` rows; a row costlier than its budget runs alone. The n-th
    batches of the lanes share one turn (``Batch.sub``), so planning a round
    of arrivals adds as many turns as its busiest lane needs, as it would
    without lanes. The batches are approximate: a row's neighbours, padding
    and GEMM shapes depend on the rows beside it.
    """
    lanes: dict[str, list[tuple[Job, int]]] = {}
    for job in jobs:
        for index, item in enumerate(job.items):
            name = lane(item) if lane is not None else "text"
            lanes.setdefault(name, []).append((job, index))
    planned: list[list[Batch]] = []
    for name, rows in lanes.items():
        cap = budget(name) if callable(budget) else budget
        if not packed:
            rows.sort(key=lambda pair: -cost(pair[0].items[pair[1]]))
        batches: list[Batch] = []
        current: dict[int, tuple[Job, list[int]]] = {}
        width = total = count = 0
        for job, index in rows:
            length = padded(cost(job.items[index]))
            new_width = max(width, length)
            size = total + length if packed else new_width * (count + 1)
            if current and (size > cap or count + 1 > max_rows):
                batches.append(_batch(current, exact=False, lane=name))
                current, total, count = {}, 0, 0
                new_width = length
            current.setdefault(id(job), (job, []))[1].append(index)
            width, total, count = new_width, total + length, count + 1
        if current:
            batches.append(_batch(current, exact=False, lane=name))
        planned.append(batches)
    turns = []
    for group in itertools.zip_longest(*planned):
        sub = tuple(batch for batch in group if batch is not None)
        turns.append(sub[0] if len(sub) == 1 else _turn(sub))
    return turns


def _turn(sub: tuple[Batch, ...]) -> Batch:
    """One turn for batches of several lanes: their union, run batch by batch."""
    parts: dict[int, tuple[Job, list[int]]] = {}
    for batch in sub:
        for job, indices in batch.parts:
            parts.setdefault(id(job), (job, []))[1].extend(indices)
    for _, indices in parts.values():
        indices.sort()
    return Batch(
        parts=list(parts.values()),
        exact=False,
        lane="+".join(batch.lane for batch in sub),
        sub=sub,
    )


def shared_batches(trees: list[Batch], budget: int, max_rows: int) -> list[Batch]:
    """Shared-context batches of one request each, packed into batches of several requests.

    Only for a model whose items each name their own context. A request's
    items stay together; requests go longest first into batches of at most
    ``budget`` tokens (the sum of their items' costs: a state-once question
    costs its block and its share of the state) and ``max_rows`` items. A
    request costlier than ``budget``, or with more items than ``max_rows``,
    runs alone.
    """
    units = sorted(trees, key=lambda batch: -sum(cost(item) for item in batch.items()))
    batches: list[Batch] = []
    current: list[Batch] = []
    tokens = rows = 0
    for unit in units:
        size = sum(cost(item) for item in unit.items())
        count = len(unit.items())
        if current and (tokens + size > budget or rows + count > max_rows):
            batches.append(_shared(current))
            current, tokens, rows = [], 0, 0
        current.append(unit)
        tokens, rows = tokens + size, rows + count
    if current:
        batches.append(_shared(current))
    return batches


def _shared(units: list[Batch]) -> Batch:
    return Batch(
        parts=[part for unit in units for part in unit.parts],
        shared_prefix=max(unit.shared_prefix for unit in units),
    )


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
    the forwards replay graphs. The values are the break-evens measured on one
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


class SharedContext:
    """Which multi-question requests run their common state once, as one packed shared-context tree.

    Tree mode packs the shared prefix and every question suffix into one row
    (the engine's ``shared_prefix`` forward). The shared prefix is the longest
    token prefix of all of a request's questions, cut before the first option
    endpoint and rounded down to ``align``. A request shares it when it has at
    least ``min_questions`` questions and saves at least ``min_shared_tokens``
    prefix tokens, ``(questions - 1) * prefix`` (None: the backbone's measured
    break-even), and its packed row fits the forward token budget; every
    other request is batched as usual. This is the released runtime's
    shared-context switch with its recommended policy (tree mode, τ 0,
    automatic break-even), and the engine runs the same operations. Answers
    can differ from those of batched rows by rounding.
    """

    def __init__(self, policy: SharePolicy | None = None):
        self.policy = policy or SharePolicy()
        self.threshold = self.policy.min_shared_tokens
        self.model: LoadedModel[Any, Any] | None = None

    def unavailable(self, model: LoadedModel[Any, Any]) -> str | None:
        """None when ``model`` can run shared-context trees, else the reason it cannot."""
        if not model.engine_model.supports_shared_context:
            return "the loaded engine model has no shared-context forward"
        if self.policy.mode != "tree" or self.policy.tau > 0:
            return "only tree mode with tau 0 is implemented"
        return None

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        self.model = model
        if self.threshold is None:
            engine = model.engine_model
            config = engine.spec.backbone.config if engine.spec is not None else {}
            self.threshold = auto_shared_tokens(config, engine.replays_graphs)

    def share(self, items: list[WorkItem], token_budget: int | None) -> int:
        """The prefix this job shares, or 0 when its rows are batched as usual."""
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


class Batcher:
    """Turns a model's queued jobs into forwards, as its engine arguments say.

    ``coalesces`` says the scheduler should hold a queue for the batching
    window before planning it, so rows of concurrent requests share forwards:
    for any model but a batch-invariant one, once a forward may hold more than
    one row. Every item of every job is planned exactly once. Limits the
    engine arguments leave unset take the CPU's defaults; the runtime resolves
    them for the model's device before it builds the batcher.
    """

    def __init__(self, args: EngineArgs, shared: SharedContext | None = None):
        self.max_num_seqs = args.max_num_seqs or DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = (
            args.max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS
        )
        self.shared = shared
        self.model: LoadedModel[Any, Any] | None = None

    def bind(self, model: LoadedModel[Any, Any]) -> None:
        """Take what planning needs from the model this batcher serves."""
        self.model = model
        if self.shared is not None:
            self.shared.bind(model)

    @property
    def coalesces(self) -> bool:
        return self.max_num_seqs > 1 and not (
            self.model is not None and self.model.batch_invariant
        )

    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        trees: list[Batch] = []
        rest: list[Job] = []
        for job in jobs:
            prefix = self.shared.share(job.items, token_budget) if self.shared else 0
            if prefix:
                trees.append(
                    Batch(
                        parts=[(job, list(range(len(job.items))))], shared_prefix=prefix
                    )
                )
            else:
                rest.append(job)
        model = self.model
        if len(trees) > 1 and model is not None and model.merges_shared_batches:
            budget = min(
                token_budget or self.max_num_batched_tokens, self.max_num_batched_tokens
            )
            family = model.coalesced_token_budget()
            trees = shared_batches(
                trees, min(budget, family) if family else budget, self.max_num_seqs
            )
        if not rest:
            return trees
        if model is not None and model.batch_invariant:
            cap = min(
                token_budget or MERGED_BATCH_TOKENS,
                MERGED_BATCH_TOKENS,
                self.max_num_batched_tokens,
            )
            return trees + merged(rest, cap, not model.packs_rows, self.max_num_seqs)
        budget = min(
            token_budget or self.max_num_batched_tokens, self.max_num_batched_tokens
        )
        if model is None:
            return trees + coalesced(rest, budget, self.max_num_seqs)

        def lane_budget(lane: str) -> int:
            family = model.coalesced_token_budget(lane)
            return min(budget, family) if family else budget

        return trees + coalesced(
            rest, lane_budget, self.max_num_seqs, model.lane, model.packs_merged_rows
        )
