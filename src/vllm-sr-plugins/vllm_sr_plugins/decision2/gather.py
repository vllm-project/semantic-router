"""Which hidden rows of a prefill step the candidate head reads.

A request carries its option endpoints and query position in
``PoolingParams.extra_kwargs[POSITIONS_KEY]``, as produced by the package's own
``encode``: strictly increasing candidate positions, then the query position,
which is the last prompt token. A step computes prompt positions
``[seq_len - scheduled, seq_len)`` of each request; with chunked prefill a
request spans several steps and is scored once its last prompt token is
computed. This module has no vLLM dependency so the plan can be tested alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POSITIONS_KEY = "decision2"
MAX_CANDIDATES = 255
# Model Runner V2 (the default runner) serves only the built-in pooling tasks;
# token_classify is the one whose outputs vary in length per request.
POOLING_TASK = "token_classify"


@dataclass(frozen=True)
class Positions:
    candidates: tuple[int, ...]
    query: int

    @property
    def rows(self) -> tuple[int, ...]:
        return (*self.candidates, self.query)


def parse_positions(extra_kwargs: Any, prompt_len: int) -> Positions | None:
    """Validated positions for one request, or None when absent or malformed."""
    if not isinstance(extra_kwargs, dict):
        return None
    entry = extra_kwargs.get(POSITIONS_KEY)
    if not isinstance(entry, dict):
        return None
    candidates, query = entry.get("candidate_positions"), entry.get("query_position")
    if (
        not isinstance(candidates, (list, tuple))
        or not 2 <= len(candidates) <= MAX_CANDIDATES
        or type(query) is not int
        or query != prompt_len - 1
        or any(type(p) is not int for p in candidates)
        or candidates[0] < 0
        or any(a >= b for a, b in zip(candidates, candidates[1:]))
        or candidates[-1] >= query
    ):
        return None
    return Positions(tuple(candidates), query)


@dataclass(frozen=True)
class StepPlan:
    rows: list[int]
    """Row indices into this step's flattened hidden states, request by request."""
    counts: list[int]
    """How many of ``rows`` belong to each request."""
    finished: list[bool]
    """Whether the request's last prompt token is computed in this step."""


def plan_step(
    scheduled: list[int],
    seq_lens: list[int],
    prompt_lens: list[int],
    positions: list[Positions | None],
) -> StepPlan:
    rows: list[int] = []
    counts: list[int] = []
    finished: list[bool] = []
    offset = 0
    for size, seq_len, prompt_len, position in zip(
        scheduled, seq_lens, prompt_lens, positions, strict=True
    ):
        start = seq_len - size
        taken = (
            []
            if position is None
            else [offset + p - start for p in position.rows if start <= p < seq_len]
        )
        rows.extend(taken)
        counts.append(len(taken))
        finished.append(seq_len >= prompt_len)
        offset += size
    return StepPlan(rows, counts, finished)
