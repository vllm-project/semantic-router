"""The ``qwen3.5-decision`` runtime of Decision 1.0: Eos, Sol, Nox and Lux.

A Qwen3.5 backbone reads the segmented candidate-endpoint prompt
(``text/segments.py``) and the shared candidate head (``heads/candidate.py``)
scores each option's endpoint against the query row in FP32. The released
numerics: on GPUs the backbone holds every parameter in BF16 under BF16
autocast, on CPU it runs in FP32; a request's questions run in physical
batches of eight in request order, padded to a multiple of 32 tokens; the
probabilities are the softmax of the logits over the package's per-type
temperature, renormalized on the host.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ...errors import PackageError
from ...plugins.decisions import RenderedItem
from ...text import segments
from .questions import NoulDefaults, Row

PHYSICAL_BATCH = 8
PAD_MULTIPLE = 32
MAX_INPUT_TOKENS = 16384
PROMPT_VERSION = "structured-segmented-candidate-endpoints-global-query-v2"
NOUL_DEFAULTS = NoulDefaults(
    false="The answer to the question is no.",
    true="The answer to the question is yes.",
    null_is_default=False,
)
# Per-model choices of the released runtime, keyed by the package's model name.
NULL_CHOICE_AS_KEY = frozenset({"Decision-1.0-Nox-4B"})
FP64_CONV_ON_GFX942 = frozenset({"Decision-1.0-Eos-0.8B"})


def check_config(config: dict[str, Any]) -> int:
    """The candidate head width of a pointer-v2 Decision 1.0 decoder; raises on anything else."""
    if config.get("prompt_version") != PROMPT_VERSION:
        raise PackageError("not a pointer-v2 Decision 1.0 decoder (prompt_version)")
    head_dim = config.get("head_dim")
    if type(head_dim) is not int or head_dim < 1:
        raise PackageError("decision_config.json needs a positive head_dim")
    return head_dim


def render(
    row: Row,
    state: str,
    encode_text: Callable[[str], list[int]],
    max_length: int,
    null_choice_as_key: bool,
) -> RenderedItem:
    """The question's segments, endpoints and query; a null Choice description renders as null or the key."""
    options = [
        {
            "key": candidate.key,
            "description": (
                candidate.key
                if candidate.description is None
                and row.kind == "choice"
                and null_choice_as_key
                else candidate.description
            ),
        }
        for candidate in row.candidates
    ]
    encoded = segments.encode(
        row.question_id,
        state,
        row.kind,
        row.instructions,
        options,
        encode_text,
        max_length,
    )
    return RenderedItem(
        question_id=row.question_id,
        task_type=row.kind,
        ids=encoded["ids"],
        gather=encoded["endpoints"],
        query=encoded["query"],
        keys=row.keys,
        descriptions=[candidate.description for candidate in row.candidates],
    )


def physical_batches(
    items: list[RenderedItem], budget: int | None = None
) -> list[list[int]]:
    """Item indices in request order, in batches of eight.

    A batch whose padded size exceeds ``budget`` tokens splits into
    consecutive parts that fit. The released packages never reach it (eight
    rows at the input limit stay within it); wider packages could.
    """
    indices = list(range(len(items)))
    batches = [
        indices[start : start + PHYSICAL_BATCH]
        for start in range(0, len(indices), PHYSICAL_BATCH)
    ]
    if budget is None:
        return batches
    return [part for batch in batches for part in _within(batch, items, budget)]


def _within(
    batch: list[int], items: list[RenderedItem], budget: int
) -> list[list[int]]:
    """``batch`` in consecutive parts whose padded size stays within ``budget`` tokens."""
    parts: list[list[int]] = []
    width = 0
    for index in batch:
        padded = -(-len(items[index].ids) // PAD_MULTIPLE) * PAD_MULTIPLE
        if padded > budget:
            raise ValueError("a single question exceeds the forward token budget")
        if parts and max(width, padded) * (len(parts[-1]) + 1) <= budget:
            parts[-1].append(index)
            width = max(width, padded)
        else:
            parts.append([index])
            width = padded
    return parts


def probabilities(
    scores: torch.Tensor, items: list[RenderedItem], temperatures: dict[str, float]
) -> list[list[float] | None]:
    """Per item ``softmax(logits / T)`` on the device, renormalized on the host; None when not finite."""
    staged = [
        (scores[slot, : len(item.keys)].float() / temperatures[item.task_type]).softmax(
            -1
        )
        for slot, item in enumerate(items)
    ]
    host = torch.cat(staged).tolist()
    out: list[list[float] | None] = []
    offset = 0
    for item in items:
        values = host[offset : offset + len(item.keys)]
        offset += len(item.keys)
        if any(not math.isfinite(value) for value in values):
            out.append(None)
            continue
        total = sum(values)
        out.append([value / total for value in values])
    return out
