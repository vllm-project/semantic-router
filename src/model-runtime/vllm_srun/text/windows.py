"""Long inputs: reject, truncate or overlapping windows, as the legacy router bindings read them.

A text is tokenized once, without special tokens: content token IDs and their
code-point offsets. The tokenizer's fixed special-token envelope (``<bos>`` …
``<eos>``) frames every forward, so a window never re-tokenizes a slice of the
text and literal special-token text in the content stays content.

- ``reject``: an input whose framed length exceeds the budget is an item error.
- ``truncate``: sequence and score heads keep the content's first tokens inside
  the envelope. Token heads read the longest text prefix whose own tokenization
  fits (``fit_prefix``), so spans come from exactly what the model read.
- ``window``: windows of ``size`` tokens including the envelope, each starting
  ``size - envelope - overlap`` content tokens after the previous one; the last
  ends at the content's end. Score and distribution windows reduce per label
  with the maximum (``reduce_max``); token windows keep, for each content
  token, the window that gives it the most context on its shorter side, the
  earlier window on ties (``merge_token_windows``), then decode spans once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import MAX_LENGTH_EXCEEDED, SCAN_BUDGET_EXCEEDED
from . import bounds

PROBES = ("", "a", "Hello, world!", "<bos> <eos> 中文 🚀\n\t")


class InputTooLongError(ValueError):
    """An input exceeds its token budget under the requested overflow policy.

    ``code`` is the item error: ``scan_budget_exceeded`` for an input read in
    windows (none of it was scanned), ``max_length_exceeded`` otherwise.
    """

    def __init__(
        self,
        tokens: int,
        limit: int,
        what: str = "input",
        code: str = MAX_LENGTH_EXCEEDED,
    ):
        super().__init__(f"{what} has {tokens} tokens, the budget is {limit}")
        self.tokens = tokens
        self.limit = limit
        self.code = code


def over_budget(tokens: int, limit: int, overflow: str) -> InputTooLongError:
    """The error for an input over its budget under ``overflow`` (``tokens`` may be a lower bound)."""
    code = SCAN_BUDGET_EXCEEDED if overflow == "window" else MAX_LENGTH_EXCEEDED
    return InputTooLongError(tokens, limit, code=code)


@dataclass(frozen=True)
class Envelope:
    """The special tokens a tokenizer wraps around one sequence."""

    prefix: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def frame(self, content: Sequence[int]) -> list[int]:
        return [*self.prefix, *content, *self.suffix]

    @classmethod
    def of(cls, tokenizer: Any) -> Envelope:
        """The tokenizer's single-sequence framing; refused when it depends on the text."""
        specials = list(tokenizer.encode("", add_special_tokens=True).ids)
        content = tokenizer.encode("a", add_special_tokens=False).ids
        framed = tokenizer.encode("a", add_special_tokens=True).ids
        split = next(
            (
                index
                for index in range(len(specials) + 1)
                if framed == [*specials[:index], *content, *specials[index:]]
            ),
            None,
        )
        if split is None:
            raise ValueError("the tokenizer's special tokens are not a fixed envelope")
        envelope = cls(tuple(specials[:split]), tuple(specials[split:]))
        for probe in PROBES:
            content = tokenizer.encode(probe, add_special_tokens=False).ids
            if tokenizer.encode(probe, add_special_tokens=True).ids != envelope.frame(
                content
            ):
                raise ValueError(
                    "the tokenizer's special tokens are not a fixed envelope"
                )
        return envelope


@dataclass(frozen=True)
class Encoded:
    """One text tokenized once: content token IDs and their code-point offsets.

    ``complete`` is False when only the text's first tokens were read (a text
    over its budget, ``bounds.read``); ``tokens`` then exceeds the budget and is
    a lower bound.
    """

    text: str
    content: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    envelope: Envelope
    complete: bool = True

    @property
    def tokens(self) -> int:
        """Framed length, special tokens included (what the legacy budgets count)."""
        return len(self.content) + self.envelope.size

    def framed(self, start: int = 0, end: int | None = None) -> list[int]:
        return self.envelope.frame(self.content[start:end])

    def usage(self) -> dict[str, Any]:
        """The input usage of reading the text whole (``tokens_lower_bound`` when it was read in part)."""
        tokens = self.tokens
        usage: dict[str, Any] = {
            "tokens": tokens,
            "processed_tokens": tokens,
            "truncated": False,
        }
        if not self.complete:
            usage["tokens_lower_bound"] = True
        return usage


def encode(
    tokenizer: Any,
    envelope: Envelope,
    text: str,
    budget: int | None = None,
    overflow: str = "truncate",
) -> Encoded:
    """``text`` tokenized once; with ``budget``, only as far as deciding and reading that budget needs.

    Unless ``overflow`` truncates, a text with certainly more tokens than the
    budget (``bounds.surely_over``) fails before it is tokenized.
    """
    if (
        budget is not None
        and overflow != "truncate"
        and bounds.surely_over(tokenizer, text, budget - envelope.size)
    ):
        raise over_budget(budget + 1, budget, overflow)
    if budget is None:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        return Encoded(
            text, tuple(encoding.ids), tuple(map(tuple, encoding.offsets)), envelope
        )
    read = bounds.read(tokenizer, text, budget - envelope.size + 1)
    count = read.tokens
    return Encoded(
        text,
        tuple(read.encoding.ids[:count]),
        tuple(map(tuple, read.encoding.offsets[:count])),
        envelope,
        read.complete,
    )


def truncate(encoded: Encoded, budget: int) -> list[int]:
    """The framed IDs of the content's first tokens that fit ``budget`` (tokenizer truncation)."""
    keep = budget - encoded.envelope.size
    if keep < 0:
        raise InputTooLongError(encoded.tokens, budget, "the special-token envelope")
    return encoded.framed(0, keep)


def fit_prefix(tokenizer: Any, encoded: Encoded, budget: int) -> tuple[Encoded, bool]:
    """The longest text prefix whose own tokenization fits ``budget``, and whether it was cut.

    The cut ends where the budget's last content token ends; a prefix whose
    re-tokenization grows across the cut loses one character at a time.
    """
    if encoded.tokens <= budget:
        return encoded, False
    keep = budget - encoded.envelope.size
    if keep < 0:
        raise InputTooLongError(encoded.tokens, budget, "the special-token envelope")
    end = max((stop for _, stop in encoded.offsets[:keep]), default=0)
    prefix = encode(tokenizer, encoded.envelope, encoded.text[:end])
    while prefix.tokens > budget:
        prefix = encode(tokenizer, encoded.envelope, prefix.text[:-1])
    return prefix, True


@dataclass(frozen=True)
class Window:
    """Content tokens ``[start, end)`` read in one forward."""

    start: int
    end: int


def plan_windows(
    content_tokens: int, envelope: Envelope, size: int, overlap: int
) -> list[Window]:
    """Windows covering every content token; ``size`` counts the envelope."""
    width = size - envelope.size
    if width <= 0:
        raise ValueError("window size must leave room for content after special tokens")
    if not 0 <= overlap < width:
        raise ValueError("window overlap must be smaller than its content width")
    if content_tokens <= 0:
        raise ValueError("window scanning requires nonempty content")
    stride = width - overlap
    windows, start = [], 0
    while True:
        end = min(start + width, content_tokens)
        windows.append(Window(start, end))
        if end == content_tokens:
            return windows
        start += stride


def reduce_max(values: Sequence[Sequence[float]]) -> list[float]:
    """Per-label maximum over windows."""
    return np.max(np.asarray(values, dtype=np.float64), axis=0).tolist()


def merge_token_windows(
    windows: Sequence[Window],
    rows: Sequence[np.ndarray],
    prefix: int,
    content_tokens: int,
) -> np.ndarray:
    """Per content token, the window row with the most context on its shorter side.

    ``rows[i]`` holds window ``i``'s per-token values over its framed IDs; the
    first ``prefix`` rows are its leading special tokens. Ties keep the earlier
    window; every label follows the same choice.
    """
    best = np.full(content_tokens, -1, dtype=np.int64)
    out: np.ndarray | None = None
    for window, row in zip(windows, rows, strict=True):
        tokens = np.arange(window.start, window.end)
        context = np.minimum(tokens - window.start, window.end - tokens - 1)
        chosen = context > best[tokens]
        if out is None:
            out = np.empty((content_tokens, row.shape[-1]), dtype=row.dtype)
        picked = tokens[chosen]
        out[picked] = row[prefix + picked - window.start]
        best[picked] = context[chosen]
    if out is None or (best < 0).any():
        raise ValueError("token windows did not observe every content token")
    return out
