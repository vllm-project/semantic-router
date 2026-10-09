"""Word units and the span decoder of the Vela 2.0 packages.

Span questions are scored per word: a word is a GLiNER2-style whitespace unit
(CJK characters one by one, URLs, e-mail addresses and handles whole), read at
its first sub-word token. Decoding labels each word by its most probable label
above the threshold, votes and fills inside a unit, joins neighbours with the
same label and trims brackets, quotes and URL tails. Every rule reproduces the
packages' engine exactly (``vela2_inference.py``), including its float32
decoding and its character classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The packages' CJK class, code point for code point. Its compatibility block
# starts at U+8C48 (an NFC-normalised U+F900 in the released source), so the
# range also covers Hangul and everything up to U+FAFF; parity needs it as is.
_CJK = (
    "\u2e80-\u2fff\u3000-\u303f\u3040-\u30ff\u3100-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\ua000-\ua4cf\uac00-\ud7af\u8c48-\ufaff\ufe30-\ufe4f\uff00-\uffef"
)
_TRAIL = ".,;:!?)\\]}>\"'\u201d\u2019\u00bb"
_WORD = re.compile(
    rf"""(?:https?://|www\.)[^\s{_CJK}]+?(?=[{_TRAIL}]*(?![^\s{_CJK}]))
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}
    |@[a-z0-9_]+
    |[{_CJK}]
    |[^\W{_CJK}]+(?:[-_][^\W{_CJK}]+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)

_UNIT_BODY = rf"[^\s\]\)\}}>\u300c\u300d\u300e\u300f\"'\u201c\u201d{_CJK}]"
_UNIT = re.compile(
    rf"(?:https?://|www\.){_UNIT_BODY}+?(?=[.,;:!?]*(?!{_UNIT_BODY}))"
    rf"|(?:(?![{_CJK}])[^\W_]|[@._\-+])+"
    rf"|\S"
)
_URL_START = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_CJK_CHAR = re.compile(rf"[{_CJK}]")
EDGE = frozenset(
    "[](){}<>\u300c\u300d\u300e\u300f\u3010\u3011\u300a\u300b\u3008\u3009"
    "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u2039\u203a"
)
URL_TRAIL = frozenset(".,;:!?)")


@dataclass(frozen=True)
class Words:
    """The words of a text that overlap its tokens.

    ``offsets`` are ``[start, end)`` code-point offsets (W, 2); ``first`` the
    index of each word's first overlapping sub-word token (W,).
    """

    offsets: NDArray[np.int32]
    first: NDArray[np.int32]

    def __len__(self) -> int:
        return len(self.first)


def split_words(text: str) -> list[tuple[int, int]]:
    """The word units of ``text`` as code-point ranges."""
    return [(match.start(), match.end()) for match in _WORD.finditer(text)]


def words_of(text: str, token_offsets: NDArray[np.int32] | None) -> Words:
    """Every word of ``text`` with the first token that overlaps it (words no token covers are dropped)."""
    units = split_words(text)
    if not units or token_offsets is None or len(token_offsets) == 0:
        return Words(np.zeros((0, 2), np.int32), np.zeros(0, np.int32))
    offsets = np.asarray(token_offsets)
    starts, ends = offsets[:, 0], offsets[:, 1]
    valid = ends > starts
    count = len(offsets)
    reach = np.maximum.accumulate(np.where(valid, ends, -1))
    spans = np.asarray(units, np.int64).reshape(-1, 2)
    # The first token whose running end passes the word start, moved on to the next non-empty token.
    token = np.searchsorted(reach, spans[:, 0], side="right")
    nonempty = np.nonzero(valid)[0]
    position = np.searchsorted(nonempty, token)
    token = np.where(
        position < len(nonempty),
        (
            nonempty[np.minimum(position, max(len(nonempty) - 1, 0))]
            if len(nonempty)
            else count
        ),
        count,
    )
    covered = token < count
    covered[covered] &= starts[token[covered]] < spans[covered, 1]
    return Words(spans[covered].astype(np.int32), token[covered].astype(np.int32))


def _word_units(text: str, offsets: NDArray[np.int32]) -> NDArray[np.intp]:
    """The unit (of ``_UNIT``) each word's first non-space character belongs to, -1 for none."""
    bounds = np.asarray(
        [match.span() for match in _UNIT.finditer(text)], np.int64
    ).reshape(-1, 2)
    first = offsets[:, 0].astype(np.int64)
    for index in np.nonzero([text[a].isspace() for a in first.tolist()])[0]:
        a, b = int(offsets[index, 0]), int(offsets[index, 1])
        while a < b and text[a].isspace():
            a += 1
        first[index] = a
    unit = np.searchsorted(bounds[:, 0], first, side="right") - 1
    inside = (
        (unit >= 0)
        & (first < offsets[:, 1])
        & (first < len(text))
        & (first < bounds[np.maximum(unit, 0), 1] if len(bounds) else False)
    )
    return np.where(inside, unit, -1)


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop surrounding whitespace, brackets and quotes, and a URL's trailing punctuation or CJK tail."""
    while start < end and (text[start].isspace() or text[start] in EDGE):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in EDGE):
        end -= 1
    if not _URL_START.match(text, start):
        return start, end
    cjk = _CJK_CHAR.search(text, start, end)
    if cjk:
        end = cjk.start()
    while end > start and (
        text[end - 1] in URL_TRAIL or text[end - 1] in EDGE or text[end - 1].isspace()
    ):
        end -= 1
    return start, end


class DecodedSpan(TypedDict):
    start: int
    end: int
    label: str
    probability: float


def decode_spans(
    probabilities: NDArray[np.float64],
    offsets: NDArray[np.int32],
    labels: list[str],
    text: str,
    threshold: float,
) -> list[DecodedSpan]:
    """Labelled spans from word x label probabilities (the packages' word readout decoder).

    Each word (a non-empty span of ``words_of``) takes its most probable label
    when that probability exceeds the threshold (in float32, as the packages
    compare). Inside one unit of the text, conflicting labels are resolved by
    summed probability and gaps between labelled words are filled; only units
    that hold a labelled word can change. Consecutive words with the same label
    form a span, whose probability is the mean over its words.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    offsets = np.asarray(offsets)
    count = len(offsets)
    if count == 0:
        return []
    best = probs.argmax(1)
    best_probability = probs[np.arange(count), best]
    label = np.where(best_probability > threshold, best, -1)
    labelled = np.nonzero(label >= 0)[0]
    if not len(labelled):
        return []
    unit = _word_units(text, offsets)
    change = np.ones(count, bool)
    change[1:] = unit[1:] != unit[:-1]
    run_start = np.nonzero(change)[0]
    run_end = np.append(run_start[1:], count)
    run_of = np.cumsum(change) - 1
    for run in np.unique(run_of[labelled]):
        first, end = int(run_start[run]), int(run_end[run])
        if unit[first] < 0:
            continue
        members = [i for i in range(first, end) if label[i] >= 0]
        values = [int(label[i]) for i in members]
        if len(set(values)) > 1:
            votes: dict[int, float] = {}
            for i, value in zip(members, values, strict=True):
                votes[value] = votes.get(value, 0.0) + float(best_probability[i])
            winner = max(votes, key=votes.__getitem__)
            label[members] = winner
        low, high = members[0], members[-1]
        gap = label[low + 1 : high] < 0
        label[low + 1 : high][gap] = label[low]
    edges = np.flatnonzero(np.diff(np.concatenate(([-2], label, [-2]))) != 0)
    spans: list[DecodedSpan] = []
    for begin, stop in pairwise(edges):
        value = int(label[begin])
        if value < 0:
            continue
        start, end = trim(text, int(offsets[begin, 0]), int(offsets[stop - 1, 1]))
        if end <= start:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "label": labels[value],
                "probability": float(np.mean(probs[list(range(begin, stop)), value])),
            }
        )
    return spans
