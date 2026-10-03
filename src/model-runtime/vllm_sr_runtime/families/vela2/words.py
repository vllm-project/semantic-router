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

import numpy as np

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
_ALNUM = re.compile(r"[^\W_]")
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

    offsets: np.ndarray
    first: np.ndarray

    def __len__(self) -> int:
        return len(self.first)

    def select(self, keep: np.ndarray) -> Words:
        return Words(self.offsets[keep], self.first[keep])


def split_words(text: str) -> list[tuple[int, int]]:
    """The word units of ``text`` as code-point ranges."""
    return [(match.start(), match.end()) for match in _WORD.finditer(text)]


def words_of(text: str, token_offsets: np.ndarray | None) -> Words:
    """Every word of ``text`` with the first token that overlaps it (words no token covers are dropped)."""
    units = split_words(text)
    if not units or token_offsets is None or len(token_offsets) == 0:
        return Words(np.zeros((0, 2), np.int32), np.zeros(0, np.int32))
    offsets = np.asarray(token_offsets)
    starts, ends = offsets[:, 0], offsets[:, 1]
    valid = ends > starts
    reach = np.maximum.accumulate(np.where(valid, ends, -1))
    word_offsets, first = [], []
    for start, end in units:
        token = int(np.searchsorted(reach, start, side="right"))
        while token < len(offsets) and not valid[token]:
            token += 1
        if token < len(offsets) and starts[token] < end:
            word_offsets.append((start, end))
            first.append(token)
    return Words(
        np.asarray(word_offsets, np.int32).reshape(-1, 2),
        np.asarray(first, np.int32),
    )


def _units(text: str) -> np.ndarray:
    unit = np.full(len(text) + 1, -1, dtype=np.int64)
    for index, match in enumerate(_UNIT.finditer(text)):
        unit[match.start() : match.end()] = index
    return unit


def _first_nonspace(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


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


def decode_spans(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    labels: list[str],
    text: str,
    threshold: float,
) -> list[dict[str, object]]:
    """Labelled spans from word x label probabilities (the packages' word readout decoder).

    Each word takes its most probable label when that probability exceeds the
    threshold (in float32, as the packages compare). Inside one unit of the
    text, conflicting labels are resolved by summed probability and gaps
    between words of the same label are filled; consecutive words with the
    same label form a span, whose probability is the mean over its words.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    offsets = np.asarray(offsets)
    count = len(offsets)
    if count == 0:
        return []
    valid = offsets[:, 1] > offsets[:, 0]
    best = probs.argmax(1)
    best_probability = probs[np.arange(count), best]
    label = np.where(valid & (best_probability > threshold), best, -1)
    units = _units(text)
    word_unit = np.full(count, -1, dtype=np.int64)
    for index in range(count):
        if not valid[index]:
            continue
        start, end = int(offsets[index, 0]), int(offsets[index, 1])
        first = _first_nonspace(text, start, end)
        word_unit[index] = units[first] if first < end and first < len(text) else -1
    index = 0
    while index < count:
        if word_unit[index] < 0:
            index += 1
            continue
        last = index
        while last + 1 < count and (
            word_unit[last + 1] == word_unit[index] or not valid[last + 1]
        ):
            last += 1
        members = [i for i in range(index, last + 1) if valid[i]]
        labelled = [i for i in members if label[i] >= 0]
        if labelled:
            if len({int(label[i]) for i in labelled}) > 1:
                votes: dict[int, float] = {}
                for i in labelled:
                    votes[int(label[i])] = votes.get(int(label[i]), 0.0) + float(
                        best_probability[i]
                    )
                winner = max(votes, key=votes.get)
                for i in labelled:
                    label[i] = winner
            for value in {int(label[i]) for i in labelled}:
                positions = [i for i in labelled if label[i] == value]
                for i in members:
                    if positions[0] < i < positions[-1] and label[i] < 0:
                        label[i] = value
        index = last + 1
    runs: list[list] = []
    current: list | None = None
    for index in range(count):
        if not valid[index]:
            continue
        value = int(label[index])
        if current is not None and value == current[0]:
            current[2] = int(offsets[index, 1])
            current[3].append(index)
            continue
        if current is not None:
            runs.append(current)
        current = (
            [value, int(offsets[index, 0]), int(offsets[index, 1]), [index]]
            if value >= 0
            else None
        )
    if current is not None:
        runs.append(current)
    spans = []
    for value, first, last, members in runs:
        start, end = trim(text, first, last)
        if end <= start:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "label": labels[value],
                "probability": float(np.mean(probs[members, value])),
            }
        )
    return spans
