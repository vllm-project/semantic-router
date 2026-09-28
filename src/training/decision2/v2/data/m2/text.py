"""Language-aware sentence units and answer mentions for the v2 constructions.

Units are character spans of the original text, so publisher answer offsets
map onto them directly. Boundaries: after 。！？ (no space needed), after
. ! ? ؟ । ॥ followed by whitespace, and at line breaks; Thai marks sentence
breaks with spaces, so every whitespace run is a boundary there.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from v2.data.textnorm import _CJK, normalize

NO_SPACE_JOIN = frozenset({"zh", "zh-hant", "ja"})
_BOUNDARY = re.compile(r"[。！？]+|[.!?؟।॥]+(?=\s)|\n+")
_THAI_BOUNDARY = re.compile(r"\s+")


def units(text: str, language: str) -> list[tuple[int, int]]:
    """Non-empty unit spans ``(start, end)`` in order, whitespace-trimmed."""
    pattern = _THAI_BOUNDARY if language == "th" else _BOUNDARY
    spans, start = [], 0
    for match in pattern.finditer(text):
        end = match.end() if language != "th" else match.start()
        spans.append((start, end))
        start = match.end()
    spans.append((start, len(text)))
    trimmed = []
    for low, high in spans:
        while low < high and text[low].isspace():
            low += 1
        while high > low and text[high - 1].isspace():
            high -= 1
        if low < high:
            trimmed.append((low, high))
    return trimmed


def join(parts: Sequence[str], language: str) -> str:
    return ("" if language in NO_SPACE_JOIN else " ").join(parts)


def mentions(text: str, answer: str) -> bool:
    """Normalized containment; answers without CJK characters must match at
    word boundaries so that short answers do not match inside words."""
    haystack, needle = normalize(text), normalize(answer)
    if not needle:
        return False
    if _CJK.search(needle):
        return needle in haystack
    return (
        re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack, flags=re.UNICODE)
        is not None
    )


def usable_answer(answer: str) -> bool:
    """At least two normalized characters, and not a lone digit run shorter
    than three characters (those occur almost everywhere)."""
    value = normalize(answer)
    if len(value) < 2:
        return False
    return not (value.isdigit() and len(value) < 3)
