"""Span decoding of the Vela 3.0 d3 sizes: word scores to labelled character spans.

The released runtime decodes a span question's word x label probabilities
with its word-aware decoder (``span_decode_v2.decode_v2``, ``extend=False``),
where each "token" is a word of the target text:

1. each word takes its most probable label when that probability is above
   the threshold (FP32 compare), else none;
2. words group into units (GLiNER2's splitter: a URL, a run of letters,
   digits and ``@._-+``, or one other character; every CJK, kana, hangul and
   full-width character is its own unit); a word belongs to the unit of its
   first non-space character;
3. within a unit with several labels, the label with the largest summed
   probability over the unit's labelled words relabels them all, and
   unlabelled words between two words of one label take that label;
4. contiguous words of one label form a span, trimmed of whitespace and of
   enclosing bracket and quote characters.

Evidence questions decode by sentence or paragraph (``evidence_spans``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# The released decoder's class, code point for code point. Unlike the word splitter's (vela2 ``words._CJK``,
# whose compatibility block starts at U+8C48), the decoder's starts at U+F900.
CJK = (
    "\u2e80-\u2fff\u3000-\u303f\u3040-\u30ff\u3100-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\ua000-\ua4cf\uac00-\ud7af\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef"
)
UNIT = re.compile(
    rf"(?:https?://|www\.)[^\s\]\)\}}>\u300c\u300d\u300e\u300f\"'\u201c\u201d{CJK}]+"
    rf"|(?:(?![{CJK}])[^\W_]|[@._\-+])+"
    rf"|\S"
)
EDGE = frozenset(
    "[](){}<>\u300c\u300d\u300e\u300f\u3010\u3011\u300a\u300b\u3008\u3009\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u2039\u203a"
)
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
PARAGRAPH_BREAK = re.compile(r"(\n\s*\n|\n)")


def units(text: str) -> np.ndarray:
    """Character index -> unit index (-1 for whitespace)."""
    out = np.full(len(text) + 1, -1, dtype=np.int64)
    for index, match in enumerate(UNIT.finditer(text)):
        out[match.start() : match.end()] = index
    return out


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in EDGE):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in EDGE):
        end -= 1
    return start, end


def _first_nonspace(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def decode(
    probabilities: Any,
    offsets: Any,
    names: Sequence[str],
    text: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """``[{start, end, label}]`` (code-point offsets) from word x label probabilities."""
    probs = np.asarray(probabilities, dtype=np.float32)
    offs = np.asarray(offsets)
    count = len(offs)
    if count == 0:
        return []
    valid = offs[:, 1] > offs[:, 0]
    best = probs.argmax(1)
    top = probs[np.arange(count), best]
    label = np.where(valid & (top > threshold), best, -1)
    unit_of = units(text)
    word_unit = np.full(count, -1, dtype=np.int64)
    for t in range(count):
        if not valid[t]:
            continue
        a, b = int(offs[t, 0]), int(offs[t, 1])
        first = _first_nonspace(text, a, b)
        word_unit[t] = unit_of[first] if first < b and first < len(text) else -1
    t = 0
    while t < count:
        if word_unit[t] < 0:
            t += 1
            continue
        last = t
        while last + 1 < count and (
            word_unit[last + 1] == word_unit[t] or not valid[last + 1]
        ):
            last += 1
        members = [i for i in range(t, last + 1) if valid[i]]
        labelled = [i for i in members if label[i] >= 0]
        if labelled:
            if len({int(label[i]) for i in labelled}) > 1:
                scores: dict[int, float] = {}
                for i in labelled:
                    scores[int(label[i])] = scores.get(int(label[i]), 0.0) + float(
                        top[i]
                    )
                winner = max(scores, key=lambda k: scores[k])
                for i in labelled:
                    label[i] = winner
            for c in {int(label[i]) for i in labelled}:
                positions = [i for i in labelled if label[i] == c]
                for i in members:
                    if positions[0] < i < positions[-1] and label[i] < 0:
                        label[i] = c
        t = last + 1
    runs: list[list[int]] = []
    current: list[int] | None = None
    for t in range(count):
        if not valid[t]:
            continue
        c = int(label[t])
        if current is not None and c == current[0]:
            current[2] = int(offs[t, 1])
            continue
        if current is not None:
            runs.append(current)
        current = [c, int(offs[t, 0]), int(offs[t, 1])] if c >= 0 else None
    if current is not None:
        runs.append(current)
    out = []
    for c, first, last in runs:
        start, end = trim(text, first, last)
        if end > start:
            out.append({"start": start, "end": end, "label": names[c]})
    return out


def paragraphs(text: str) -> list[tuple[int, int]]:
    out, position = [], 0
    for block in PARAGRAPH_BREAK.split(text):
        if block and not re.fullmatch(r"\s*", block):
            out.append((position, position + len(block)))
        position += len(block)
    return out


def _mean_at_least(
    first: np.ndarray, offsets: np.ndarray, a: int, b: int, threshold: float
) -> bool:
    inside = (offsets[:, 0] >= a) & (offsets[:, 1] <= b)
    return bool(inside.any() and first[inside].mean() >= threshold)


def evidence_spans(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    names: Sequence[str],
    text: str,
    threshold: float,
    decoding: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evidence spans of the first label: ``para`` keeps paragraphs, ``sentfill`` sentences (a paragraph
    when at least half of its sentences pass), ``sent`` sentences, anything else the word decoder; then
    spans at most ``gap`` characters apart merge and spans shorter than ``min_chars`` drop.
    """
    first = probabilities[:, 0]
    kind = decoding["name"]
    spans: list[list[int]]
    if kind == "para":
        spans = [
            [a, b]
            for a, b in paragraphs(text)
            if _mean_at_least(first, offsets, a, b, threshold)
        ]
    elif kind in ("sentfill", "sent"):
        sentences = [(m.start(), m.end()) for m in SENTENCE.finditer(text)]
        keep = [_mean_at_least(first, offsets, a, b, threshold) for a, b in sentences]
        if kind == "sent":
            spans = [list(s) for s, k in zip(sentences, keep, strict=True) if k]
        else:
            spans = []
            for a, b in paragraphs(text):
                inside = [i for i, (x, y) in enumerate(sentences) if x >= a and y <= b]
                if inside and sum(keep[i] for i in inside) * 2 >= len(inside):
                    spans.append([a, b])
                else:
                    spans += [list(sentences[i]) for i in inside if keep[i]]
    else:
        spans = (
            [
                [s["start"], s["end"]]
                for s in decode(first[:, None], offsets, names[:1], text, threshold)
            ]
            if first.max() > threshold
            else []
        )
    spans.sort()
    merged: list[list[int]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= decoding.get("gap", 0):
            merged[-1][1] = max(merged[-1][1], span[1])
        else:
            merged.append(list(span))
    return [
        {"start": a, "end": b, "label": names[0]}
        for a, b in merged
        if b - a >= decoding.get("min_chars", 0)
    ]
