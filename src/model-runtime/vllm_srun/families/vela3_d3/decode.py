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

These are the Vela 2.0 decoder's rules (``vela2.words.decode_spans``); only
the unit class differs, and URL tails are not trimmed.

Evidence questions decode by sentence or paragraph (``evidence_spans``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..vela2.words import Dialect, decode_spans

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
# The released decoder (``span_decode_v2``): its own unit class, and no URL-tail trim.
DIALECT = Dialect(UNIT, url_tails=False)
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
PARAGRAPH_BREAK = re.compile(r"(\n\s*\n|\n)")


def units(text: str) -> np.ndarray:
    """Character index -> unit index (-1 for whitespace)."""
    out = np.full(len(text) + 1, -1, dtype=np.int64)
    for index, match in enumerate(UNIT.finditer(text)):
        out[match.start() : match.end()] = index
    return out


def decode(
    probabilities: Any,
    offsets: Any,
    names: Sequence[str],
    text: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """``[{start, end, label}]`` (code-point offsets) from word x label probabilities.

    The Vela 2.0 decoder with this decoder's unit class and no URL-tail trim;
    words no character spans are left out first.
    """
    offs = np.asarray(offsets).reshape(-1, 2)
    kept = offs[:, 1] > offs[:, 0]
    if not kept.any():
        return []
    probs = np.asarray(probabilities, dtype=np.float32)
    return [
        {"start": span["start"], "end": span["end"], "label": span["label"]}
        for span in decode_spans(
            probs[kept],
            offs[kept],
            list(names),
            text,
            threshold,
            dialect=DIALECT,
        )
    ]


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
