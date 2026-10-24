"""Composed images (System One ``assets``) and Box answers for state-once Decision 3 packages.

A request's ``assets`` map reference strings to images. In the state a
reference is hoisted: each use becomes an image after the request's own
``images`` (one per use, in reading order) and its text the label
``[image k]`` (``HOISTED_LABEL``). In a question's instructions or options a
reference stays inline: the image's placeholder sits in that question's block
only, so no other question sees it. Only string values are searched (object
keys are never replaced); a value equal to a reference is replaced whole,
otherwise references are replaced left to right without overlap, the longest
first when several start at one position. A whole-value image inside a JSON
value replaces the JSON string, quotes included.

A Box question's block ends with four edge markers after the generation
prompt (``top:``, ``bottom:``, ``left:``, ``right:``; a track moment adds
``visible:`` first). Each marker's last token is read by the package's
255-code readout: an edge over the first ``BINS`` codes (bin k's centre
``(k + 0.5) / BINS``), ``visible`` over Noul's ``false`` / ``true``. The edge
is the expected bin centre; the confidence is the probability that a box
drawn from the four edge distributions has IoU >= 0.5 with the reported box,
over fixed low-discrepancy draws. The text is the package's own prompt
format, byte for byte, so the readout sees the blocks it reads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from . import prompt

TEXT = "text"
IMAGE = "image"
MAX_REFERENCE_BYTES = 1024
MAX_REFERENCES_BYTES = 65536
HOISTED_LABEL = "[image {k}]"
BINS = 128
EDGES = ("top", "bottom", "left", "right")
PRESENCE = "visible"
IOU_HIT = 0.5
CONFIDENCE_SAMPLES = 512
_OPEN, _CLOSE, _WHOLE = "\ue000", "\ue001", "\ue002"

Part = tuple[str, Any]


def check_references(references: Sequence[str]) -> list[str]:
    """Reference limits; the references longest first (the match order at one position)."""
    refs = list(references)
    total = 0
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("an asset reference must be a non-blank string")
        size = len(ref.encode("utf-8"))
        if size > MAX_REFERENCE_BYTES:
            raise ValueError(
                f"an asset reference has {size} bytes, over {MAX_REFERENCE_BYTES}"
            )
        if any(mark in ref for mark in (_OPEN, _CLOSE, _WHOLE)):
            raise ValueError("an asset reference contains a reserved character")
        total += size
    if total > MAX_REFERENCES_BYTES:
        raise ValueError(
            f"asset references have {total} bytes, over {MAX_REFERENCES_BYTES}"
        )
    return sorted(refs, key=lambda ref: (-len(ref), ref))


def split_string(text: str, refs: Sequence[str]) -> list[Part]:
    if text in refs:
        return [(IMAGE, text)]
    parts: list[Part] = []
    at = start = 0
    while at < len(text):
        hit = next((ref for ref in refs if text.startswith(ref, at)), None)
        if hit is None:
            at += 1
            continue
        if at > start:
            parts.append((TEXT, text[start:at]))
        parts.append((IMAGE, hit))
        at += len(hit)
        start = at
    if start < len(text):
        parts.append((TEXT, text[start:]))
    return parts


def _mark(value: Any, refs: Sequence[str], order: list[str]) -> Any:
    if isinstance(value, str):
        parts = split_string(value, refs)
        if not any(kind == IMAGE for kind, _ in parts):
            return value
        if len(parts) == 1:
            order.append(parts[0][1])
            return f"{_WHOLE}{len(order) - 1}{_WHOLE}"
        out = []
        for kind, item in parts:
            if kind == TEXT:
                out.append(item)
            else:
                order.append(item)
                out.append(f"{_OPEN}{len(order) - 1}{_CLOSE}")
        return "".join(out)
    if isinstance(value, Mapping):
        return {key: _mark(item, refs, order) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mark(item, refs, order) for item in value]
    return value


def _unmark(text: str, order: Sequence[str]) -> list[Part]:
    parts: list[Part] = []
    buffer: list[str] = []
    at = 0
    while at < len(text):
        char = text[at]
        if char in (_OPEN, _WHOLE):
            close = _CLOSE if char == _OPEN else _WHOLE
            end = text.index(close, at + 1)
            index = int(text[at + 1 : end])
            at = end + 1
            if (
                char == _WHOLE
                and buffer
                and buffer[-1].endswith('"')
                and text.startswith('"', at)
            ):
                buffer[-1] = buffer[-1][:-1]
                at += 1
            if buffer:
                parts.append((TEXT, "".join(buffer)))
                buffer = []
            parts.append((IMAGE, order[index]))
            continue
        buffer.append(char)
        at += 1
    if buffer:
        parts.append((TEXT, "".join(buffer)))
    return parts


def describe_parts(value: Any, refs: Sequence[str]) -> list[Part]:
    """``prompt.describe(value)`` with references as image parts."""
    if isinstance(value, str):
        return split_string(value, refs)
    order: list[str] = []
    marked = _mark(value, refs, order)
    if not order:
        return [(TEXT, prompt.describe(value))]
    return _unmark(json.dumps(marked, ensure_ascii=False), order)


def hoist_state(state: Any, refs: Sequence[str], front: int) -> tuple[Any, list[str]]:
    """``(state with each use written '[image k]', the hoisted references, one per use, in order)``."""
    hoisted: list[str] = []

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            parts = split_string(value, refs)
            if not any(kind == IMAGE for kind, _ in parts):
                return value
            out = []
            for kind, item in parts:
                if kind == TEXT:
                    out.append(item)
                else:
                    hoisted.append(item)
                    out.append(HOISTED_LABEL.format(k=front + len(hoisted)))
            return "".join(out)
        if isinstance(value, Mapping):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        return value

    return walk(state), hoisted


def to_text(parts: Sequence[Part]) -> tuple[str, list[str]]:
    """``(text with one image placeholder per image part, the references in order)``."""
    out, refs = [], []
    for kind, item in parts:
        if kind == TEXT:
            out.append(item)
        else:
            out.append(prompt.IMAGE_PLACEHOLDER)
            refs.append(item)
    return "".join(out), refs


def question_block(
    kind: str,
    instructions: Any,
    criteria: Any,
    codes: Sequence[str],
    refs: Sequence[str],
) -> tuple[str, list[str]]:
    """The question block's text (``prompt.question_block`` with inline images) and its images' references."""
    parts: list[Part] = [(TEXT, "Question:\n")]
    parts += describe_parts(instructions or prompt.DEFAULT_INSTRUCTIONS, refs)
    parts.append((TEXT, "\n\nOptions:"))
    if kind == "choice":
        options = [
            (
                [(TEXT, key)]
                if value is None
                else [(TEXT, f"{key}: "), *describe_parts(value, refs)]
            )
            for key, value in criteria.items()
        ]
    else:
        given = criteria or {}
        options = [
            describe_parts(given.get("false") or prompt.NOUL_DESCRIPTIONS[0], refs),
            describe_parts(given.get("true") or prompt.NOUL_DESCRIPTIONS[1], refs),
        ]
    for code, option in zip(codes, options, strict=False):
        parts.append((TEXT, f"\n{code}: "))
        parts += option
    parts.append((TEXT, "\n\nReply with only the code of the best option."))
    parts.append((TEXT, GENERATION))
    return to_text(parts)


GENERATION = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_seconds(value: float) -> str:
    return f"{float(value):.1f}"


def markers(presence: bool = False) -> tuple[str, ...]:
    return ((PRESENCE,) if presence else ()) + EDGES


def suffix(presence: bool = False) -> str:
    return "\n".join(f"{name}:" for name in markers(presence))


def tail_text(
    codes: Sequence[str], at: float | None = None, presence: bool = False
) -> str:
    last = codes[BINS - 1]
    when = "" if at is None else f" in the video frame at {format_seconds(at)} seconds"
    lines = [
        "",
        "",
        "Box:",
        f"Find the target{when}. Give its top, bottom, left and right edges, each as a code on a {BINS}-step "
        f"scale across the image: {codes[0]} is the top or left side, {last} the bottom or right side.",
    ]
    if presence:
        lines.append(
            f"Also say whether the target is visible then: {codes[0]} no, {codes[1]} yes."
        )
    lines += ["", "Reply with the edge codes."]
    return "\n".join(lines)


def box_block(
    instructions: Any,
    codes: Sequence[str],
    refs: Sequence[str],
    at: float | None = None,
    presence: bool = False,
) -> tuple[str, list[str]]:
    """A Box question's block: its instructions, the Box lines, the generation prompt and the edge markers."""
    parts: list[Part] = [(TEXT, "Question:\n")]
    parts += describe_parts(instructions, refs)
    parts.append((TEXT, tail_text(codes, at, presence)))
    parts.append((TEXT, GENERATION + suffix(presence)))
    return to_text(parts)


def marker_offsets(tokenizer: Any, presence: bool) -> list[int]:
    """Offsets of each marker's colon token from the block's last token (the suffix tokenizes on its own)."""
    text = suffix(presence)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    stops = [stop for _, stop in encoding.offsets]
    ends, at = [], 0
    for name in markers(presence):
        at = text.index(f"{name}:", at) + len(name) + 1
        ends.append(at)
    from_end = stops[::-1]
    out = []
    for end in ends:
        if end not in from_end:
            raise ValueError("an edge marker's colon is not the end of a token")
        out.append(from_end.index(end))
    return out


def expected(probabilities: Sequence[float]) -> float:
    total = float(sum(probabilities))
    count = len(probabilities)
    return sum(p * (k + 0.5) / count for k, p in enumerate(probabilities)) / total


def order_edges(a: float, b: float, bins: int = BINS) -> tuple[float, float]:
    if a < b:
        return a, b
    mid = (a + b) / 2
    half = 0.5 / bins
    lo, hi = max(0.0, mid - half), min(1.0, mid + half)
    if lo >= hi:
        lo, hi = (max(0.0, hi - 2 * half), hi) if hi >= 1.0 else (lo, lo + 2 * half)
    return lo, hi


def decode(edges: Sequence[Sequence[float]]) -> dict[str, float]:
    top, bottom, left, right = (expected(p) for p in edges)
    top, bottom = order_edges(top, bottom)
    left, right = order_edges(left, right)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _r4_points(n: int) -> list[tuple[float, float, float, float]]:
    phi = 1.0
    for _ in range(64):
        phi = (1 + phi) ** (1 / 5)
    alphas = [(1 / phi) ** (d + 1) % 1 for d in range(4)]
    return [tuple(((0.5 + alphas[d] * (i + 1)) % 1.0) for d in range(4)) for i in range(n)]  # type: ignore[misc]


_POINTS = _r4_points(CONFIDENCE_SAMPLES)


def confidence(edges: Sequence[Sequence[float]], box: Mapping[str, float]) -> float:
    import numpy as np

    probabilities = np.asarray(edges, dtype=np.float64)
    cdf = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    bins = probabilities.shape[1]
    points = np.asarray(_POINTS, dtype=np.float64)
    draws = np.empty_like(points)
    for d in range(4):
        index = np.minimum(np.searchsorted(cdf[d], points[:, d], side="left"), bins - 1)
        draws[:, d] = (index + 0.5) / bins
    top, bottom, left, right = draws.T
    valid = (top < bottom) & (left < right)
    inter_h = np.clip(
        np.minimum(bottom, box["bottom"]) - np.maximum(top, box["top"]), 0.0, None
    )
    inter_w = np.clip(
        np.minimum(right, box["right"]) - np.maximum(left, box["left"]), 0.0, None
    )
    inter = inter_h * inter_w
    area = np.clip(bottom - top, 0.0, None) * np.clip(right - left, 0.0, None)
    reported = max(0.0, box["bottom"] - box["top"]) * max(
        0.0, box["right"] - box["left"]
    )
    union = area + reported - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return float(np.count_nonzero(valid & (iou >= IOU_HIT))) / len(points)


def calibrate(value: float, table: Sequence[Sequence[float]] | None) -> float:
    """The package's monotone (isotonic) confidence map: piecewise linear through its knots, or the raw value."""
    if not table:
        return value
    xs, ys = [float(k[0]) for k in table], [float(k[1]) for k in table]
    if value <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in pairwise(zip(xs, ys, strict=True)):
        if value <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return ys[-1]


def box_answer(
    edges: Sequence[Sequence[float]],
    table: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if len(edges) != len(EDGES) or any(
        len(p) != BINS or any(not math.isfinite(v) or v < 0 for v in p) or sum(p) <= 0
        for p in edges
    ):
        raise ValueError("a Box needs four finite edge distributions over the bins")
    box = decode(edges)
    return {
        "type": "box",
        "box": box,
        "confidence": calibrate(confidence(edges, box), table),
    }
