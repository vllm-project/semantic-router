"""Span prompts of the Vela 3.0 d3 sizes: d3's own prompt over a target part, in word-aligned windows.

A window's prompt is the state (the target part's text; several parts as
``<role>:`` blocks), the question and one option line per label
(``<code>: <label>[: <description>]``). Targets over ``MAX_TARGET`` tokens
split into windows of at most ``WINDOW`` tokens, ``STRIDE`` apart, that start
and end on word boundaries.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import QuestionError
from ..decision3.prompt import render
from ..vela2.words import words_of

WINDOW, STRIDE, MAX_TARGET = 1800, 1536, 2048


@dataclass(frozen=True)
class Window:
    """One forward of a span question: its token IDs, the target words' first tokens and offsets, the option lines."""

    ids: tuple[int, ...]
    word_tokens: tuple[int, ...]
    word_offsets: tuple[tuple[int, int], ...]
    option_ranges: tuple[tuple[int, int], ...]


def windows(firsts: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Word-aligned token windows of at most ``WINDOW`` tokens, ``STRIDE`` apart.

    A window ends at the last word start before ``WINDOW`` tokens (or the
    target's end) and the next starts at the first word start ``STRIDE`` or
    more tokens in, as the released planner does, in O(n log n).
    """
    starts = sorted({int(x) for x in firsts})
    if not starts:
        return [(0, count)]
    out, a = [], starts[0]
    while True:
        limit = a + WINDOW
        if limit >= count:
            b = count
        else:
            last = bisect_left(starts, limit) - 1
            b = starts[last] if last >= 0 and starts[last] > a else limit
        out.append((a, b))
        if b >= count:
            break
        following = bisect_left(starts, a + STRIDE)
        a = (
            starts[following]
            if following < len(starts) and starts[following] < b
            else b
        )
    return out


class Renderer:
    """Renders a span question's windows with the base's tokenizer and answer codes."""

    def __init__(self, tokenizer: Any, codes: list[str]):
        self.tokenizer = tokenizer
        self.codes = codes

    @staticmethod
    def state(parts: list[tuple[str, str]], target: str, text: str) -> tuple[str, int]:
        if len(parts) == 1:
            return text, 0
        blocks, start, position = [], 0, 0
        for role, value in parts:
            header = f"{role}:\n"
            body = text if role == target else value
            if role == target:
                start = position + len(header)
            blocks.append(header + body)
            position += len(header) + len(body) + 2
        return "\n\n".join(blocks), start

    def window(
        self,
        parts: list[tuple[str, str]],
        target: str,
        question: str,
        labels: list[tuple[str, str]],
        text: str,
        shift: int,
    ) -> Window:
        state, start = self.state(parts, target, text)
        options = [
            f"{code}: {name if not description else name + ': ' + description}"
            for code, (name, description) in zip(self.codes, labels, strict=False)
        ]
        head = [
            "State:",
            state if state else "(empty)",
            "",
            "Question:",
            question,
            "",
            "Options:",
        ]
        user = "\n".join(
            head + options + ["", "Reply with only the code of the best option."]
        )
        rendered = render(user)
        user_start = rendered.index(user)
        text_start = user_start + len("State:\n") + start
        encoding = self.tokenizer.encode(rendered, add_special_tokens=False)
        offsets = np.asarray(encoding.offsets, dtype=np.int64).reshape(-1, 2)
        words = words_of(text, offsets - text_start)
        spans, firsts = words.offsets, words.first
        position = user_start + len("\n".join(head)) + 1
        ranges = []
        begins, ends = offsets[:, 0], offsets[:, 1]
        for line in options:
            inside = np.nonzero((begins < position + len(line)) & (ends > position))[0]
            ranges.append((int(inside[0]), int(inside[-1])))
            position += len(line) + 1
        return Window(
            ids=tuple(encoding.ids),
            word_tokens=tuple(int(t) for t in firsts),
            word_offsets=tuple((int(a) + shift, int(b) + shift) for a, b in spans),
            option_ranges=tuple(ranges),
        )

    def windows(
        self,
        parts: list[tuple[str, str]],
        target: str,
        question: str,
        labels: list[tuple[str, str]],
        refuse: Callable[[int], QuestionError | None] = lambda count: None,
    ) -> tuple[list[Window], int]:
        """The question's windows and its target's token count.

        ``refuse`` sees the target's token count before any window is
        rendered; the error it returns fails the question.
        """
        text = dict(parts)[target]
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        count = len(encoding.ids)
        refused = refuse(count)
        if refused is not None:
            raise refused
        if count <= MAX_TARGET:
            return [self.window(parts, target, question, labels, text, 0)], count
        offsets = np.asarray(encoding.offsets, dtype=np.int64).reshape(-1, 2)
        firsts = words_of(text, offsets).first
        out = []
        for a, b in windows(firsts, count):
            start, end = int(offsets[a][0]), int(offsets[b - 1][1])
            out.append(
                self.window(parts, target, question, labels, text[start:end], start)
            )
        return out, count
