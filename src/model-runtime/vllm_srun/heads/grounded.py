"""Grounded heads: spans of an answer that its context does not support (Vela Halu).

The package's operating point fixes the pair the model reads (``("User request:
{question}\\n\\n{context}", answer)``), the token threshold and its comparison,
and the input budget. The pair is framed as the tokenizer frames pairs; a pair
over the budget is rejected or, under ``truncate``, keeps the whole answer and
the prompt's first tokens (the tokenizer's ``only_first`` truncation). A span is
a run of answer tokens whose positive-label probability passes the threshold
(tokens without a source span neither extend nor end a run); its probability is
the run's maximum. Offsets are code points into the answer.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..text import bounds
from ..text.windows import InputTooLongError
from .task import (
    ClassifierHead,
    HeadOptions,
    Prepared,
    Rows,
    TaskHead,
    token_probabilities,
)
from .token import token_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROMPT_FIELDS = {"question", "context"}
PAIR_SIZE = 2
GROUNDED_LABELS = 2
GROUNDED_KEYS = {"context", "question", "answer"}
PAIR_PROBES = (("", "x"), ("Hello, world!", "中文 🚀"), ("<eos> a", "b <bos>"))


@dataclass(frozen=True)
class PairEnvelope:
    """The special tokens a tokenizer puts around and between the two sequences of a pair."""

    prefix: tuple[int, ...]
    middle: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.prefix) + len(self.middle) + len(self.suffix)

    def frame(self, first: Sequence[int], second: Sequence[int]) -> list[int]:
        return [*self.prefix, *first, *self.middle, *second, *self.suffix]

    @classmethod
    def of(cls, tokenizer: Any) -> PairEnvelope:
        """The tokenizer's pair framing; refused when it depends on the texts."""
        encoding = tokenizer.encode("a", "b", add_special_tokens=True)
        ids, sequence = list(encoding.ids), list(encoding.sequence_ids)
        first = [index for index, s in enumerate(sequence) if s == 0]
        second = [index for index, s in enumerate(sequence) if s == 1]
        if not first or not second:
            raise ValueError("the tokenizer does not encode pairs")
        envelope = cls(
            tuple(ids[: first[0]]),
            tuple(ids[first[-1] + 1 : second[0]]),
            tuple(ids[second[-1] + 1 :]),
        )
        for a, b in PAIR_PROBES:
            framed = envelope.frame(
                tokenizer.encode(a, add_special_tokens=False).ids,
                tokenizer.encode(b, add_special_tokens=False).ids,
            )
            if framed != tokenizer.encode(a, b, add_special_tokens=True).ids:
                raise ValueError("the tokenizer's pair framing is not a fixed envelope")
        return envelope


@dataclass(frozen=True)
class GroundingPolicy:
    """The pair template, token threshold and budget a grounded head was calibrated under."""

    prompt: str
    threshold: float
    strict: bool
    max_tokens: int
    positive: int

    @classmethod
    def parse(cls, document: dict[str, Any], labels: Sequence[str]) -> GroundingPolicy:
        pair = document.get("input_pair")
        if not (
            isinstance(pair, list) and len(pair) == PAIR_SIZE and pair[1] == "answer"
        ):
            raise ValueError(
                "the grounding policy must pair a prompt template with the answer"
            )
        fields = {name for _, name, _, _ in string.Formatter().parse(pair[0]) if name}
        if fields != PROMPT_FIELDS:
            raise ValueError(
                "the grounding prompt must name exactly {question} and {context}"
            )
        label2id = document.get("label2id") or {}
        if {labels[index]: index for index in range(len(labels))} != label2id:
            raise ValueError(
                "grounding policy labels differ from the classifier's labels"
            )
        if document.get("answer_offsets") not in (None, "Unicode code points"):
            raise ValueError("unsupported grounding offsets")
        comparison = document.get("threshold_comparison", "strictly_greater")
        if comparison not in ("strictly_greater", "greater_or_equal"):
            raise ValueError(f"unsupported grounding comparison {comparison!r}")
        positive = [index for index, label in enumerate(labels) if label != "supported"]
        if len(labels) != GROUNDED_LABELS or len(positive) != 1:
            raise ValueError("a grounded head has a supported and one positive label")
        return cls(
            pair[0],
            float(document["token_threshold"]),
            comparison == "strictly_greater",
            int(document["max_input_tokens"]),
            positive[0],
        )

    def passes(self, probability: float, threshold: float) -> bool:
        return probability > threshold if self.strict else probability >= threshold


@dataclass(frozen=True)
class GroundedState:
    answer: str
    offsets: tuple[tuple[int, int], ...]
    start: int
    threshold: float
    return_tokens: bool


def grounded_input(value: Any) -> tuple[str, str, str]:
    if (
        not isinstance(value, dict)
        or set(value) - GROUNDED_KEYS
        or "answer" not in value
    ):
        raise ValueError("a grounded input is {context, question, answer}")
    parts = [value.get(key, "") for key in ("context", "question", "answer")]
    if not all(isinstance(part, str) for part in parts):
        raise ValueError("grounded input fields are strings")
    if not parts[2]:
        raise ValueError("the answer is empty")
    return parts[0], parts[1], parts[2]


def answer_spans(
    offsets: Sequence[tuple[int, int]],
    probabilities: Sequence[float],
    passes: Callable[[float], bool],
) -> list[dict[str, Any]]:
    """Runs of answer tokens that pass, as ``{start, end, probability}`` (the run's maximum)."""
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for (start, end), probability in zip(offsets, probabilities, strict=True):
        if end <= start:
            continue
        if passes(probability):
            if current is None:
                current = {"start": start, "end": end, "probability": probability}
            else:
                current["end"] = max(current["end"], end)
                current["probability"] = max(current["probability"], probability)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    return spans


class GroundedHead(TaskHead):
    """Answer spans from a token classifier over ``(prompt, answer)`` pairs."""

    kind = "token"

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        layer: int,
        tokenizer: Any,
        pair: PairEnvelope,
        classifier: ClassifierHead,
        policy: GroundingPolicy,
    ):
        super().__init__(name, labels, layer)
        self.tokenizer = tokenizer
        self.pair = pair
        self.classifier = classifier
        self.policy = policy

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "labels": self.labels,
            "inputs": ("grounded",),
            "default_threshold": self.policy.threshold,
            "overflow": "reject",
            "reduction": None,
        }

    def prepare(self, value: Any, options: HeadOptions, identity: str) -> Prepared:
        context, question, answer = grounded_input(value)
        prompt = self.policy.prompt.format(question=question, context=context)
        # The answer is read whole or not at all; the prompt only as far as the budget the answer leaves.
        room = options.max_tokens - self.pair.size
        if bounds.surely_over(self.tokenizer, answer, room):
            raise InputTooLongError(
                options.max_tokens + 1, options.max_tokens, "the complete answer"
            )
        answered = bounds.read(self.tokenizer, answer, room + 1)
        second = answered.encoding
        first: list[int] = []
        prompted = answered
        if answered.complete:
            left = room - answered.tokens
            if options.overflow != "truncate" and bounds.surely_over(
                self.tokenizer, prompt, left
            ):
                raise InputTooLongError(
                    options.max_tokens + 1, options.max_tokens, "grounded pair"
                )
            prompted = bounds.read(self.tokenizer, prompt, left + 1)
            first = prompted.encoding.ids[: prompted.tokens]
        tokens = len(first) + answered.tokens + self.pair.size
        usage: dict[str, Any] = {
            "tokens": tokens,
            "processed_tokens": tokens,
            "truncated": False,
        }
        if not (answered.complete and prompted.complete):
            usage["tokens_lower_bound"] = True
        if tokens > options.max_tokens:
            if options.overflow != "truncate":
                raise InputTooLongError(tokens, options.max_tokens, "grounded pair")
            keep = options.max_tokens - answered.tokens - self.pair.size
            if keep < 0:
                raise InputTooLongError(
                    tokens, options.max_tokens, "the complete answer"
                )
            first = first[:keep]
            usage.update(processed_tokens=options.max_tokens, truncated=True)
        ids = self.pair.frame(first, second.ids)
        threshold = (
            self.policy.threshold if options.threshold is None else options.threshold
        )
        state = GroundedState(
            answer,
            tuple(map(tuple, second.offsets)),
            len(self.pair.prefix) + len(first) + len(self.pair.middle),
            threshold,
            options.return_tokens,
        )
        return Prepared(self.items([ids], identity), usage, state)

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        return token_probabilities(self.classifier, rows, sequences, self.layer)

    def result(self, prepared: Prepared, values: Sequence[Any]) -> dict[str, Any]:
        state: GroundedState = prepared.state
        rows: NDArray[np.float32] = values[0][
            state.start : state.start + len(state.offsets)
        ]
        positive = rows[:, self.policy.positive].tolist()
        spans = answer_spans(
            state.offsets, positive, lambda p: self.policy.passes(p, state.threshold)
        )
        label = self.labels[self.policy.positive]
        out: dict[str, Any] = {
            "spans": [
                {
                    **span,
                    "label": label,
                    "text": state.answer[span["start"] : span["end"]],
                }
                for span in spans
            ],
            "input": prepared.usage,
        }
        if state.return_tokens:
            out["tokens"] = token_rows(state.offsets, rows)
        return out
