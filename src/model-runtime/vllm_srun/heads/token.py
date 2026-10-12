"""Token heads: a label per token, decoded from BIO tags into labelled spans.

Decoding follows the legacy router's token classifier exactly: each token
takes its first arg-max label and that label's probability; ``B-X`` opens an
entity, ``I-X`` continues an open ``X`` entity or opens one (models need not
emit ``B-``), anything else closes it; an entity's probability is the mean of
its tokens' and its span is trimmed of surrounding whitespace. Offsets are
Unicode code points into the text the model read, end exclusive. Windows are
merged per token before decoding (``windows.merge_token_windows``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..text.windows import (
    Encoded,
    Envelope,
    Window,
    encode,
    fit_prefix,
    merge_token_windows,
    over_budget,
    plan_windows,
)
from .sequence import text_input
from .task import (
    ClassifierHead,
    HeadOptions,
    Prepared,
    Rows,
    TaskHead,
    token_probabilities,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Rust's char::is_whitespace (Unicode White_Space); Python's isspace also strips U+001C-U+001F.
WHITESPACE = "".join(
    map(
        chr,
        [
            *range(0x09, 0x0E),
            0x20,
            0x85,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ],
    )
)


@dataclass(frozen=True)
class TokenState:
    """What decoding needs: the text the model read, its windows, whether to list token rows."""

    encoded: Encoded
    windows: list[Window] | None
    return_tokens: bool


@dataclass
class Entity:
    label: str
    start: int
    end: int
    probability: float
    tokens: int = 1

    def extend(self, end: int, probability: float) -> None:
        self.end = end
        self.tokens += 1
        self.probability += (probability - self.probability) / self.tokens


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """``[start, end)`` without surrounding whitespace; unchanged when it is all whitespace."""
    piece = text[start:end]
    if not piece.strip(WHITESPACE):
        return start, end
    lead = len(piece) - len(piece.lstrip(WHITESPACE))
    tail = len(piece) - len(piece.rstrip(WHITESPACE))
    return start + lead, end - tail


def bio_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    probabilities: NDArray[np.float32],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Labelled spans from per-token label probabilities (``[tokens, labels]``)."""
    best = np.argmax(probabilities, axis=-1)
    entities: list[Entity] = []
    current: Entity | None = None
    for (start, end), index, row in zip(
        offsets, best.tolist(), probabilities, strict=True
    ):
        tag, probability = labels[index], float(row[index])
        kind = tag[2:] if tag.startswith(("B-", "I-")) else None
        if tag.startswith("I-") and current is not None and current.label == kind:
            current.extend(end, probability)
            continue
        if current is not None:
            entities.append(current)
            current = None
        if kind is not None:
            current = Entity(kind, start, end, probability)
    if current is not None:
        entities.append(current)
    spans = []
    for entity in entities:
        start, end = trim(text, entity.start, entity.end)
        if end > start:
            spans.append(
                {
                    "label": entity.label,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "probability": entity.probability,
                }
            )
    return spans


def token_rows(
    offsets: Sequence[tuple[int, int]], probabilities: NDArray[np.float32]
) -> list[dict[str, Any]]:
    return [
        {"start": start, "end": end, "probabilities": row.tolist()}
        for (start, end), row in zip(offsets, probabilities, strict=True)
    ]


class TokenHead(TaskHead):
    """BIO spans from a per-token classifier."""

    kind = "token"

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        layer: int,
        tokenizer: Any,
        envelope: Envelope,
        classifier: ClassifierHead,
        *,
        overflow: str = "reject",
        window: tuple[int, int] | None = None,
    ):
        super().__init__(name, labels, layer)
        if not any(label.startswith(("B-", "I-")) for label in labels):
            raise ValueError("a token head needs BIO labels")
        self.tokenizer = tokenizer
        self.envelope = envelope
        self.classifier = classifier
        self.overflow = overflow
        self.window = window

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "labels": self.labels,
            "inputs": ("text",),
            "overflow": self.overflow,
            "window": self.window,
            "reduction": "span_union",
        }

    def prepare(self, value: Any, options: HeadOptions, identity: str) -> Prepared:
        encoded = encode(
            self.tokenizer,
            self.envelope,
            text_input(value),
            options.max_tokens,
            options.overflow,
        )
        tokens = encoded.tokens
        usage = encoded.usage()
        if options.overflow == "window":
            if tokens > options.max_tokens:
                raise over_budget(tokens, options.max_tokens, options.overflow)
            assert options.window is not None
            windows = plan_windows(len(encoded.content), self.envelope, *options.window)
            usage["windows"] = len(windows)
            ids = [encoded.framed(w.start, w.end) for w in windows]
            state = TokenState(encoded, windows, options.return_tokens)
            return Prepared(self.items(ids, identity), usage, state)
        if tokens > options.max_tokens and options.overflow != "truncate":
            raise over_budget(tokens, options.max_tokens, options.overflow)
        read, cut = fit_prefix(self.tokenizer, encoded, options.max_tokens)
        usage.update(processed_tokens=read.tokens, truncated=cut)
        state = TokenState(read, None, options.return_tokens)
        return Prepared(self.items([read.framed()], identity), usage, state)

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        return token_probabilities(self.classifier, rows, sequences, self.layer)

    def result(self, prepared: Prepared, values: Sequence[Any]) -> dict[str, Any]:
        state: TokenState = prepared.state
        encoded, prefix = state.encoded, len(self.envelope.prefix)
        offsets = encoded.offsets
        if state.windows is None:
            probabilities = values[0][prefix : prefix + len(encoded.content)]
            # The legacy decoder drops tokens without a source span, as it drops special tokens.
            keep = [index for index, span in enumerate(offsets) if span != (0, 0)]
            spans = bio_spans(
                encoded.text,
                [offsets[index] for index in keep],
                probabilities[keep],
                self.labels,
            )
        else:
            probabilities = merge_token_windows(
                state.windows, values, prefix, len(encoded.content)
            )
            spans = bio_spans(encoded.text, offsets, probabilities, self.labels)
        out: dict[str, Any] = {"spans": spans, "input": prepared.usage}
        if state.return_tokens:
            out["tokens"] = token_rows(offsets, probabilities)
        return out
