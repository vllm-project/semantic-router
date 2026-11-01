"""Pooled sequence heads: CLS or mean pooling, the ModernBERT classifier, softmax over labels.

An input is read whole (``reject``), cut to the budget (``truncate``) or in
windows; windowed results keep every window and report the per-label maximum,
the label being its arg-max (first label on ties, as the legacy binding).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..text import bounds
from ..text.pairs import PairEnvelope
from ..text.windows import (
    Envelope,
    InputTooLongError,
    encode,
    over_budget,
    plan_windows,
    reduce_max,
    truncate,
)
from .task import ClassifierHead, HeadOptions, Prepared, Rows, TaskHead

POOLING = ("cls", "mean")


def text_input(value: Any) -> str:
    """A plain-text input: a string or ``{text}``."""
    if isinstance(value, dict) and set(value) == {"text"}:
        value = value["text"]
    if not isinstance(value, str):
        raise ValueError("this head reads plain text inputs")
    if not value:
        raise ValueError("input text is empty")
    return value


class SequenceHead(TaskHead):
    """Softmax over labels from a pooled row."""

    kind = "sequence"
    value_key = "probabilities"

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        layer: int,
        tokenizer: Any,
        envelope: Envelope,
        classifier: ClassifierHead,
        pooling: str,
        *,
        overflow: str = "reject",
        window: tuple[int, int] | None = None,
    ):
        super().__init__(name, labels, layer)
        if pooling not in POOLING:
            raise ValueError(f"unsupported classifier pooling {pooling!r}")
        self.tokenizer = tokenizer
        self.envelope = envelope
        self.classifier = classifier
        self.pooling = pooling
        self.overflow = overflow
        self.window = window

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "labels": self.labels,
            "inputs": ("text", "pair") if self.kind == "sequence" else ("text",),
            "overflow": self.overflow,
            "window": self.window,
            "reduction": "max",
        }

    def prepare(self, value: Any, options: HeadOptions, identity: str) -> Prepared:
        if self.kind == "sequence" and isinstance(value, dict) and "text_pair" in value:
            return self.prepare_pair(value, options, identity)
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
            size, overlap = options.window
            windows = plan_windows(len(encoded.content), self.envelope, size, overlap)
            rows = [encoded.framed(w.start, w.end) for w in windows]
            usage["windows"] = len(windows)
            return Prepared(self.items(rows, identity), usage, windows)
        if tokens <= options.max_tokens:
            return Prepared(self.items([encoded.framed()], identity), usage)
        if options.overflow != "truncate":
            raise over_budget(tokens, options.max_tokens, options.overflow)
        ids = truncate(encoded, options.max_tokens)
        usage.update(processed_tokens=len(ids), truncated=True)
        return Prepared(self.items([ids], identity), usage)

    def prepare_pair(self, value: Any, options: HeadOptions, identity: str) -> Prepared:
        """Read a complete hypothesis, truncating only the premise when requested."""
        if not isinstance(value, dict) or set(value) != {"text", "text_pair"}:
            raise ValueError("a pair is {text, text_pair}")
        first, second = value["text"], value["text_pair"]
        if any(
            not isinstance(text, str) or not text.strip() for text in (first, second)
        ):
            raise ValueError("pair texts must be non-blank strings")
        if options.overflow == "window":
            raise ValueError("sequence pairs are not read in windows")
        envelope = PairEnvelope.of(self.tokenizer)
        room = options.max_tokens - envelope.size - 1
        if bounds.surely_over(self.tokenizer, second, room):
            raise InputTooLongError(
                options.max_tokens + 1, options.max_tokens, "the complete hypothesis"
            )
        hypothesis = bounds.read(self.tokenizer, second, room + 1)
        if not hypothesis.complete or hypothesis.tokens > room:
            raise InputTooLongError(
                hypothesis.tokens + envelope.size + 1,
                options.max_tokens,
                "the complete hypothesis",
            )
        left = options.max_tokens - hypothesis.tokens - envelope.size
        if options.overflow != "truncate" and bounds.surely_over(
            self.tokenizer, first, left
        ):
            raise InputTooLongError(
                options.max_tokens + 1, options.max_tokens, "sequence pair"
            )
        premise = bounds.read(self.tokenizer, first, left + 1)
        tokens = premise.tokens + hypothesis.tokens + envelope.size
        if tokens > options.max_tokens and options.overflow != "truncate":
            raise InputTooLongError(tokens, options.max_tokens, "sequence pair")
        ids = envelope.frame(premise.encoding.ids[:left], hypothesis.encoding.ids)
        usage = {
            "tokens": tokens,
            "processed_tokens": len(ids),
            "truncated": len(ids) < tokens,
        }
        if not premise.complete:
            usage["tokens_lower_bound"] = True
        return Prepared(self.items([ids], identity), usage)

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def logits(self, rows: Rows, sequences: Sequence[int]) -> torch.Tensor:
        """The classifier logits, before this head's activation."""
        if self.pooling == "cls":
            pooled = rows.first(sequences, self.layer)
        else:
            pooled = rows.mean(sequences, self.layer)
        logits: torch.Tensor = self.classifier(pooled.float())
        return logits

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        values = self.activate(self.logits(rows, sequences)).cpu()
        return [tuple(row) for row in values.tolist()]

    def result(self, prepared: Prepared, values: Sequence[Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"input": prepared.usage}
        if prepared.state is None:
            reduced = list(values[0])
        else:
            out["windows"] = [
                {"start": w.start, "end": w.end, self.value_key: list(v)}
                for w, v in zip(prepared.state, values, strict=True)
            ]
            reduced = reduce_max(values)
        out[self.value_key] = reduced
        self.annotate(out, reduced)
        return out

    def annotate(self, out: dict[str, Any], reduced: Sequence[float]) -> None:
        out["label"] = self.labels[int(np.argmax(reduced))]
