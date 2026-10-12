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

from ..text.windows import (
    Envelope,
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
            "inputs": ("text",),
            "overflow": self.overflow,
            "window": self.window,
            "reduction": "max",
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

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        if self.pooling == "cls":
            pooled = rows.first(sequences, self.layer)
        else:
            pooled = rows.mean(sequences, self.layer)
        values = self.activate(self.classifier(pooled.float())).cpu()
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
