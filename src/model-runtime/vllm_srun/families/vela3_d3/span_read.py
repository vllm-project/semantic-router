"""Reading and answering span questions of the Vela 3.0 d3 sizes.

Each window is one forward of the base with the route's adapter on, under the
noncausal full-attention mask. The route's span head (``SpanHeadV2``) reads the
final-norm states of each target word's first token against the mean state of
each option line; a word read by several windows takes their mean logits.
Words over the threshold decode into spans (``decode.py``), and each span's
probability is the mean of its words' probabilities for its label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ...plugins.base import AdapterGate, ForwardBatch
from ..decision3.family import Item
from ..vela2.words import without_separators
from .decode import decode, evidence_spans
from .package import Route
from .span_render import Window


@dataclass(frozen=True)
class SpanItem(Item):
    """A span question: its route, labels, threshold and the windows it reads."""

    route: Route = "span"
    names: tuple[str, ...] = ()
    label_set: str | None = None
    threshold: float = 0.5
    evidence: dict[str, Any] | None = None
    text: str = ""
    windows: tuple[Window, ...] = field(default=())


class SpanReader:
    """Runs a span question's windows: its adapter on, the noncausal mask, its head over the gathered states."""

    def __init__(self, engine_model: Any, heads: Mapping[Route, Any]):
        self.engine_model = engine_model
        self.heads = heads

    def logits(self, item: SpanItem) -> tuple[np.ndarray, np.ndarray]:
        """The target words' offsets and their label logits, over every window that reads a word."""
        head = self.heads[item.route]
        read: list[Window] = []
        scores: list[np.ndarray] = []
        for window in item.windows:
            if not window.word_offsets:
                continue
            words = list(window.word_tokens)
            positions = words + [
                t for a, b in window.option_ranges for t in range(a, b + 1)
            ]
            length = len(window.ids)
            output = self.engine_model.forward(
                ForwardBatch(
                    input_ids=torch.tensor([window.ids]),
                    attention_mask=torch.ones((1, length), dtype=torch.long),
                    gather=torch.tensor([positions]),
                    query=torch.tensor([length - 1]),
                    lengths=[length],
                    adapter=AdapterGate(item.route),
                    noncausal=True,
                )
            )
            states = output.gathered[0]
            labels, at = [], len(words)
            for a, b in window.option_ranges:
                labels.append(states[at : at + b - a + 1].mean(0))
                at += b - a + 1
            with torch.inference_mode():
                logits = head(states[: len(words)], torch.stack(labels))
            read.append(window)
            scores.append(logits.float().cpu().numpy())
        return combine(read, scores, len(item.names))


def combine(
    windows: list[Window], logits: list[np.ndarray], labels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each target word's logits, averaged over the windows that read it, in text order."""
    seen: dict[tuple[int, int], list[Any]] = {}
    for window, scores in zip(windows, logits, strict=True):
        for index, word in enumerate(window.word_offsets):
            entry = seen.setdefault(word, [np.zeros(labels), 0])
            entry[0] += scores[index]
            entry[1] += 1
    keys = sorted(seen)
    offsets = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    values = (
        np.stack([seen[k][0] / seen[k][1] for k in keys])
        if keys
        else np.zeros((0, labels))
    )
    return offsets, values


def answer(item: SpanItem, offsets: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """``spans``, ``threshold`` and the most probable word's probability of one span question."""
    names = list(item.names)
    if len(offsets) == 0:
        return {"spans": [], "threshold": item.threshold, "noul": 0.0}
    probabilities = 1 / (1 + np.exp(-logits.astype(np.float64)))
    if not np.isfinite(probabilities).all():
        raise ValueError("the span head produced a non-finite score")
    threshold = item.threshold
    if item.evidence is not None:
        found = evidence_spans(
            probabilities, offsets, names, item.text, threshold, item.evidence
        )
    elif float(np.max(probabilities)) > threshold:
        found = decode(probabilities, offsets, names, item.text, threshold)
    else:
        found = []
    spans = []
    for span in found:
        label = names.index(span["label"])
        # The released span_answer's probability: the mean over every word the trimmed span overlaps.
        overlap = (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
        start, end = span["start"], span["end"]
        if item.evidence is None:
            start, end = without_separators(item.text, start, end)
            if end <= start:
                continue
        spans.append(
            {
                "label": span["label"],
                "start": start,
                "end": end,
                "text": item.text[start:end],
                "probability": (
                    float(probabilities[overlap, label].mean())
                    if overlap.any()
                    else 0.0
                ),
            }
        )
    return {
        "spans": spans,
        "threshold": threshold,
        "noul": float(np.max(probabilities)),
    }
