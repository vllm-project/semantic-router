"""Span questions of the Vela 3.0 d3 sizes (the released ``Vela3MiniModel.span_answer`` and ``vela3_span.py``).

A span question is one forward pass of the base with a span adapter on, under
the noncausal full-attention mask, over d3's own prompt: the state (the
target part's text; several parts as ``<role>:`` blocks), the question and
one option line per label (``<code>: <label>[: <description>]``). A span head
reads the final-norm states of each target word's first token against the
mean state of each option line, and words over the threshold decode into
spans (``decode.py``). Targets over ``MAX_TARGET`` tokens split into
word-aligned windows whose word scores average.

Routing (``route``): router label sets (PII, halu, toxic), evidence and answer
labels and questions, and wordings that name no entity run on the span
adapter and head; relation questions on the relation adapter and head; open
entity labels on the NER adapter and head. The questions follow the runtime's
contract: ``over`` names the target part, ``threshold`` overrides the
package's, and answers are ``spans`` with each span's mean word probability.
The windows read the whole target, so a question that sets
``require_full_input`` reads its input completely.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import QuestionError
from ...systemone import require_full_input
from ..decision3.family import Item
from ..decision3.prompt import render
from ..vela2.words import words_of
from ..vela3.request import expand_preset, threshold_of
from ..vela3.routing import RELATION_WORDING
from .decode import decode, evidence_spans

WINDOW, STRIDE, MAX_TARGET = 1800, 1536, 2048
MAX_LABELS = 255
FIELDS = frozenset(
    {
        "type",
        "instructions",
        "criteria",
        "labels",
        "threshold",
        "head",
        "over",
        "overflow",
        "require_full_input",
    }
)
# A span question reads its whole target, in windows: the one overflow it takes.
WINDOWED = "window"
ANSWERING = re.compile(
    r"\b(?:answers?|evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
)
ENTITY = re.compile(r"\b(?:entity|entities|types?|category|categories|named)\b")
# The adapter and head each route runs (the heads by the same names).
ADAPTERS = {"span": "span", "ner": "ner", "rel": "relation"}


def invalid(message: str, reason: str | None = None) -> QuestionError:
    return QuestionError("invalid_question", message, reason)


def is_span(question: Any) -> bool:
    """A span question, or a preset (which is one)."""
    return isinstance(question, dict) and (
        question.get("type") == "span" or "preset" in question
    )


@dataclass(frozen=True)
class Window:
    """One forward of a span question: its token IDs, the target words' first tokens and offsets, the option lines."""

    ids: tuple[int, ...]
    word_tokens: tuple[int, ...]
    word_offsets: tuple[tuple[int, int], ...]
    option_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SpanItem(Item):
    """A span question: its route, labels, threshold and the windows it reads."""

    route: str = "span"
    names: tuple[str, ...] = ()
    label_set: str | None = None
    threshold: float = 0.5
    evidence: dict[str, Any] | None = None
    text: str = ""
    windows: tuple[Window, ...] = field(default=())


# ------------------------------------------------------------------ requests


def parts_of(state: Any) -> list[tuple[str, str]]:
    if isinstance(state, str):
        return [("user", state)]
    if (
        isinstance(state, dict)
        and state
        and all(isinstance(v, str) for v in state.values())
    ):
        return list(state.items())
    raise invalid("span questions need a text state or an object of text parts")


def labels_of(question: dict[str, Any]) -> list[tuple[str, str]]:
    criteria, listed = question.get("criteria"), question.get("labels")
    if criteria is not None and listed is not None:
        raise invalid("use criteria or labels, not both")
    if isinstance(criteria, dict):
        labels = [(str(k), "" if v is None else str(v)) for k, v in criteria.items()]
    elif isinstance(criteria, list):
        labels = [(str(k), "") for k in criteria]
    elif isinstance(listed, list) and all(
        isinstance(o, dict) and "key" in o for o in listed
    ):
        labels = [
            (
                str(o["key"]),
                "" if o.get("description") is None else str(o["description"]),
            )
            for o in listed
        ]
    else:
        raise invalid(
            "span questions need criteria: {label: description or null} or a list of labels"
        )
    if not 1 <= len(labels) <= MAX_LABELS:
        raise invalid(f"a span question needs 1 to {MAX_LABELS} labels")
    names = [n for n, _ in labels]
    if len(set(names)) != len(names):
        raise invalid("span labels must be distinct")
    return labels


# ------------------------------------------------------------------ routing


class Policy:
    """A package's span routing, thresholds and label sets (``vela3_config.json`` ``span``)."""

    def __init__(self, span: dict[str, Any], heads: set[str]):
        self.span = span
        self.ner = "ner" in heads
        self.relation = "rel" in heads

    def label_set(self, names: list[str]) -> str | None:
        hit = next(
            (
                k
                for k, members in self.span["router_label_sets"].items()
                if set(names) <= set(members)
            ),
            None,
        )
        evidence = self.span.get("evidence")
        if (
            hit is None
            and evidence
            and {n.strip().lower() for n in names}
            <= {x.lower() for x in evidence["labels"]}
        ):
            return "evidence"
        return hit

    def route(
        self, label_set: str | None, names: list[str], instructions: str | None
    ) -> str:
        if not self.ner:
            return "span"
        if label_set is not None:
            return "span"
        if {n.strip().lower() for n in names} & {"answer", "answers"}:
            return "span"
        text = " ".join(str(instructions or "").split()).lower()
        if not text:
            return "ner"
        if RELATION_WORDING.search(text):
            return "rel" if self.relation else "span"
        if ANSWERING.search(text):
            return "span"
        if ENTITY.search(text):
            return "ner"
        return "span"

    def threshold(
        self,
        route: str,
        label_set: str | None,
        labels: list[tuple[str, str]],
        target_tokens: int,
    ) -> float:
        span = self.span
        described = any(d for _, d in labels)
        if route == "ner":
            ner = span["ner"]
            return float(ner["threshold"] if described else ner["threshold_names_only"])
        if label_set == "pii":
            bucket = next(
                b for b in span["pii_buckets"] if target_tokens <= b["max_tokens"]
            )
            value = float(bucket["threshold"])
        elif label_set == "evidence":
            value = float(span["evidence"]["threshold"])
        elif label_set in span["thresholds_by_set"]:
            value = float(span["thresholds_by_set"][label_set])
        else:
            value = float(span["threshold"])
        relation = span.get("relation") or {}
        dev = relation.get(
            "threshold_dev_pair" if len(labels) == 1 else "threshold_dev_multi"
        )
        if route == "rel" and isinstance(dev, int | float):
            return float(dev)
        if (
            route == "span"
            and label_set is None
            and not described
            and span.get("threshold_names_only") is not None
        ):
            return float(span["threshold_names_only"])
        return value


# ------------------------------------------------------------------ rendering


def windows(firsts: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Word-aligned token windows of at most ``WINDOW`` tokens, ``STRIDE`` apart."""
    starts = sorted({int(x) for x in firsts})
    if not starts:
        return [(0, count)]
    out, a = [], starts[0]
    while True:
        limit = a + WINDOW
        later = [s for s in starts if a < s < limit]
        b = count if limit >= count else (max(later) if later else min(count, limit))
        out.append((a, b))
        if b >= count:
            break
        candidates = [s for s in starts if a + STRIDE <= s < b]
        a = candidates[0] if candidates else b
    return out


class Renderer:
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
    ) -> tuple[list[Window], int]:
        """The question's windows and its target's token count."""
        text = dict(parts)[target]
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        count = len(encoding.ids)
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


# ------------------------------------------------------------------ planning

Definitions = Callable[[list[str], set[str]], dict[str, str]]


def plan_question(
    question_id: str,
    question: dict[str, Any],
    state: Any,
    policy: Policy,
    renderer: Renderer,
    definitions: Definitions,
) -> SpanItem:
    """One span question's item, as the released ``span_answer`` prepares it; ``QuestionError`` if invalid."""
    if "preset" in question:
        question = expand_preset(question, policy.span["schemas"])
    unknown = sorted(set(question) - FIELDS)
    if unknown:
        raise invalid(f"span questions do not take {unknown}")
    if question.get("overflow") not in (None, WINDOWED):
        raise invalid(
            f"span questions read their whole target in windows: overflow must be {WINDOWED!r}"
        )
    require_full_input(question)
    if question.get("head") not in (None, "router"):
        raise invalid("this model has no broad span head")
    parts = parts_of(state)
    roles = [role for role, _ in parts]
    target = question.get("over") or ("answer" if "answer" in roles else roles[-1])
    if not isinstance(target, str) or target not in roles:
        raise invalid(f"over names no part of the state: {target!r}")
    labels = labels_of(question)
    names = [name for name, _ in labels]
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise invalid("span instructions must be a string")
    override = threshold_of(question)
    span = policy.span
    label_set = policy.label_set(names)
    route = policy.route(label_set, names, instructions)
    asked = instructions
    if label_set in span["schemas"]:
        schema = span["schemas"][label_set]
        labels = [(n, d or schema["labels"].get(n, "")) for n, d in labels]
        asked = asked or schema["question"]
    if label_set == "evidence":
        asked = asked or span["evidence"]["question"]
    asked = asked or span["default_question"]
    if (
        label_set is None
    ):  # open labels: definitions written for the labels without a description
        empty = {n for n, d in labels if not d}
        written = definitions(names, empty) if empty else {}
        labels = [(n, written.get(n) or d) for n, d in labels]
    names_only = not any(d for _, d in labels)
    if (
        route == "ner"
        and names_only
        and (span.get("ner") or {}).get("names_only_route", "span") != "ner"
    ):
        route = "span"
    windows, count = renderer.windows(parts, target, asked, labels)
    threshold = (
        override
        if override is not None
        else policy.threshold(route, label_set, labels, count)
    )
    return SpanItem(
        question_id=question_id,
        task_type="span",
        ids=list(windows[0].ids),
        keys=names,
        descriptions=[d or None for _, d in labels],
        cost=sum(len(w.ids) for w in windows),
        route=route,
        names=tuple(names),
        label_set=label_set,
        threshold=threshold,
        evidence=span["evidence"]["decode"] if label_set == "evidence" else None,
        text=dict(parts)[target],
        windows=tuple(windows),
    )


class SpanReader:
    """Runs a span question's windows: its adapter on, the noncausal mask, its head over the gathered states."""

    def __init__(self, engine_model: Any, heads: dict[str, Any]):
        self.engine_model = engine_model
        self.heads = heads

    def logits(self, item: SpanItem) -> tuple[np.ndarray, np.ndarray]:
        import torch

        from ...plugins.base import AdapterGate, ForwardBatch

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
                    adapter=AdapterGate(ADAPTERS[item.route]),
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


# ------------------------------------------------------------------ answers


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
        overlap = (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
        spans.append(
            {
                "label": span["label"],
                "start": span["start"],
                "end": span["end"],
                "text": item.text[span["start"] : span["end"]],
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
