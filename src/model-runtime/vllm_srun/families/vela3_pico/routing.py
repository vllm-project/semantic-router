"""A pico's router questions, span label sets, windows and span thresholds (``vela3_encoder`` of the package).

A Choice or Set question whose whitespace-normalised instructions are one of
the texts in ``router_questions.json`` is a router question. A span question's
label set is the trained set its labels fall within (PII, unsupported claims,
harm categories), ``evidence`` when they are evidence labels, else none; its
text is the request's, else the trained set's question, else the evidence
question, else ``span.default_question`` (``Vela3Encoder._span_row``). Labels
are rendered as given (render31): nothing is filled from the trained set.

Thresholds and temperatures follow ``Vela3Encoder._span_policy``, the same on
the one pass and the question-by-question path; a pico has no task adapters,
so the policy's relation and NER branches never apply: PII by the target's
token count (``pii_buckets``), unsupported claims and harm at their set
thresholds (temperature ``router_set_temperature``), evidence questions by
target length (``evidence.buckets``), relation wordings at
``relation_threshold`` and every other question at ``span.threshold``
(temperature 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import render31
from ..vela3_shared import EVIDENCE_WORDING, RELATION_WORDING, normalize

TRAINED_SETS = ("pii", "halu", "toxic")
EVIDENCE = "evidence"
WINDOW_TOKENS = 8192
WINDOW_OVERLAP = 512


@dataclass(frozen=True)
class RouterQuestion:
    """A built-in router question: its trained ID, type, default part, abstain marker and table entry as written."""

    id: str
    type: str
    over: str
    abstain: bool
    entry: dict[str, Any]


@dataclass(frozen=True)
class SpanRoute:
    """How one span question reads: its label set (or ``evidence``, or None), wordings, text and labels."""

    label_set: str | None
    evidence: bool
    relation_wording: bool
    text: str
    labels: tuple[tuple[str, str], ...]
    window: int


class Routes:
    """One package's routing rules (``vela3_config.json`` ``span`` and ``router``, ``router_questions.json``)."""

    def __init__(self, pico: dict[str, Any], questions: dict[str, Any]):
        self.span = pico["span"]
        self.set_threshold = float(pico["router"]["set_threshold"])
        self.questions = {
            normalize(text): RouterQuestion(
                id=str(entry["id"]),
                type=str(entry["type"]),
                over=str(entry["over"]),
                abstain=bool(entry["abstain"]),
                entry=dict(entry),
            )
            for text, entry in questions.items()
        }
        window = self.span.get("window") or {}
        self.window = int(window.get("max_len", WINDOW_TOKENS))
        self.overlap = int(window.get("overlap", WINDOW_OVERLAP))
        self.evidence_window = int(self.span[EVIDENCE].get("max_len", self.window))

    def router_question(self, instructions: Any) -> RouterQuestion | None:
        return self.questions.get(normalize(instructions))

    def label_set(self, names: list[str]) -> str | None:
        """The trained label set ``names`` fall within, ``evidence``, or None (``_label_set``)."""
        for name, labels in self.span["router_label_sets"].items():
            if names and set(names) <= set(labels):
                return str(name)
        evidence = {label.lower() for label in self.span[EVIDENCE]["labels"]}
        if names and {name.strip().lower() for name in names} <= evidence:
            return EVIDENCE
        return None

    def span_route(self, question: dict[str, Any]) -> SpanRoute:
        """A span question (``instructions``, ``criteria`` or ``labels``) rendered and routed as ``_span_row`` does."""
        labels = render31.span_labels(question)
        label_set = self.label_set(list(labels))
        text = render31.question_text(question)
        if not text and label_set in self.span["schemas"]:
            text = render31.text(self.span["schemas"][label_set]["question"])
        if not text and label_set == EVIDENCE:
            text = render31.text(self.span[EVIDENCE]["question"])
        text = text or self.span["default_question"]
        lowered = normalize(text).lower()
        evidence = label_set == EVIDENCE or (
            label_set not in TRAINED_SETS and bool(EVIDENCE_WORDING.search(lowered))
        )
        return SpanRoute(
            label_set=label_set,
            evidence=evidence,
            relation_wording=bool(RELATION_WORDING.search(lowered)),
            text=text,
            labels=tuple(labels.items()),
            window=self.evidence_window if evidence else self.window,
        )

    def span_threshold(
        self, route: SpanRoute, target_tokens: int
    ) -> tuple[float, float]:
        """(threshold, temperature) of a span question over a target of ``target_tokens`` tokens."""
        span = self.span
        if route.label_set == "pii":
            bucket = next(
                (b for b in span["pii_buckets"] if target_tokens <= b["max_tokens"]),
                None,
            )
            if bucket is None:
                raise ValueError(
                    f"no PII threshold bucket covers {target_tokens} tokens"
                )
            return float(bucket["threshold"]), float(span["router_set_temperature"])
        if route.label_set in ("halu", "toxic"):
            return (
                float(span["thresholds_by_set"][route.label_set]),
                float(span["router_set_temperature"]),
            )
        if route.evidence:
            evidence = span[EVIDENCE]
            if evidence.get("buckets"):
                bucket = next(
                    (
                        b
                        for b in evidence["buckets"]
                        if target_tokens <= b["max_tokens"]
                    ),
                    None,
                )
                if bucket is None:
                    raise ValueError(
                        f"no evidence threshold bucket covers {target_tokens} tokens"
                    )
                return float(bucket["threshold"]), 1.0
            return float(evidence["threshold"]), 1.0
        if route.relation_wording:
            return float(span["relation_threshold"]), 1.0
        return float(span["threshold"]), 1.0
