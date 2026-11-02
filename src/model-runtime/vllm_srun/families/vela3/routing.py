"""Which Vela 3.0 network answers a question, and at which threshold (``vela3_encoder.system_one``).

Choice and Noul questions run on the decision backbone, except a Choice or Set
question whose instructions are one of the trained router texts (whitespace
normalised, ``router_questions.json``): it runs on the router adapter, or on
its question ID's own router adapter (domain, fact check, feedback). Span
questions route by their label set and wording: the trained label sets (PII,
unsupported claims, harm categories) take their question text, label
descriptions, temperature and thresholds (PII by target length) and, for PII
and unsupported claims, the set adapter; evidence questions (an evidence label
set, or evidence / support / cite wording) read one long window on the
evidence adapter; relation wordings run on the relation adapter and
names-only open labels on the NER adapter; every other span question on the
span adapter. Every rule and constant is the released runtime's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .package import Roster

ROUTER_TYPES = ("choice", "set")
# vela3_encoder._REL_Q and _EV_Q, character for character.
RELATION_WORDING = re.compile(
    r"\bspans? (?:are|is) the .+ (?:of|for)\b|\brelat(?:ed|ion|ions|es|ing)?\b|\b(?:head|tail)\b|\bstands? in\b"
    r"|\bwith respect to\b|\bcounterpart\b|\bsuch that\b|\brole of\b"
)
EVIDENCE_WORDING = re.compile(
    r"\b(?:evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
)
TRAINED_SETS = ("pii", "halu", "toxic")
EVIDENCE = "evidence"
WINDOW_TOKENS = 8192
WINDOW_OVERLAP = 512
SPAN_BUDGET = 32768
SPAN_ROWS = 32


def normalize(text: Any) -> str:
    """Whitespace-normalised text (``vela3_encoder._norm``)."""
    return " ".join(str(text or "").split())


@dataclass(frozen=True)
class RouterQuestion:
    """A trained router question: its schema ID, type, default part, abstain marker and options."""

    id: str
    type: str
    over: str
    abstain: bool
    options: dict[str, str]


def router_questions(raw: dict[str, Any]) -> dict[str, RouterQuestion]:
    """The router texts, by normalised instructions."""
    return {
        normalize(text): RouterQuestion(
            id=str(schema["id"]),
            type=str(schema["type"]),
            over=str(schema["over"]),
            abstain=bool(schema["abstain"]),
            options={
                str(name): str(description or "")
                for name, description in (schema.get("options") or {}).items()
            },
        )
        for text, schema in raw.items()
    }


@dataclass(frozen=True)
class SpanRoute:
    """How one span question is answered.

    ``adapter`` names the network; ``head`` the released runtime's head name
    (``span``, ``span_set``, ``relation``, ``ner``); ``label_set`` the trained
    label set it matched (or ``evidence``); ``evidence`` whether it reads one
    long window; ``text`` and ``labels`` the question as rendered, the trained
    set's question text and label descriptions filling what the request left
    empty. ``window`` is the engine's input length.
    """

    adapter: str
    head: str
    label_set: str | None
    evidence: bool
    relation_wording: bool
    text: str
    labels: tuple[tuple[str, str], ...]
    window: int


class Router:
    """One package's routing rules (``vela3_config.json`` ``span`` and ``router``)."""

    def __init__(
        self, vela3: dict[str, Any], roster: Roster, questions: dict[str, Any]
    ):
        self.span = vela3["span"]
        self.set_threshold = float(vela3["router"]["set_threshold"])
        self.roster = roster
        self.questions = router_questions(questions)
        self.entity_route = str(self.span.get("entity_route", "ner"))
        window = self.span.get("window") or {}
        self.window = int(window.get("max_len", WINDOW_TOKENS))
        self.overlap = int(window.get("overlap", WINDOW_OVERLAP))
        self.evidence_window = int(self.span[EVIDENCE].get("max_len", self.window))

    def router_question(self, instructions: Any) -> RouterQuestion | None:
        return self.questions.get(normalize(instructions))

    def router_adapters(self, schema_id: str) -> tuple[str, tuple[str, ...]]:
        """(answering head, adapters) of a router question: its own adapters, else the shared one."""
        own = self.roster.router_tasks.get(schema_id)
        if own is not None:
            return f"router:{own[0]}", own[1]
        return "router", (self.roster.router,)

    def label_set(self, names: list[str]) -> str | None:
        """The trained label set ``names`` fall within, ``evidence``, or None (``_label_set``)."""
        for name, labels in self.span["router_label_sets"].items():
            if names and set(names) <= set(labels):
                return str(name)
        evidence = {label.lower() for label in self.span[EVIDENCE]["labels"]}
        if names and {name.strip().lower() for name in names} <= evidence:
            return EVIDENCE
        return None

    def span_route(
        self, instructions: str | None, labels: list[tuple[str, str]]
    ) -> SpanRoute:
        """The network and rendering of a span question (``Vela3Encoder.span_answer``)."""
        names = [name for name, _ in labels]
        label_set = self.label_set(names)
        question = instructions
        schemas = self.span["schemas"]
        if label_set in schemas:
            schema = schemas[label_set]
            labels = [
                (name, description or schema["labels"].get(name, ""))
                for name, description in labels
            ]
            question = question or schema["question"]
        if label_set == EVIDENCE:
            question = question or self.span[EVIDENCE]["question"]
        question = question or self.span["default_question"]
        lowered = normalize(question).lower()
        evidence = label_set == EVIDENCE or (
            label_set not in TRAINED_SETS and bool(EVIDENCE_WORDING.search(lowered))
        )
        adapter = self.roster.evidence if evidence else self.roster.span
        head = "span"
        relation_wording = bool(RELATION_WORDING.search(lowered))
        names_only = not any(description for _, description in labels)
        if label_set is None and not evidence:
            if relation_wording:
                if "relation" in self.roster.tasks:
                    adapter, head = self.roster.tasks["relation"], "relation"
            elif (
                names_only and "ner" in self.roster.tasks and self.entity_route == "ner"
            ):
                adapter, head = self.roster.tasks["ner"], "ner"
        set_adapter = self.span.get("set_adapter")
        if (
            set_adapter
            and self.roster.span_set is not None
            and label_set in set_adapter["label_sets"]
        ):
            adapter, head = self.roster.span_set, "span_set"
        return SpanRoute(
            adapter=adapter,
            head=head,
            label_set=label_set,
            evidence=evidence,
            relation_wording=relation_wording,
            text=question,
            labels=tuple(labels),
            window=self.evidence_window if evidence else self.window,
        )

    def span_threshold(
        self, route: SpanRoute, target_tokens: int
    ) -> tuple[float, float]:
        """(threshold, temperature) of a span question over a target of ``target_tokens`` tokens."""
        span = self.span
        temperature = float(span["router_set_temperature"])
        thresholds = dict(span["thresholds_by_set"])
        buckets = span["pii_buckets"]
        if route.head == "span_set":
            overrides = span["set_adapter"]
            temperature = float(overrides["router_set_temperature"])
            thresholds.update(overrides["thresholds_by_set"])
            buckets = overrides["pii_buckets"]
        if route.label_set == "pii":
            bucket = next(b for b in buckets if target_tokens <= b["max_tokens"])
            return float(bucket["threshold"]), temperature
        if route.label_set in ("halu", "toxic"):
            return float(thresholds[route.label_set]), temperature
        if route.evidence:
            evidence = span[EVIDENCE]
            if evidence.get("buckets"):
                bucket = next(
                    b for b in evidence["buckets"] if target_tokens <= b["max_tokens"]
                )
                return float(bucket["threshold"]), 1.0
            return float(evidence["threshold"]), 1.0
        if route.head == "relation":
            relation = span["relation"]
            key = "threshold_pair" if len(route.labels) == 1 else "threshold_multi"
            return float(relation[key]), 1.0
        if route.relation_wording:
            return float(span["relation_threshold"]), 1.0
        if route.head == "ner":
            return float(span["ner"]["threshold"]), 1.0
        return float(span["threshold"]), 1.0
