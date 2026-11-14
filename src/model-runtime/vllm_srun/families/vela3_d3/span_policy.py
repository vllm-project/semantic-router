"""Span routing and thresholds of the Vela 3.0 d3 sizes (``vela3_config.json`` ``span``).

``Policy.resolve`` decides everything about a span question the package
decides: its label set, its route (the adapter and head it runs: the
same name), the question it asks and each label's description.

Routing (the released ``_span_route``): router label sets (PII, halu, toxic),
evidence and answer labels and questions, and wordings that name no entity
run on the span adapter and head; relation questions on the relation adapter
and head; open entity labels on the NER adapter and head, unless they are
asked by name only and the package keeps those on the span adapter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..vela3_shared import RELATION_WORDING
from .package import Route, SpanConfig

EVIDENCE = "evidence"
ANSWERING = re.compile(
    r"\b(?:answers?|evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
)
ENTITY = re.compile(r"\b(?:entity|entities|types?|category|categories|named)\b")

# Writes descriptions for the open labels a question names without one.
Definitions = Callable[[list[str], set[str]], dict[str, str]]


@dataclass(frozen=True)
class SpanRoute:
    """What the package decides for one span question."""

    route: Route
    label_set: str | None
    question: str
    labels: list[tuple[str, str]]
    evidence: dict[str, Any] | None


@dataclass(frozen=True)
class Policy:
    """A package's span routing and thresholds, over the heads it ships."""

    config: SpanConfig
    definitions: Definitions

    def label_set(self, names: list[str]) -> str | None:
        """The router label set that holds every label, else ``evidence`` for evidence labels."""
        hit = next(
            (
                name
                for name, members in self.config.router_label_sets.items()
                if set(names) <= members
            ),
            None,
        )
        evidence = self.config.evidence
        if (
            hit is None
            and evidence is not None
            and {n.strip().lower() for n in names} <= evidence.labels
        ):
            return EVIDENCE
        return hit

    def route(
        self, label_set: str | None, names: list[str], instructions: str | None
    ) -> Route:
        """The released ``_span_route``: the route a question's labels and wording take."""
        if self.config.ner is None:
            return "span"
        if label_set is not None:
            return "span"
        if {n.strip().lower() for n in names} & {"answer", "answers"}:
            return "span"
        text = " ".join(str(instructions or "").split()).lower()
        if not text:
            return "ner"
        if RELATION_WORDING.search(text):
            return "span" if self.config.relation is None else "relation"
        if ANSWERING.search(text):
            return "span"
        if ENTITY.search(text):
            return "ner"
        return "span"

    def resolve(
        self, labels: list[tuple[str, str]], instructions: str | None
    ) -> SpanRoute:
        """A question's route, label set, question and described labels.

        A label set's labels take its schema's descriptions and question; open
        labels without a description take ``definitions``; a question still
        asked by name only leaves the NER route unless the package keeps it.
        """
        config = self.config
        names = [name for name, _ in labels]
        label_set = self.label_set(names)
        route = self.route(label_set, names, instructions)
        asked = instructions
        if label_set in config.schemas:
            schema = config.schemas[label_set]
            labels = [(n, d or schema["labels"].get(n, "")) for n, d in labels]
            asked = asked or schema["question"]
        if label_set == EVIDENCE and config.evidence is not None:
            asked = asked or config.evidence.question
        if label_set is None:
            empty = {n for n, d in labels if not d}
            written = self.definitions(names, empty) if empty else {}
            labels = [(n, written.get(n) or d) for n, d in labels]
        names_only = not any(d for _, d in labels)
        if (
            route == "ner"
            and names_only
            and not (config.ner is not None and config.ner.keeps_names_only)
        ):
            route = "span"
        return SpanRoute(
            route=route,
            label_set=label_set,
            question=asked or config.default_question,
            labels=labels,
            evidence=(
                config.evidence.decode
                if label_set == EVIDENCE and config.evidence is not None
                else None
            ),
        )

    def threshold(
        self,
        route: Route,
        label_set: str | None,
        labels: list[tuple[str, str]],
        target_tokens: int,
    ) -> float:
        """The package's threshold for a question on ``route`` over ``target_tokens`` target tokens."""
        config = self.config
        ner, relation = config.ner, config.relation
        described = any(d for _, d in labels)
        if route == "ner":
            # route() takes the NER and relation routes only when the package ships their heads.
            assert ner is not None
            return ner.threshold if described else ner.threshold_names_only
        if route == "relation":
            assert relation is not None
            return (
                relation.threshold_pair
                if len(labels) == 1
                else relation.threshold_multi
            )
        if label_set == "pii":
            # SpanConfig.parse checks that a bucket reaches SPAN_TARGET_TOKENS, the most a question reads.
            return next(
                threshold
                for limit, threshold in config.pii_buckets
                if target_tokens <= limit
            )
        if label_set == EVIDENCE and config.evidence is not None:
            return config.evidence.threshold
        if label_set in config.thresholds_by_set:
            return config.thresholds_by_set[label_set]
        if label_set is None and not described:
            return config.threshold_names_only
        return config.threshold
