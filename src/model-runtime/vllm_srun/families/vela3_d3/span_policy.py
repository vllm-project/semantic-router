"""Span routing and thresholds of the Vela 3.0 d3 sizes (``vela3_config.json`` ``span``).

``SpanConfig`` is the package's ``span`` section, read once when the package is
verified. ``Policy.resolve`` decides everything about a span question the
package decides: its label set, its route (the adapter and head it runs: the
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
from typing import Any, Literal, get_args

from ...errors import PackageError
from ..vela3.routing import RELATION_WORDING

# A route names the adapter it runs and the head that reads it.
Route = Literal["span", "ner", "relation"]
ROUTES: tuple[Route, ...] = get_args(Route)
EVIDENCE = "evidence"
ANSWERING = re.compile(
    r"\b(?:answers?|evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
)
ENTITY = re.compile(r"\b(?:entity|entities|types?|category|categories|named)\b")

# Writes descriptions for the open labels a question names without one.
Definitions = Callable[[list[str], set[str]], dict[str, str]]


@dataclass(frozen=True)
class Evidence:
    """``span.evidence``: the evidence labels (lower case), question, threshold and decoding."""

    labels: frozenset[str]
    question: str
    threshold: float
    decode: dict[str, Any]


@dataclass(frozen=True)
class Ner:
    """``span.ner``: its thresholds with and without label descriptions, and where name-only labels run."""

    threshold: float
    threshold_names_only: float
    keeps_names_only: bool


@dataclass(frozen=True)
class SpanConfig:
    """A package's ``span`` section: label sets, schemas, questions and thresholds."""

    router_label_sets: dict[str, frozenset[str]]
    schemas: dict[str, dict[str, Any]]
    default_question: str
    threshold: float
    threshold_names_only: float | None
    thresholds_by_set: dict[str, float]
    pii_buckets: tuple[tuple[float, float], ...]
    evidence: Evidence | None
    ner: Ner | None
    relation_pair: float | None
    relation_multi: float | None

    @classmethod
    def parse(cls, span: Any) -> SpanConfig:
        """The ``span`` section, or a ``PackageError`` naming what is missing or malformed."""
        try:
            return cls._parse(span)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PackageError(
                f"vela3_config.json: span is malformed ({exc!r})"
            ) from exc

    @classmethod
    def _parse(cls, span: dict[str, Any]) -> SpanConfig:
        evidence = span.get("evidence")
        ner = span.get("ner")
        relation = span.get("relation") or {}
        schemas = {str(name): dict(schema) for name, schema in span["schemas"].items()}
        for schema in schemas.values():
            str(schema["question"])
            dict(schema["labels"])
        return cls(
            router_label_sets={
                str(name): frozenset(map(str, members))
                for name, members in span["router_label_sets"].items()
            },
            schemas=schemas,
            default_question=str(span["default_question"]),
            threshold=float(span["threshold"]),
            threshold_names_only=_number(span.get("threshold_names_only")),
            thresholds_by_set={
                str(name): float(value)
                for name, value in span["thresholds_by_set"].items()
            },
            pii_buckets=tuple(
                (float(bucket["max_tokens"]), float(bucket["threshold"]))
                for bucket in span.get("pii_buckets") or ()
            ),
            evidence=(
                Evidence(
                    labels=frozenset(str(x).lower() for x in evidence["labels"]),
                    question=str(evidence["question"]),
                    threshold=float(evidence["threshold"]),
                    decode=dict(evidence["decode"]),
                )
                if evidence
                else None
            ),
            ner=(
                Ner(
                    threshold=float(ner["threshold"]),
                    threshold_names_only=float(ner["threshold_names_only"]),
                    keeps_names_only=ner.get("names_only_route", "span") == "ner",
                )
                if isinstance(ner, dict)
                else None
            ),
            relation_pair=_number(relation.get("threshold_dev_pair")),
            relation_multi=_number(relation.get("threshold_dev_multi")),
        )


def _number(value: Any) -> float | None:
    """A configured number, or None for anything else (the packages put prose beside some)."""
    return float(value) if isinstance(value, int | float) else None


@dataclass(frozen=True)
class SpanRoute:
    """What the package decides for one span question."""

    route: Route
    label_set: str | None
    question: str
    labels: list[tuple[str, str]]
    evidence: dict[str, Any] | None


class Policy:
    """A package's span routing and thresholds, over the heads it ships."""

    def __init__(self, config: SpanConfig, heads: set[str]):
        self.config = config
        self.ner = "ner" in heads
        self.relation = "relation" in heads

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
            return "relation" if self.relation else "span"
        if ANSWERING.search(text):
            return "span"
        if ENTITY.search(text):
            return "ner"
        return "span"

    def resolve(
        self,
        labels: list[tuple[str, str]],
        instructions: str | None,
        definitions: Definitions,
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
            written = definitions(names, empty) if empty else {}
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
        described = any(d for _, d in labels)
        if route == "ner":
            if config.ner is None:
                raise PackageError("vela3_config.json: span.ner names no thresholds")
            ner = config.ner
            return ner.threshold if described else ner.threshold_names_only
        if label_set == "pii":
            value = next(
                threshold
                for limit, threshold in config.pii_buckets
                if target_tokens <= limit
            )
        elif label_set == EVIDENCE and config.evidence is not None:
            value = config.evidence.threshold
        elif label_set in config.thresholds_by_set:
            value = config.thresholds_by_set[label_set]
        else:
            value = config.threshold
        dev = config.relation_pair if len(labels) == 1 else config.relation_multi
        if route == "relation" and dev is not None:
            return dev
        if (
            route == "span"
            and label_set is None
            and not described
            and config.threshold_names_only is not None
        ):
            return config.threshold_names_only
        return value
