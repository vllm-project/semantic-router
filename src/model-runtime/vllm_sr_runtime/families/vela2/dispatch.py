"""Which span head reads a span question (decoder releases with a broad head).

First match wins: the question's ``head``; the calibration key ``pii``,
``halu`` or ``toxic``; a label set equal to one the router head was trained
on (17 PII types, ``unsupported``, the toxic categories); a non-empty subset
of the PII types; one label that is a hallucination alias (asked as the
trained ``unsupported`` label and answered with the caller's name); else the
broad head. Without a broad head every span question goes to the router head.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .calibration import BROAD_HEAD, ROUTER_HEAD, Calibration
from .request import Question

ROUTER_KEYS = ("pii", "halu", "toxic")
HALU_LABEL = "unsupported"
HALU_ALIASES = frozenset(
    {
        "unsupported",
        "unsupportedclaim",
        "hallucinated",
        "hallucination",
        "hallucinatedspan",
        "notsupported",
        "unsupportedspan",
        "fabricated",
        "fabrication",
    }
)
HALU_DESCRIPTION = "a claim not supported by the context"


@dataclass(frozen=True)
class SpanHead:
    """The head a span question goes to and the question as rendered (``alias`` maps labels back)."""

    head: str
    question: Question
    alias: dict[str, str] | None = None


def _normalized(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def halu_alias(question: Question) -> str | None:
    names = question.names
    return (
        names[0] if len(names) == 1 and _normalized(names[0]) in HALU_ALIASES else None
    )


class Dispatcher:
    """The dispatch rule of one package (``broad_head`` in its calibration)."""

    def __init__(self, calibration: Calibration, broad_loaded: bool):
        block = calibration.broad or {}
        self.enabled = bool(block) and broad_loaded
        sets = [frozenset(labels) for labels in block.get("router_label_sets", ())]
        halu = (calibration.schema("halu") or {}).get("labels", {})
        self.router_sets = sets or [calibration.pii_types, frozenset(halu)]
        self.pii_types = calibration.pii_types
        self.halu_description = halu.get(HALU_LABEL, HALU_DESCRIPTION)

    def route(self, question: Question) -> SpanHead:
        if not self.enabled:
            return SpanHead(ROUTER_HEAD, question)
        head = self._choose(question)
        alias = halu_alias(question)
        if head == ROUTER_HEAD and alias is not None and alias != HALU_LABEL:
            rendered = replace(question, options=((HALU_LABEL, self.halu_description),))
            return SpanHead(head, rendered, {HALU_LABEL: alias})
        return SpanHead(head, question)

    def _choose(self, question: Question) -> str:
        if question.head is not None:
            return question.head
        names = frozenset(question.names)
        if (
            question.key in ROUTER_KEYS
            or any(names == labels for labels in self.router_sets)
            or (names and names <= self.pii_types)
            or halu_alias(question) is not None
        ):
            return ROUTER_HEAD
        return BROAD_HEAD
