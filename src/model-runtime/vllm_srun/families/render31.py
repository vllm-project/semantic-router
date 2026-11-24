"""render31: how a Vela 3.1 package turns a question into text (``render31.py``, ``render31/1``).

Vela 3.1 trains, reads and ships one renderer, and every Vela 3.1 package
carries a byte-identical copy. This module reimplements it, function for
function; the runtime never imports the package's copy. The rule: a question
is rendered exactly as it is given, and nothing is filled in.

- An option or label with a description renders ``name: description``; one
  without renders ``name``.
- A Noul is the two-option choice ``{"no": false text, "yes": true text}``; a
  side with no description renders bare.
- A built-in question asked by its text with no criteria renders as its table
  entry writes it (``builtin``: a ``router_questions.json`` entry, or a span
  set of ``vela3_config.json``). Criteria the request does give render as
  given: names only stay names only.
- A state or a description that is not a string renders as compact JSON
  (separators ``,`` and ``:``, non-ASCII kept).

The functions take a question as the package does, a mapping with ``type``,
``instructions`` and ``criteria`` (or ``labels`` for a span), and raise
``ValueError`` where the package raises. ``tests/test_render31.py`` checks
them against the package's own outputs on recorded inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

VERSION = "render31/1"
NOUL_OPTIONS = {"no": "", "yes": ""}


def text(value: Any) -> str:
    """A state, an instruction or a description: a string as is, None as ``""``, anything else as compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def option_text(name: Any, description: Any) -> str:
    """One option or label block: ``name: description``, or ``name`` when there is no description."""
    rendered = text(description)
    return str(name) if not rendered else f"{name}: {rendered}"


def as_given(criteria: Any) -> dict[str, str]:
    """Criteria as given, as ordered ``{name: description or ""}``.

    An object keeps its descriptions; a list or tuple gives names only, which
    must be distinct.
    """
    if isinstance(criteria, dict):
        return {str(name): text(value) for name, value in criteria.items()}
    if isinstance(criteria, (list, tuple)):
        names = [str(name) for name in criteria]
        if len(set(names)) != len(names):
            raise ValueError("option / label names must be distinct")
        return dict.fromkeys(names, "")
    raise ValueError(
        "criteria must be an object {name: description or null} or a list of names"
    )


def _empty(criteria: Any) -> bool:
    return criteria is None or (
        isinstance(criteria, (dict, list, tuple)) and len(criteria) == 0
    )


def options(
    question: Mapping[str, Any], builtin: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """A question's ordered ``{option name: description or ""}``, exactly as given.

    Noul: ``{"no": false text, "yes": true text}`` from ``true`` / ``false``
    criteria. Choice, Set and Score: the criteria as given; with none, a
    built-in question's options as written (``builtin`` holds ``options``),
    else an error.
    """
    kind = question.get("type")
    if kind == "noul":
        criteria = question.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise ValueError(
                "noul criteria must be an object {true: text, false: text}"
            )
        return {"no": text(criteria.get("false")), "yes": text(criteria.get("true"))}
    criteria = question.get("criteria")
    if _empty(criteria):
        if builtin is not None and builtin.get("options"):
            return as_given(builtin["options"])
        raise ValueError(f"a {kind} question needs criteria")
    return as_given(criteria)


def span_labels(
    question: Mapping[str, Any], builtin: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """A span question's ordered ``{label: description or ""}``, as given (``criteria``, else ``labels``).

    With none given, a built-in label set as written (``builtin`` holds
    ``labels``), else an error.
    """
    criteria = (
        question.get("criteria")
        if question.get("criteria") is not None
        else question.get("labels")
    )
    if _empty(criteria):
        if builtin is not None and builtin.get("labels"):
            return as_given(builtin["labels"])
        raise ValueError(
            "span questions need criteria: {label: description or null} or a list of labels"
        )
    return as_given(criteria)


def question_text(
    question: Mapping[str, Any], builtin: Mapping[str, Any] | None = None
) -> str:
    """The question text as given; a built-in question's text as written only when the request gives none."""
    rendered = text(question.get("instructions"))
    if not rendered and builtin is not None:
        rendered = text(builtin.get("question") or builtin.get("text"))
    return rendered


def decision(question: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """A Choice or Noul decision's (question text, options): the question of ``to_row`` (no ``[ABS]``)."""
    return text(question.get("instructions", "")), options(question)
