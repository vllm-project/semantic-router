"""What both Vela 3.0 families (``vela3`` and the d3 sizes, ``vela3_d3``) read the same way.

The package description's name, model text, a question's threshold override,
the trained presets and the wording that marks a relation question come from the same released code for
every Vela 3.0 package, so the two families share them here rather than one
importing the other's internals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ..errors import INVALID_QUESTION, QuestionError
from ..registry.artifacts import package_file

# Both families' package description.
CONFIG_FILE = "vela3_config.json"
PRESETS = ("pii", "halu")
# The span heads both families have: the router head only, never a broad one.
SPAN_HEADS = ("router",)
# The fields a preset question keeps; its instructions and labels are the preset's.
PRESET_KEPT = ("over", "threshold", "overflow", "require_full_input")
# vela3_encoder._REL_Q, character for character.
RELATION_WORDING = re.compile(
    r"\bspans? (?:are|is) the .+ (?:of|for)\b|\brelat(?:ed|ion|ions|es|ing)?\b|\b(?:head|tail)\b|\bstands? in\b"
    r"|\bwith respect to\b|\bcounterpart\b|\bsuch that\b|\brole of\b"
)


def invalid(message: str, reason: str | None = None) -> QuestionError:
    """An ``invalid_question`` error with its ``reason``."""
    return QuestionError(INVALID_QUESTION, message, reason=reason)


def config_file(value: Any, where: str) -> str:
    """A file ``vela3_config.json`` names at ``where``, as a file of the package."""
    return package_file(value, f"{CONFIG_FILE}: {where}")


def text(value: Any) -> str:
    """Model text: a string as it is, anything else as compact JSON (``ub_d3_common.text``)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def threshold_of(question: dict[str, Any]) -> float | None:
    """A question's threshold override, ``None`` when it sets none."""
    threshold = question.get("threshold")
    if threshold is None:
        return None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise invalid("threshold must be a number in [0, 1]")
    return float(threshold)


def expand_preset(
    question: dict[str, Any], schemas: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """A preset question as the span question it names: the package's trained question and labels for it."""
    name = question["preset"]
    if name not in PRESETS or name not in schemas:
        raise invalid(f"preset must be one of {list(PRESETS)}", "preset_unknown")
    if set(question) - {"preset", "type", *PRESET_KEPT}:
        allowed = ", ".join(("preset", "type", *PRESET_KEPT))
        raise invalid(f"a preset question takes only {allowed}")
    if question.get("type") not in (None, "span"):
        raise invalid(f"preset {name} is a span question")
    schema = schemas[name]
    return {
        "type": "span",
        "instructions": schema["question"],
        "criteria": dict(schema["labels"]),
        **{key: question[key] for key in PRESET_KEPT if key in question},
    }
