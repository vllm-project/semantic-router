"""What the Vela 3.0 families (``vela3``, the d3 sizes ``vela3_d3``, the picos ``vela3_pico``) read the same way.

The package description's name, model text, a question's threshold override,
the trained presets, the wordings that mark a relation or evidence question,
the base UltraBERT tokenizer, the marker layout and the marker check come from
the same released code for every Vela 3.0 package, so the families share them
here rather than one importing another's internals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..errors import INVALID_QUESTION, PackageError, QuestionError
from ..registry.artifacts import package_file
from .vela2.encoder_layout import MARKERS, EncoderLayout
from .vela2.layout import Tokens

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
# vela3_encoder._EV_Q, character for character.
EVIDENCE_WORDING = re.compile(
    r"\b(?:evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
)
TOKEN_CACHE = 8192


def normalize(text: Any) -> str:
    """Whitespace-normalised text (``vela3_encoder._norm``)."""
    return " ".join(str(text or "").split())


def tokens(root: Path, markers: Mapping[str, Any]) -> Tokens:
    """The base UltraBERT tokenizer without the marker tokens, so text never creates one."""
    from tokenizers import Tokenizer

    data = json.loads((Path(root) / "tokenizer.json").read_text(encoding="utf-8"))
    data["added_tokens"] = [
        token
        for token in data.get("added_tokens", [])
        if token["content"] not in markers
    ]
    tokenizer = Tokenizer.from_str(json.dumps(data))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    cached = lru_cache(maxsize=TOKEN_CACHE)(
        lambda value: tuple(tokenizer.encode(value, add_special_tokens=False).ids)
    )
    return Tokens(
        encode=lambda value: tokenizer.encode(value, add_special_tokens=False),
        ids=lambda value: list(cached(value)) if value else [],
        tokenizer=tokenizer,
    )


def layout(config: Mapping[str, Any], window: int, overlap: int) -> EncoderLayout:
    """The Vela 2.0 marker layout of a package's ``config.json`` at one window."""
    return EncoderLayout(
        {
            "marker_ids": config["decision"]["marker_ids"],
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "pad_token_id": config["pad_token_id"],
            "max_length": window,
            "window_overlap": overlap,
        }
    )


def check_markers(config: Mapping[str, Any]) -> None:
    """Every marker a distinct row of the backbone's embedding table, so no input indexes past it."""
    vocab = config.get("vocab_size")
    if type(vocab) is not int or vocab <= 0:
        raise PackageError("config.json: vocab_size must be a positive integer")
    markers = config["decision"].get("marker_ids")
    if not isinstance(markers, dict) or set(markers) != set(MARKERS):
        raise PackageError(
            f"config.json: decision.marker_ids must name the markers {list(MARKERS)}"
        )
    wrong = sorted(
        name
        for name, value in markers.items()
        if type(value) is not int or not 0 <= value < vocab
    )
    if wrong:
        raise PackageError(
            f"config.json: decision.marker_ids {wrong} are not token IDs below vocab_size {vocab}"
        )
    if len(set(markers.values())) != len(markers):
        raise PackageError("config.json: decision.marker_ids must be distinct")


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
