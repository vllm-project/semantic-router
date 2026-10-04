"""Plain-text rendering of System One rows for the teacher, and option-key parsing."""

from __future__ import annotations

import json
import re
from typing import Any


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_problem(row: dict[str, Any]) -> str:
    """The row as a readable problem: context, question and keyed options."""
    lines = [
        f"Context:\n{_text(row['state'])}",
        "",
        f"Question ({row['task_type']}):",
        _text(row["instructions"]),
    ]
    lines.append("Options:")
    for option in row["options"]:
        lines.append(f"[{option['key']}] {_text(option['description'])}")
    return "\n".join(lines)


_FINAL = re.compile(r"FINAL:\s*\[?([A-Za-z0-9_\-]+)\]?", re.I)


def parse_final_key(text: str, keys: list[str]) -> str | None:
    """The option key on the last ``FINAL:`` line, matched case-insensitively against the offered keys."""
    found = _FINAL.findall(text or "")
    if not found:
        return None
    lookup = {key.lower(): key for key in keys}
    return lookup.get(found[-1].strip().lower())


_YES = re.compile(r"\b(yes|true)\b", re.I)
_NO = re.compile(r"\b(no|false)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    """The first yes / no (or true / false) word of a short answer; None when absent or both appear first-equal."""
    text = (text or "").strip()
    yes, no = _YES.search(text), _NO.search(text)
    if yes and (not no or yes.start() < no.start()):
        return True
    if no and (not yes or no.start() < yes.start()):
        return False
    return None
