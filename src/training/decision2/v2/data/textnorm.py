"""Text normalization shared by the v2 data audits.

`normalize`, `compact` and `text_leaves` (with the default `decode_json=False`)
are byte-for-byte the rules of `training.data.audit_vitaminc_score`.
"""

from __future__ import annotations

import functools
import json
import re
import string
import sys
import unicodedata
from typing import Any

from training.model.data import canonical

_CJK_RANGES = (
    (0x1100, 0x11FF),
    (0x2E80, 0x2FDF),
    (0x3040, 0x30FF),
    (0x3100, 0x312F),
    (0x3130, 0x318F),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0x20000, 0x323AF),
)
_CJK = re.compile(
    "[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _CJK_RANGES) + "]"
)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def compact(value: str) -> str:
    return "".join(char for char in normalize(value) if char.isalnum())


def decode_canonical_json(value: str) -> dict[str, Any] | list[Any] | None:
    """Return the container when `value` is exactly canonical JSON of one."""
    if not value or value[0] not in "{[":
        return None
    try:
        decoded = json.loads(value)
        if isinstance(decoded, (dict, list)) and canonical(decoded) == value:
            return decoded
    except (TypeError, ValueError):
        pass
    return None


def text_leaves(value: Any, *, decode_json: bool = False) -> list[str]:
    """Recursive string values; dict keys and non-string scalars are ignored."""
    if isinstance(value, str):
        if decode_json:
            decoded = decode_canonical_json(value)
            if decoded is not None:
                return text_leaves(decoded, decode_json=True)
        return [value]
    if isinstance(value, dict):
        return [
            leaf
            for child in value.values()
            for leaf in text_leaves(child, decode_json=decode_json)
        ]
    if isinstance(value, list):
        return [
            leaf
            for child in value
            for leaf in text_leaves(child, decode_json=decode_json)
        ]
    return []


@functools.cache
def _punctuation() -> str:
    unicode_punctuation = "".join(
        char
        for char in map(chr, range(sys.maxunicode + 1))
        if unicodedata.category(char).startswith("P")
    )
    return "".join(sorted(set(unicode_punctuation + string.punctuation)))


def word_tokens(value: str) -> list[str]:
    """Whitespace tokens of `normalize(value)` with edge punctuation removed.

    Punctuation is every Unicode P* character plus ASCII `string.punctuation`.
    """
    strip = _punctuation()
    tokens = (raw.strip(strip) for raw in normalize(value).split())
    return [token for token in tokens if token]


def is_cjk_heavy(value: str, threshold: float = 0.3) -> bool:
    """True when at least `threshold` of the compact characters are CJK
    (Han, kana, Hangul, Bopomofo)."""
    chars = compact(value)
    return bool(chars) and _CJK.subn("", chars)[1] >= threshold * len(chars)
