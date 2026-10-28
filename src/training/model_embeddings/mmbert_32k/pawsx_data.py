"""PAWS-X placeholder cleanup before sampling or scoring.

The dataset's README identifies ``NS`` as an untranslated sentence. Matching is
case-insensitive and ignores surrounding whitespace; valid text is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def pawsx_pair_problem(row: Mapping) -> str | None:
    """Return one exclusion reason, or None for an intact sentence pair."""
    normalized = []
    for key in ("sentence1", "sentence2"):
        value = row.get(key)
        if not isinstance(value, str):
            raise ValueError(f"PAWS-X {key} must be a string")
        normalized.append(value.strip().casefold())
    if "" in normalized:
        return "empty_sentence"
    if "ns" in normalized:
        return "untranslated_ns"
    return None


def clean_pawsx_pairs(rows: Iterable[Mapping]) -> tuple[list[Mapping], dict[str, int]]:
    """Filter either-sentence placeholders and report mutually exclusive counts."""
    kept = []
    counts = {
        "input_pairs": 0,
        "kept_pairs": 0,
        "empty_sentence": 0,
        "untranslated_ns": 0,
    }
    for row in rows:
        counts["input_pairs"] += 1
        problem = pawsx_pair_problem(row)
        if problem is None:
            kept.append(row)
            counts["kept_pairs"] += 1
        else:
            counts[problem] += 1
    return kept, counts
