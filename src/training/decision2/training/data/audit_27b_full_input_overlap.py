"""Prospective, gold-free full-input overlap screen for a 27B data arm.

This pure CPU checker consumes projected input rows, not source or benchmark
answer keys. It is a new bounded screen; it does not alter frozen 27B receipts
or establish semantic/source isolation, data rights, or teacher-mask admission.
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import json
import unicodedata
from typing import Any

from training.data.build_pilot import _simhash
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    PARTITION_ROLE_COUNTS,
    project_partition_row,
    validate_core_rows,
)
from training.model.data import canonical

INPUT_SURFACE = ("state", "questions", "instructions", "options")
MIN_EXACT_CHARS = 20
MIN_NEAR_CHARS = 40


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _content(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def _structured(value: Any) -> Any:
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def _leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _leaves(child)]
    return []


def input_spans(row: dict[str, Any]) -> tuple[str, ...]:
    """Include each input field, its text leaves and the whole input surface."""
    present = [(field, row[field]) for field in INPUT_SURFACE if field in row]
    if not present or "state" not in row:
        raise ValueError("Input row lacks state")
    pieces = []
    spans = []
    for field, value in present:
        raw = _content(value)
        pieces.append(f"{field}: {raw}")
        spans.append(raw)
        spans.extend(_leaves(_structured(value)))
    spans.append("\n".join(pieces))
    return tuple(
        sorted({span for span in spans if len(_normalize(span)) >= MIN_EXACT_CHARS})
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def full_input_overlap_rows(
    schedule: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return aggregate row-pair counts, with no row IDs or text in results."""
    right_raw: dict[str, set[str]] = collections.defaultdict(set)
    right_norm: dict[str, set[str]] = collections.defaultdict(set)
    right_near: list[tuple[str, str, int]] = []
    bands: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for row in protected:
        for span in input_spans(row):
            right_raw[_sha(span)].add(row["id"])
            normalized = _normalize(span)
            right_norm[_sha(normalized)].add(row["id"])
            if len(normalized) < MIN_NEAR_CHARS:
                continue
            bits = _simhash(normalized)
            index = len(right_near)
            right_near.append((row["id"], normalized, bits))
            for band in range(8):
                bands[(band, (bits >> (8 * band)) & 255)].add(index)
    matches: dict[str, set[tuple[str, str]]] = {
        name: set()
        for name in ("same_row_ids", "exact_raw", "exact_normalized", "near")
    }
    right_ids = {row["id"] for row in protected}
    candidates_examined = 0
    for row in schedule:
        identifier = row["id"]
        if identifier in right_ids:
            matches["same_row_ids"].add((identifier, identifier))
        for span in input_spans(row):
            normalized = _normalize(span)
            matches["exact_raw"].update(
                (identifier, other) for other in right_raw.get(_sha(span), ())
            )
            matches["exact_normalized"].update(
                (identifier, other) for other in right_norm.get(_sha(normalized), ())
            )
            if len(normalized) < MIN_NEAR_CHARS:
                continue
            bits = _simhash(normalized)
            candidates = set()
            for band in range(8):
                candidates.update(bands.get((band, (bits >> (8 * band)) & 255), ()))
            candidates_examined += len(candidates)
            for index in candidates:
                other_id, other, other_bits = right_near[index]
                if abs(len(normalized) - len(other)) > 0.08 * max(
                    len(normalized), len(other)
                ):
                    continue
                if (bits ^ other_bits).bit_count() > 8:
                    continue
                if difflib.SequenceMatcher(None, normalized, other).ratio() >= 0.94:
                    matches["near"].add((identifier, other_id))
    return {
        "counts": {name: len(pairs) for name, pairs in matches.items()},
        "spans": {
            "schedule": sum(len(input_spans(row)) for row in schedule),
            "protected": sum(len(input_spans(row)) for row in protected),
        },
        "near_candidates_examined": candidates_examined,
    }


def audit_core_full_input(
    schedule: list[dict[str, Any]], roles: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Require all eight core roles and a canonical-equivalent TRAIN input view."""
    missing_partitions = sorted(set(PARTITION_ROLE_COUNTS) - set(roles))
    if missing_partitions:
        return {
            "status": "HOLD_MISSING_PARTITION_ROLES",
            "missing_roles": missing_partitions,
        }
    missing = sorted(CORE_ROLES - set(roles))
    if missing:
        return {"status": "HOLD_MISSING_CORE_ROLES", "missing_roles": missing}
    extras = sorted(set(roles) - CORE_ROLES)
    if extras:
        return {"status": "HOLD_UNATTESTED_OPTIONAL_ROLES", "roles": extras}
    try:
        counts = validate_core_rows(roles)
        train_by_id = {row["id"]: row for row in roles["rights_clean_train"]}
        selected = [project_partition_row(row, "train") for row in schedule]
    except (KeyError, TypeError, ValueError):
        return {"status": "HOLD_CORE_SCHEMA_OR_SCHEDULE"}
    if (
        not selected
        or len({row["id"] for row in selected}) != len(selected)
        or any(train_by_id.get(row["id"]) != row for row in selected)
    ):
        return {"status": "HOLD_SCHEDULE_INPUT_IDENTITY"}
    by_role = {
        role: full_input_overlap_rows(selected, prompts)
        for role, prompts in sorted(roles.items())
        if role != "rights_clean_train"
    }
    blocked = any(any(result["counts"].values()) for result in by_role.values())
    return {
        "status": (
            "HOLD_CANDIDATE_OVERLAP" if blocked else "PASS_BOUNDED_FULL_INPUT_SCREEN"
        ),
        "role_counts": counts,
        "by_role": by_role,
        "method": "Exact input-span SHA-256 plus eight-band 64-bit SimHash candidates; Hamming<=8, relative length<=8%, SequenceMatcher>=.94. No gold or outcome input.",
        "limitation": "All input text fields are represented, but short common spans are not compared separately; approximate search may miss paraphrases and source/semantic leakage remains unproven.",
    }
