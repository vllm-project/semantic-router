#!/usr/bin/env python3
"""Versioned query-outcome snapshots and query-level splitting.

Selector training reads one row per (query, model) pair, so a naive row split
puts the same query on both sides of the boundary and every model learns from
answers it is later scored against. This module makes the query the unit: one
QueryOutcomeSet per query, split before any feature, label, or utility is
derived from it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SNAPSHOT_VERSION = 1

# Digests are truncated for readable ids; collisions are not a security boundary.
_DIGEST_CHARS = 16


def _digest(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_DIGEST_CHARS]


@dataclass(frozen=True)
class CandidateOutcome:
    """What one candidate model did on one query."""

    model_ref: str
    success: bool
    quality: float
    latency_ms: float
    cost: float = 0.0
    # Extra approved signals, kept typed per snapshot rather than folded into quality.
    signals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryOutcomeSet:
    """One query, its source identity, and every candidate outcome recorded for it."""

    query: str
    source: str
    category: str
    outcomes: Tuple[CandidateOutcome, ...]
    version: int = SNAPSHOT_VERSION

    @property
    def query_id(self) -> str:
        """Identity of this query within its source, stable across processes."""
        return _digest(self.source, self.query)

    @property
    def candidate_set_id(self) -> str:
        """Identity of the candidate set, so a changed model roster is a different snapshot."""
        return _digest(*sorted(o.model_ref for o in self.outcomes))

    @property
    def model_refs(self) -> Tuple[str, ...]:
        return tuple(sorted(o.model_ref for o in self.outcomes))


@dataclass(frozen=True)
class SplitAssignment:
    """Query-level train/validation/test partition of one snapshot."""

    train: Tuple[QueryOutcomeSet, ...]
    validation: Tuple[QueryOutcomeSet, ...]
    test: Tuple[QueryOutcomeSet, ...]
    seed: int

    def counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def build_query_outcome_sets(
    records: Iterable, source: str, *, drop_single_candidate: bool = False
) -> List[QueryOutcomeSet]:
    """Group per-(query, model) records into one snapshot per query.

    `records` are RoutingRecord-shaped: query, category, model_name, quality,
    latency_ms. A repeated (query, model) keeps the first outcome, since a
    duplicate pair is a data defect rather than a second candidate.
    """
    grouped: Dict[str, List] = {}
    order: List[str] = []
    for record in records:
        if record.query not in grouped:
            grouped[record.query] = []
            order.append(record.query)
        grouped[record.query].append(record)

    snapshots: List[QueryOutcomeSet] = []
    for query in order:
        rows = grouped[query]
        seen: set = set()
        outcomes: List[CandidateOutcome] = []
        for row in rows:
            if row.model_name in seen:
                continue
            seen.add(row.model_name)
            outcomes.append(
                CandidateOutcome(
                    model_ref=row.model_name,
                    success=bool(getattr(row, "success", True)),
                    quality=float(row.quality),
                    latency_ms=float(row.latency_ms),
                    cost=float(getattr(row, "cost", 0.0)),
                )
            )
        if drop_single_candidate and len(outcomes) < 2:
            continue
        snapshots.append(
            QueryOutcomeSet(
                query=query,
                source=source,
                category=rows[0].category,
                outcomes=tuple(outcomes),
            )
        )
    return snapshots


def _bucket(query_id: str, seed: int) -> float:
    """Stable [0, 1) position for a query, independent of iteration order and PYTHONHASHSEED."""
    raw = _digest(query_id, str(seed))
    return int(raw, 16) / float(16**_DIGEST_CHARS)


def split_by_query(
    snapshots: Sequence[QueryOutcomeSet],
    *,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
) -> SplitAssignment:
    """Partition whole queries, so no (query, model) pair can appear in two splits.

    Assignment is a pure function of query identity and the seed, so the same
    snapshot splits identically on any machine and in any order.
    """
    total = train + validation + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    if min(train, validation, test) < 0:
        raise ValueError("split ratios must be non-negative")

    train_edge = train / total
    validation_edge = (train + validation) / total

    buckets: Dict[str, List[QueryOutcomeSet]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for snapshot in snapshots:
        position = _bucket(snapshot.query_id, seed)
        if position < train_edge:
            buckets["train"].append(snapshot)
        elif position < validation_edge:
            buckets["validation"].append(snapshot)
        else:
            buckets["test"].append(snapshot)

    return SplitAssignment(
        train=tuple(buckets["train"]),
        validation=tuple(buckets["validation"]),
        test=tuple(buckets["test"]),
        seed=seed,
    )


def leaked_pairs(assignment: SplitAssignment) -> List[Tuple[str, str]]:
    """Any (query_id, model_ref) present in more than one split; empty when the split is sound."""
    seen: Dict[Tuple[str, str], str] = {}
    leaks: List[Tuple[str, str]] = []
    for name, group in (
        ("train", assignment.train),
        ("validation", assignment.validation),
        ("test", assignment.test),
    ):
        for snapshot in group:
            for model_ref in snapshot.model_refs:
                key = (snapshot.query_id, model_ref)
                if key in seen and seen[key] != name:
                    leaks.append(key)
                else:
                    seen[key] = name
    return leaks
