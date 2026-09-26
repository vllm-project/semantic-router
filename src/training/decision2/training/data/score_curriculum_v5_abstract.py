"""Run the preregistered Score v5 weighted pairing feasibility screen.

This enumerates abstract integers only. It never renders text, loads a parent
dataset, opens a protected benchmark, or emits example-level records.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SEED = "decision20-score-v5-abstract-20260927"
PREREG_SHA256 = "5619c03d6ac19d8e8a9492d41f383409eee4ee0ede14e651363b000f6973d54a"
WEIGHT_POOLS = ((1, 2, 3, 4, 5), (1, 2, 2, 4, 5), (1, 2, 3, 3, 5))
MARK_POOLS = (
    (0, 1, 2, 3, 4),
    (0, 1, 1, 3, 4),
    (0, 1, 2, 2, 4),
    (0, 1, 2, 3, 5),
    (0, 0, 2, 3, 5),
)
GROUPS_PER_FAMILY = 81
EN_GROUPS = 60
ZH_GROUPS = 21


@dataclass(frozen=True)
class Plan:
    marks: tuple[int, ...]
    total: int


@dataclass(frozen=True)
class Group:
    index: int
    language: str
    pool: str
    weights: tuple[int, ...]
    plans: tuple[Plan, Plan, Plan]
    target_positions: tuple[int, int, int]
    eligible_triples: int
    min_span: int


def _digest(value: object) -> str:
    return hashlib.sha256(repr((SEED, value)).encode()).hexdigest()


def _plans(weights: tuple[int, ...], marks: tuple[int, ...]) -> list[Plan]:
    anchor = weights.index(max(weights))
    highest = max(marks)
    rest = list(marks)
    rest.remove(highest)
    indices = [position for position in range(5) if position != anchor]
    result = []
    for permutation in sorted(set(itertools.permutations(rest))):
        assigned = [0] * 5
        assigned[anchor] = highest
        for position, mark in zip(indices, permutation):
            assigned[position] = mark
        result.append(
            Plan(
                marks=tuple(assigned),
                total=sum(weight * mark for weight, mark in zip(weights, assigned)),
            )
        )
    return result


def _group(index: int) -> Group | None:
    language = "en" if index < EN_GROUPS else "zh"
    language_index = index if language == "en" else index - EN_GROUPS
    # All 15 pool pairs occur before any second pass through the schedule.
    weight_pool = index % len(WEIGHT_POOLS)
    mark_pool = (index // len(WEIGHT_POOLS)) % len(MARK_POOLS)
    rng = random.Random(_digest(("weights", index)))
    weights = list(WEIGHT_POOLS[weight_pool])
    rng.shuffle(weights)
    fixed_weights = tuple(weights)
    available = _plans(fixed_weights, MARK_POOLS[mark_pool])

    best_span: int | None = None
    chosen: tuple[Plan, Plan, Plan] | None = None
    chosen_hash: str | None = None
    eligible_count = 0
    for trio in itertools.combinations(available, 3):
        ordered = tuple(sorted(trio, key=lambda plan: plan.total))
        if len({plan.total for plan in ordered}) != 3:
            continue
        if any(
            len({plan.marks[position] for plan in ordered}) == 3
            for position in range(5)
        ):
            continue
        eligible_count += 1
        span = ordered[2].total - ordered[0].total
        tie = _digest(("pairing", index, tuple(plan.marks for plan in ordered)))
        if (
            best_span is None
            or span < best_span
            or (span == best_span and (chosen_hash is None or tie < chosen_hash))
        ):
            best_span, chosen, chosen_hash = span, ordered, tie
    if chosen is None or best_span is None:
        return None

    # Every target display position occurs once per complete group, and
    # every level-position combination is balanced in each language.
    shift = language_index % 3
    target_positions = tuple((level + shift) % 3 for level in range(3))
    assert all(chosen[level].total < chosen[level + 1].total for level in (0, 1))
    return Group(
        index=index,
        language=language,
        pool=f"w{weight_pool}_m{mark_pool}",
        weights=fixed_weights,
        plans=chosen,
        target_positions=target_positions,
        eligible_triples=eligible_count,
        min_span=best_span,
    )


Feature = Callable[[Group, int], object]


def _heldout_correct(groups: list[Group], feature: Feature) -> int:
    correct = 0
    for held in groups:
        lookup: dict[object, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for group in groups:
            if group.index == held.index:
                continue
            for level in range(3):
                lookup[feature(group, level)][level] += 1
        for level in range(3):
            counts = lookup[feature(held, level)]
            guess = (
                min(counts, key=lambda label: (-counts[label], label)) if counts else 0
            )
            correct += guess == level
    return correct


def _feature_results(groups: list[Group]) -> dict[str, object]:
    scopes = {
        "all": groups,
        "en": [group for group in groups if group.language == "en"],
        "zh": [group for group in groups if group.language == "zh"],
    }
    report = {}
    for scope, scoped in scopes.items():
        positions = {}
        for position in range(5):
            positions[str(position)] = {
                "weight_mark": _heldout_correct(
                    scoped,
                    lambda group, level, p=position: (
                        group.weights[p],
                        group.plans[level].marks[p],
                    ),
                ),
                "mark": _heldout_correct(
                    scoped,
                    lambda group, level, p=position: group.plans[level].marks[p],
                ),
                "product": _heldout_correct(
                    scoped,
                    lambda group, level, p=position: (
                        group.weights[p] * group.plans[level].marks[p]
                    ),
                ),
            }
        report[scope] = {
            "rows": len(scoped) * 3,
            "max_40pct_correct": (len(scoped) * 3 * 2) // 5,
            "by_position": positions,
            "target_position_correct": _heldout_correct(
                scoped, lambda group, level: group.target_positions[level]
            ),
            "unweighted_sum_correct": _heldout_correct(
                scoped, lambda group, level: sum(group.plans[level].marks)
            ),
            "max_product_correct": _heldout_correct(
                scoped,
                lambda group, level: max(
                    weight * mark
                    for weight, mark in zip(group.weights, group.plans[level].marks)
                ),
            ),
        }
    return report


def screen() -> dict[str, object]:
    prereg = (
        Path(__file__).parents[2]
        / "research"
        / "score-curriculum-v5-prereg-2026-09-27.md"
    )
    if hashlib.sha256(prereg.read_bytes()).hexdigest() != PREREG_SHA256:
        raise ValueError("Prospective v5 method SHA mismatch")
    groups = [_group(index) for index in range(GROUPS_PER_FAMILY)]
    missing = [index for index, group in enumerate(groups) if group is None]
    eligible = [group for group in groups if group is not None]
    pool_coverage = collections.Counter(group.pool for group in eligible)
    spans = collections.Counter(group.min_span for group in eligible)
    features = _feature_results(eligible) if not missing else None
    failures = []
    if missing:
        failures.append("insufficient_eligible_abstract_triplets")
    if features is not None:
        for scope, measurements in features.items():
            ceiling = measurements["max_40pct_correct"]
            for position, values in measurements["by_position"].items():
                for field, correct in values.items():
                    if correct > ceiling:
                        failures.append(f"{scope}:signal{position}:{field}")
            chance = measurements["rows"] // 3
            for field in (
                "target_position_correct",
                "unweighted_sum_correct",
                "max_product_correct",
            ):
                if measurements[field] != chance:
                    failures.append(f"{scope}:{field}")
    return {
        "schema_version": "decision20-score-v5-abstract-feasibility/1",
        "prereg_sha256": PREREG_SHA256,
        "seed": SEED,
        "status": "PASS_ABSTRACT_ONLY" if not failures else "HOLD_NO_CORPUS",
        "groups_requested": GROUPS_PER_FAMILY,
        "groups_eligible": len(eligible),
        "groups_without_eligible_triple": len(missing),
        "pool_coverage": dict(sorted(pool_coverage.items())),
        "minimum_score_span_counts": dict(sorted(spans.items())),
        "eligible_triples_min": min(
            (group.eligible_triples for group in eligible), default=0
        ),
        "eligible_triples_max": max(
            (group.eligible_triples for group in eligible), default=0
        ),
        "features": features,
        "failed_gates": failures,
        "corpus_emitted": False,
        "gpu_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = screen()
    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if args.output.parent.stat().st_mode & 0o077:
        raise PermissionError("Abstract feasibility output directory is not private")
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "groups_eligible": report["groups_eligible"],
                "failed_gates": report["failed_gates"],
                "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
