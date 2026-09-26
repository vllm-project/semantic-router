"""Screen Score v6 two-source intersections before rendering any TRAIN text.

All records are abstract four-claim sets. No protected corpus, model, or
natural-language candidate is read or emitted by this module.
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

SEED = "decision20-score-v6-abstract-20260927"
PREREG_SHA256 = "e82e92826cf3ae106579b48612b9b435a581662a5b06be78119e73089138ac34"
GROUP_COUNT = 81
CLAIMS = frozenset(range(4))
PAIRS = tuple(frozenset(pair) for pair in itertools.combinations(range(4), 2))


@dataclass(frozen=True)
class Group:
    index: int
    language: str
    source_a: tuple[int, int, int]
    source_b: tuple[int, int, int]
    claim_order: tuple[int, int, int, int]
    source_order: tuple[str, str]
    repeated_pattern: str


def _digest(value: object) -> str:
    return hashlib.sha256(repr((SEED, value)).encode()).hexdigest()


def oracle(source_a: int, source_b: int) -> int:
    """Independently count the claim bits attested in both sources."""
    if source_a.bit_count() != 2 or source_b.bit_count() != 2:
        raise ValueError("Each source must attest exactly two distinct claims")
    return (source_a & source_b).bit_count()


def _mask(claims: frozenset[int]) -> int:
    return sum(1 << claim for claim in claims)


def _repeat_pair(values: tuple[frozenset[int], ...]) -> str | None:
    equal = [
        f"{left}{right}"
        for left in range(3)
        for right in range(left + 1, 3)
        if values[left] == values[right]
    ]
    return equal[0] if len(equal) == 1 else None


def eligible_templates() -> (
    dict[str, list[tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...]]]]
):
    by_level = {
        level: [(a, b) for a in PAIRS for b in PAIRS if len(a & b) == level]
        for level in range(3)
    }
    result: dict[
        str, list[tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...]]]
    ] = collections.defaultdict(list)
    for triplet in itertools.product(*(by_level[level] for level in range(3))):
        a_values = tuple(pair[0] for pair in triplet)
        b_values = tuple(pair[1] for pair in triplet)
        a_repeat, b_repeat = _repeat_pair(a_values), _repeat_pair(b_values)
        if a_repeat is None or b_repeat is None or a_repeat == b_repeat:
            continue
        result[f"A{a_repeat}_B{b_repeat}"].append((a_values, b_values))
    return dict(result)


def _group(
    index: int,
    templates: dict[
        str, list[tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...]]]
    ],
) -> Group:
    rng = random.Random(int(_digest(("group", index))[:16], 16))
    patterns = sorted(templates)
    pattern = patterns[index % len(patterns)]
    a_values, b_values = rng.choice(templates[pattern])
    permuted = list(range(4))
    rng.shuffle(permuted)
    permutation = dict(enumerate(permuted))

    def mapped(values: tuple[frozenset[int], ...]) -> tuple[int, int, int]:
        return tuple(
            _mask(frozenset(permutation[claim] for claim in group)) for group in values
        )  # type: ignore[return-value]

    order = list(range(4))
    rng.shuffle(order)
    source_order = ("A", "B") if index % 2 == 0 else ("B", "A")
    group = Group(
        index=index,
        language="en" if index < 60 else "zh",
        source_a=mapped(a_values),
        source_b=mapped(b_values),
        claim_order=tuple(order),  # type: ignore[arg-type]
        source_order=source_order,
        repeated_pattern=pattern,
    )
    if [oracle(a, b) for a, b in zip(group.source_a, group.source_b)] != [0, 1, 2]:
        raise AssertionError("Independent intersection oracle mismatch")
    return group


def _source_alternatives(mask: int, target_level: int) -> tuple[int, ...]:
    return tuple(
        alternative
        for alternative in (_mask(pair) for pair in PAIRS)
        if oracle(mask, alternative) != target_level
    )


def _ordered_claims(mask: int, order: tuple[int, ...]) -> tuple[int, int]:
    return tuple(claim for claim in order if mask & (1 << claim))  # type: ignore[return-value]


Feature = Callable[[Group, int], object]


def _heldout_predictions(groups: list[Group], feature: Feature) -> tuple[int, int]:
    correct = perfect_groups = 0
    for held in groups:
        lookup: dict[object, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for group in groups:
            if group.index == held.index:
                continue
            for level in range(3):
                lookup[feature(group, level)][level] += 1
        group_correct = 0
        for level in range(3):
            counts = lookup[feature(held, level)]
            prediction = (
                min(counts, key=lambda answer: (-counts[answer], answer))
                if counts
                else 0
            )
            group_correct += prediction == level
        correct += group_correct
        perfect_groups += group_correct == 3
    return correct, perfect_groups


def _features(groups: list[Group]) -> dict[str, dict[str, int]]:
    features: dict[str, Feature] = {
        "source_a_set": lambda g, level: g.source_a[level],
        "source_b_set": lambda g, level: g.source_b[level],
        "source_a_first": lambda g, level: _ordered_claims(
            g.source_a[level], g.claim_order
        )[0],
        "source_a_second": lambda g, level: _ordered_claims(
            g.source_a[level], g.claim_order
        )[1],
        "source_b_first": lambda g, level: _ordered_claims(
            g.source_b[level], g.claim_order
        )[0],
        "source_b_second": lambda g, level: _ordered_claims(
            g.source_b[level], g.claim_order
        )[1],
        "source_order": lambda g, level: g.source_order,
        "source_a_length": lambda g, level: g.source_a[level].bit_count(),
        "source_b_length": lambda g, level: g.source_b[level].bit_count(),
    }
    for source in ("a", "b"):
        for claim in range(4):
            features[f"source_{source}_claim{claim}_present"] = (
                lambda g, level, s=source, c=claim: bool(
                    (g.source_a if s == "a" else g.source_b)[level] & (1 << c)
                )
            )
    return {
        name: dict(
            zip(("correct", "perfect_groups"), _heldout_predictions(groups, function))
        )
        for name, function in features.items()
    }


def screen() -> dict[str, object]:
    prereg = (
        Path(__file__).parents[2]
        / "research"
        / "score-curriculum-v6-prereg-2026-09-27.md"
    )
    if hashlib.sha256(prereg.read_bytes()).hexdigest() != PREREG_SHA256:
        raise ValueError("Prospective v6 method SHA mismatch")
    templates = eligible_templates()
    patterns = sorted(templates)
    expected_patterns = {
        f"A{a}_B{b}" for a in ("01", "02", "12") for b in ("01", "02", "12") if a != b
    }
    if set(patterns) != expected_patterns:
        raise AssertionError("A source repetition pattern is infeasible")
    groups = [_group(index, templates) for index in range(GROUP_COUNT)]
    witness_failures = 0
    for group in groups:
        if len(set(group.source_a)) != 2 or len(set(group.source_b)) != 2:
            witness_failures += 1
        for a, b, level in zip(group.source_a, group.source_b, range(3)):
            if not _source_alternatives(a, level) or not _source_alternatives(b, level):
                witness_failures += 1
            if {
                oracle(a, alternative)
                for alternative in (_mask(pair) for pair in PAIRS)
            } != {0, 1, 2}:
                witness_failures += 1
            if {
                oracle(alternative, b)
                for alternative in (_mask(pair) for pair in PAIRS)
            } != {0, 1, 2}:
                witness_failures += 1
    feature_report = _features(groups)
    threshold = (GROUP_COUNT * 3 * 2) // 3
    failed = []
    if witness_failures:
        failed.append("source_necessity_or_oracle")
    for name, values in feature_report.items():
        if values["correct"] > threshold:
            failed.append(f"shallow_accuracy:{name}")
        if values["perfect_groups"]:
            failed.append(f"perfect_triplet:{name}")
    sequence = [
        (group.source_a, group.source_b, group.claim_order, group.source_order)
        for group in groups
    ]
    return {
        "schema_version": "decision20-score-v6-abstract-feasibility/1",
        "prereg_sha256": PREREG_SHA256,
        "status": "PASS_ABSTRACT_ONLY" if not failed else "HOLD_NO_CORPUS",
        "groups": len(groups),
        "languages": dict(collections.Counter(group.language for group in groups)),
        "eligible_pattern_count": len(patterns),
        "eligible_template_counts": {
            pattern: len(templates[pattern]) for pattern in patterns
        },
        "chosen_pattern_counts": dict(
            collections.Counter(group.repeated_pattern for group in groups)
        ),
        "source_witness_failures": witness_failures,
        "shallow_cap_correct": threshold,
        "features": feature_report,
        "failed_gates": failed,
        "abstract_sequence_sha256": _digest(sequence),
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
        raise PermissionError("Private abstract output directory is too broad")
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "groups": report["groups"],
                "failed_gates": report["failed_gates"],
                "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
