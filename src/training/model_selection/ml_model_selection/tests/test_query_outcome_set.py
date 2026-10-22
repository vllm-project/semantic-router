"""Query-level snapshot and split behaviour for selector training data."""

import subprocess
import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ("a", "b", "c")
MODEL_COUNT = len(DEFAULT_MODELS)
RATIO_SAMPLE = 2000
TRAIN_LOW, TRAIN_HIGH = 0.75, 0.85
VALIDATION_LOW, VALIDATION_HIGH = 0.05, 0.15
sys.path.insert(0, str(SERVICE_DIR))

from data_loader import RoutingRecord  # noqa: E402
from query_outcome_set import (  # noqa: E402
    SplitAssignment,
    build_query_outcome_sets,
    leaked_pairs,
    split_by_query,
)


def _records(queries, models=DEFAULT_MODELS):
    return [
        RoutingRecord(
            query=q,
            category="math",
            model_name=m,
            quality=0.5,
            latency_ms=100.0,
        )
        for q in queries
        for m in models
    ]


def test_records_collapse_into_one_snapshot_per_query():
    snapshots = build_query_outcome_sets(_records(["q1", "q2"]), source="bench")
    assert [s.query for s in snapshots] == ["q1", "q2"]
    assert all(len(s.outcomes) == MODEL_COUNT for s in snapshots)


def test_duplicate_query_model_pair_is_dropped():
    """A repeated (query, model) is a data defect, not a second candidate."""
    records = _records(["q1"], models=("a", "b")) + _records(["q1"], models=("a",))
    snapshot = build_query_outcome_sets(records, source="bench")[0]
    assert snapshot.model_refs == ("a", "b")


def test_split_keeps_every_query_whole():
    """The leak this replaces: a row split puts one query on both sides."""
    snapshots = build_query_outcome_sets(
        _records([f"q{i}" for i in range(200)]), source="bench"
    )
    assignment = split_by_query(snapshots)

    assert leaked_pairs(assignment) == []
    ids = [
        {s.query_id for s in assignment.train},
        {s.query_id for s in assignment.validation},
        {s.query_id for s in assignment.test},
    ]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
    assert sum(assignment.counts().values()) == len(snapshots)


def test_leak_detector_reports_a_query_in_two_splits():
    """Guards the guard: leaked_pairs must fail a hand-built leak, not just return empty."""
    snapshot = build_query_outcome_sets(_records(["q1"]), source="bench")[0]
    leaked = SplitAssignment(train=(snapshot,), validation=(), test=(snapshot,), seed=0)

    assert leaked_pairs(leaked) == [
        (snapshot.query_id, "a"),
        (snapshot.query_id, "b"),
        (snapshot.query_id, "c"),
    ]


def test_split_is_deterministic_and_order_independent():
    snapshots = build_query_outcome_sets(
        _records([f"q{i}" for i in range(150)]), source="bench"
    )
    first = split_by_query(snapshots, seed=7)
    shuffled = list(reversed(snapshots))
    second = split_by_query(shuffled, seed=7)

    assert {s.query_id for s in first.train} == {s.query_id for s in second.train}
    assert {s.query_id for s in first.test} == {s.query_id for s in second.test}


def test_split_is_stable_across_processes():
    """Assignment must not ride on PYTHONHASHSEED, or a rerun reshuffles the holdout."""
    script = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(SERVICE_DIR)!r})",
            "from data_loader import RoutingRecord",
            "from query_outcome_set import build_query_outcome_sets, split_by_query",
            "rs = [RoutingRecord(query=f'q{i}', category='math', model_name='a',",
            "                    quality=0.5, latency_ms=1.0) for i in range(50)]",
            "s = build_query_outcome_sets(rs, source='bench')",
            "print(','.join(sorted(x.query_id for x in split_by_query(s, seed=3).test)))",
        ]
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"split moved with PYTHONHASHSEED: {runs}"


def test_different_seeds_give_different_holdouts():
    snapshots = build_query_outcome_sets(
        _records([f"q{i}" for i in range(200)]), source="bench"
    )
    assert {s.query_id for s in split_by_query(snapshots, seed=1).test} != {
        s.query_id for s in split_by_query(snapshots, seed=2).test
    }


def test_ratios_are_respected_within_tolerance():
    snapshots = build_query_outcome_sets(
        _records([f"q{i}" for i in range(RATIO_SAMPLE)]), source="bench"
    )
    counts = split_by_query(snapshots, train=0.8, validation=0.1, test=0.1).counts()
    assert TRAIN_LOW < counts["train"] / RATIO_SAMPLE < TRAIN_HIGH
    assert VALIDATION_LOW < counts["validation"] / RATIO_SAMPLE < VALIDATION_HIGH


def test_candidate_set_identity_tracks_the_model_roster():
    """A changed roster must be a different snapshot, not silently the same one."""
    three = build_query_outcome_sets(
        _records(["q1"], models=("a", "b", "c")), source="bench"
    )[0]
    two = build_query_outcome_sets(_records(["q1"], models=("a", "b")), source="bench")[
        0
    ]
    reordered = build_query_outcome_sets(
        _records(["q1"], models=("c", "a", "b")), source="bench"
    )[0]

    assert three.candidate_set_id != two.candidate_set_id
    assert three.candidate_set_id == reordered.candidate_set_id


def test_query_identity_is_scoped_to_its_source():
    a = build_query_outcome_sets(_records(["q1"]), source="bench-a")[0]
    b = build_query_outcome_sets(_records(["q1"]), source="bench-b")[0]
    assert a.query_id != b.query_id


def test_invalid_ratios_are_rejected():
    snapshots = build_query_outcome_sets(_records(["q1"]), source="bench")
    with pytest.raises(ValueError):
        split_by_query(snapshots, train=0, validation=0, test=0)
    with pytest.raises(ValueError):
        split_by_query(snapshots, train=-1, validation=1, test=1)
