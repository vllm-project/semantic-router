"""Small independent checks for the unreleased 0.6B Score replacement."""

from __future__ import annotations

import json

import pytest

from training.data.build_score_replacement_d06 import (
    _blind_packet,
    _independent_oracle,
    _matched_score_groups,
)


@pytest.mark.parametrize(
    ("family", "state", "expected"),
    [
        (
            "score_obligation_review",
            {
                "events": [
                    {
                        "scope": "core",
                        "control": "a",
                        "timestamp": 1,
                        "assessment": "rejected",
                    },
                    {
                        "scope": "core",
                        "control": "a",
                        "timestamp": 3,
                        "assessment": "accepted",
                    },
                    {
                        "scope": "core",
                        "control": "b",
                        "timestamp": 2,
                        "assessment": "unresolved",
                    },
                    {
                        "scope": "informational",
                        "control": "c",
                        "timestamp": 4,
                        "assessment": "rejected",
                    },
                ]
            },
            1,
        ),
        (
            "score_evidence_intersection",
            {
                "eligible_claims": ["a", "b", "c", "d"],
                "documents": [
                    {"format": "checklist", "checked": ["a", "b"]},
                    {
                        "format": "table",
                        "rows": [
                            {"claim": "a", "status": "active"},
                            {"claim": "c", "status": "active"},
                            {"claim": "b", "status": "superseded"},
                        ],
                    },
                ],
            },
            1,
        ),
        (
            "score_route_depth",
            {
                "start": "A",
                "finish": "D",
                "links": [
                    {"from": "A", "to": "B"},
                    {"from": "B", "to": "C"},
                    {"from": "C", "to": "D"},
                    {"from": "X", "to": "D"},
                ],
            },
            1,
        ),
        (
            "score_timely_streak",
            {"days": [{"day": i, "on_time": i in {2, 3, 4, 8}} for i in range(14)]},
            1,
        ),
    ],
)
def test_displayed_facts_determine_score(family, state, expected):
    assert _independent_oracle(family, state) == expected


def test_matcher_replaces_complete_old_groups_with_fixed_row_count():
    rows = [
        {"id": "s1", "group_id": "s1"},
        {"id": "s2", "group_id": "s2"},
        {"id": "s3", "group_id": "s3"},
        {"id": "s4", "group_id": "s4"},
        {"id": "p1a", "group_id": "p1"},
        {"id": "p1b", "group_id": "p1"},
        {"id": "p2a", "group_id": "p2"},
        {"id": "p2b", "group_id": "p2"},
    ]
    lengths = dict(zip((r["id"] for r in rows), (2, 3, 5, 7, 11, 13, 17, 19)))
    chosen, audit = _matched_score_groups(rows, lengths, target=27, replace_rows=4)
    assert audit["removed_rows"] == 4
    assert audit["removed_pair_groups"] == 1
    assert audit["removed_single_groups"] == 2
    assert sum(len([r for r in rows if r["group_id"] == g]) for g in chosen) == 4
    assert audit["removed_tokens"] == sum(
        lengths[r["id"]] for r in rows if r["group_id"] in chosen
    )


def test_blind_packet_has_no_source_id_or_gold():
    rows = [
        {
            "id": f"private-v6-id-{i}",
            "group_id": "private-group",
            "family": "score_route_depth",
            "language": "en",
            "label": i,
            "state": {"links": [], "variant": i},
            "instructions": "Follow the links.",
            "options": [{"key": str(i), "description": str(i)} for i in range(3)],
        }
        for i in range(3)
    ]
    packet_bytes, key_bytes, pairs_bytes = _blind_packet(rows, b"x" * 32)
    packet = [json.loads(line) for line in packet_bytes.splitlines()]
    key = [json.loads(line) for line in key_bytes.splitlines()]
    pairs = [json.loads(line) for line in pairs_bytes.splitlines()]
    assert "private-v6-id-0" not in packet_bytes.decode()
    assert "private-group" not in packet_bytes.decode() + pairs_bytes.decode()
    assert all("label" not in item for item in packet)
    assert {item["label"] for item in key} == {0, 1, 2}
    assert set(pairs[0]["review_ids"]) == {item["review_id"] for item in packet}
