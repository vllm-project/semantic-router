"""Programmatic evidence witnesses and label-blind packet contract."""

from __future__ import annotations

import json

from training.data.score_v7p_build import build
from training.data.score_v7p_quality import _blind_packet, _source_witnesses, audit


def test_source_necessity_and_archived_control_for_every_case() -> None:
    rows = build(b"a" * 32, "train", 80)
    result = audit(rows, "train")
    assert result["source_necessity_witnesses"] == {
        "document_a_required": 80,
        "document_b_required": 80,
        "cross_document_veto_required": 80,
        "archived_control_invariant": 80,
    }
    assert all(_source_witnesses(row).values() for row in rows if row["label"] == 2)


def test_blind_packet_has_no_labels_or_level_ids_and_separate_key() -> None:
    rows = build(b"a" * 32, "select", 2)
    packet, key = _blind_packet(rows, "select")
    assert len(packet) == 2
    assert len(key) == 6
    rendered = json.dumps(packet)
    assert '"label"' not in rendered
    assert "_l0" not in rendered
    assert "_l1" not in rendered
    assert "_l2" not in rendered
    assert {value["label"] for value in key.values()} == {0, 1, 2}
