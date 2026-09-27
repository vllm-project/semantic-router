"""Prospective v2 distribution and snapshot gates."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from jev_arena.authored_release_scale_v2 import (
    _distribution,
    _domain_witness_issues,
    _verify_serialized_prompts,
    _write_rows,
)
from jev_arena.authored_release_scale_v2_audit import _required_gaps
from jev_arena.authored_release_scale_v2_packets import (
    _blind_rows,
    _validate_written_packet,
    _write_packet,
)
from jev_arena.authored_release_scale_v2_repair import _write_preserving_order
from jev_arena.authored_release_scale_v2_witness_audit import audit


def test_v2_balanced_native_targets_include_joint_hold() -> None:
    cases = (
        [
            {
                "slug": f"c{i}",
                "option_order": ["A", "B", "C", "HOLD"],
            }
            for i in range(4)
        ]
        + [{"slug": f"n{i}"} for i in range(4)]
        + [{"slug": f"s{i}"} for i in range(4)]
    )
    proofs = (
        [{"type": "choice", "original": answer} for answer in ("A", "B", "C", "HOLD")]
        + [
            {"type": "noul", "original": answer}
            for answer in (True, False, True, False)
        ]
        + [{"type": "score", "original": answer} for answer in (0, 1, 2, 0)]
    )
    result = _distribution(cases, proofs)
    expected_noul_false = 2
    assert result["choice_hold_answers"] == 1
    assert result["noul_false"] == expected_noul_false
    assert result["score_levels"] == {"0": 2, "1": 1, "2": 1}
    proofs[3]["original"] = "A"
    with pytest.raises(ValueError, match="Choice"):
        _distribution(cases, proofs)


def test_v2_preflight_requires_medium_and_long_originals() -> None:
    by_type = {
        "choice": Counter({"short": 3, "long": 1}),
        "noul": Counter({"short": 3, "medium": 1}),
        "score": Counter({"short": 3, "medium": 1}),
    }
    assert _required_gaps(by_type) == []
    by_type["choice"]["long"] = 0
    assert _required_gaps(by_type) == ["length_allocation"]


def test_v2_rejects_oracle_valid_but_physically_impossible_witness() -> None:
    case = {
        "operation": "net_range",
        "sources": [
            {"data": {"gross": 42}},
            {"data": {"tare": 8}},
        ],
        "variant": {"side": "left", "data": {"gross": 51}},
        "witnesses": {
            "left": [{"gross": 0}, {"gross": 55}],
            "right": [{"tare": 9}, {"tare": 10}],
        },
        "variant_witnesses": {
            "left": [{"gross": 8}, {"gross": 60}],
            "right": [{"tare": 12}, {"tare": 13}],
        },
    }
    assert _domain_witness_issues(case) == [
        "original/left/0: impossible gross/tare relation"
    ]
    case["witnesses"]["left"][0]["gross"] = 8
    assert _domain_witness_issues(case) == []


def test_v2_witness_audit_seals_aggregate_hold(tmp_path: Path) -> None:
    case = {
        "operation": "net_range",
        "sources": [{"data": {"gross": 42}}, {"data": {"tare": 8}}],
        "variant": {"side": "left", "data": {"gross": 51}},
        "witnesses": {
            "left": [{"gross": 0}, {"gross": 55}],
            "right": [{"tare": 9}, {"tare": 10}],
        },
        "variant_witnesses": {
            "left": [{"gross": 8}, {"gross": 60}],
            "right": [{"tare": 12}, {"tare": 13}],
        },
    }
    casebook = tmp_path / "casebook.private.json"
    casebook.write_text(json.dumps({"cases": [case]}))
    output = tmp_path / "audit.private.json"
    result = audit(casebook, output)
    assert result["status"] == "HOLD_BEFORE_BLIND_PACKET"
    assert result["domain_invalid_witnesses"] == 1
    assert result["affected_originals"] == 1
    with pytest.raises(FileExistsError):
        audit(casebook, output)


def test_repair_preserves_prompt_field_order(tmp_path: Path) -> None:
    path = tmp_path / "ordered.private.json"
    _write_preserving_order(path, {"source": {"z": 1, "a": 2}})
    assert path.read_text().index('"z"') < path.read_text().index('"a"')


def test_blind_packet_removes_source_ids_and_uses_independent_salts() -> None:
    native = [
        {
            "id": "private-case",
            "state": "Two records",
            "questions": {"decision": {"type": "noul"}},
        }
    ]
    first, first_join = _blind_rows(native, b"a" * 32)
    second, second_join = _blind_rows(native, b"b" * 32)
    assert first[0]["review_id"] != second[0]["review_id"]
    assert first_join[first[0]["review_id"]] == "private-case"
    assert second_join[second[0]["review_id"]] == "private-case"
    assert "private-case" not in json.dumps(first)
    assert "answer" not in json.dumps(first)


def test_blind_packet_preserves_native_choice_option_order(tmp_path: Path) -> None:
    original = [
        {
            "id": "private-case",
            "state": "Two records",
            "questions": {
                "decision": {
                    "type": "choice",
                    "criteria": {"z_second": "Second", "a_first": "First"},
                }
            },
        }
    ]
    packet, joins = _blind_rows(original, b"c" * 32)
    path = tmp_path / "packet.private.jsonl"
    _write_packet(path, packet)
    _validate_written_packet(path, original, joins)
    written = json.loads(path.read_text())
    assert list(written["questions"]["decision"]["criteria"]) == [
        "z_second",
        "a_first",
    ]
    path.write_text(json.dumps(written, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="option order"):
        _validate_written_packet(path, original, joins)


def test_prepared_snapshot_preserves_casebook_choice_order(tmp_path: Path) -> None:
    cases = [{"operation": "coverage_cost", "option_order": ["z_second", "a_first"]}]
    originals = [
        {
            "id": "case",
            "state": "Two records",
            "questions": {
                "decision": {
                    "type": "choice",
                    "criteria": {"z_second": "Second", "a_first": "First"},
                }
            },
        }
    ]
    path = tmp_path / "originals.private.jsonl"
    _write_rows(path, originals)
    _verify_serialized_prompts(cases, originals, path)
    path.write_text(json.dumps(originals[0], sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="native prompt changed"):
        _verify_serialized_prompts(cases, originals, path)
