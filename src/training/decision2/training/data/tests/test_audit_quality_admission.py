"""Security and quarantine checks for gold-free QuALITY admission."""

from __future__ import annotations

import json

import pytest

from training.data import audit_quality_admission as quality
from training.data.audit_quality_admission import (
    _containment,
    _fixed_shortcut_sample,
    _near_matches,
    _private_review_artifacts,
    _question_surface_overlap,
)
from training.model.data import file_sha256


def test_near_article_editions_quarantine_whole_group() -> None:
    words = " ".join(f"documentword{index}" for index in range(90))
    altered = words.replace("documentword42", "revisedword42")
    assert _near_matches({"source-group": words}, {"heldout": altered}) == {
        "source-group"
    }
    assert (
        _near_matches(
            {"source-group": words},
            {"source-group": words},
            skip_identical_keys=True,
        )
        == set()
    )


def test_protected_excerpt_in_state_or_question_quarantines_article() -> None:
    words = " ".join(f"documentword{index}" for index in range(100))
    excerpt = " ".join(f"documentword{index}" for index in range(20, 45))
    roles = {
        "typed_final_goldfree": [
            {"id": "protected", "state": "unrelated", "instructions": excerpt}
        ],
    }
    report, held = _containment({"source-group": words}, roles)
    assert held == {"source-group"}
    assert report["typed_final_goldfree"]["near_or_excerpt_full_input_only_rows"] == 1


def test_identical_question_text_quarantines_article_even_if_state_differs() -> None:
    groups = {
        "source-group": {
            "questions": [
                {
                    "question": "Which permit number applies to the final renewal?",
                    "options": ["A", "B", "C", "D"],
                }
            ]
        }
    }
    roles = {
        "css15_goldfree": [
            {
                "id": "protected",
                "state": "different passage",
                "instructions": json.dumps(
                    {
                        "q1": {
                            "instructions": "Which permit number applies to the final renewal?"
                        }
                    }
                ),
            }
        ]
    }
    report, held = _question_surface_overlap(groups, roles)
    assert held == {"source-group"}
    assert report["css15_goldfree"]["exact_question_text_rows"] == 1


def test_projected_core_role_rejects_answer_field(tmp_path, monkeypatch) -> None:
    roles = {}
    for role in quality.CORE_ROLE_COUNTS:
        path = tmp_path / f"{role}.jsonl"
        row = {"id": role, "state": "x", "instructions": "q"}
        if role == "typed_final_goldfree":
            row["answer"] = "secret"
        path.write_text(json.dumps(row) + "\n")
        roles[role] = {"path": path.name, "sha256": file_sha256(path), "rows": 1}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"roles": roles, "excluded_optional_role_count": 27})
    )
    monkeypatch.setattr(quality, "EXPECTED_INVENTORY_SHA256", file_sha256(manifest))
    monkeypatch.setattr(
        quality, "CORE_ROLE_COUNTS", dict.fromkeys(quality.CORE_ROLE_COUNTS, 1)
    )
    with pytest.raises(ValueError, match="non-input"):
        quality._protected_roles(manifest)


def test_fixed_sample_requires_distinct_articles() -> None:
    with pytest.raises(ValueError, match="independent article"):
        _fixed_shortcut_sample({})


def test_blind_packet_keeps_source_labels_in_separate_private_key(tmp_path) -> None:
    groups = {}
    for index in range(24):
        groups[f"article-{index}"] = {
            "metadata": {
                "article": f"Article content {index}",
                "source": "fixture",
                "license": "unknown",
                "title": f"Title {index}",
                "author": "Author",
                "year": "2020",
                "url": "",
            },
            "questions": [
                {
                    "question_unique_id": f"question-{index}",
                    "question": "What happened?",
                    "options": ["A", "B", "C", "D"],
                    "gold_label": 2,
                }
            ],
        }
    ledger, blind, key = (tmp_path / name for name in ("rights", "blind", "key"))
    report = _private_review_artifacts(groups, ledger, blind, key)
    assert report["rights_ledger_articles"] == 24
    assert report["blind_review_rows"] == 48
    assert "gold_label" not in blind.read_text()
    assert "source_gold_position" not in blind.read_text()
    assert "source_gold_position" in key.read_text()
    assert blind.stat().st_mode & 0o777 == 0o600
