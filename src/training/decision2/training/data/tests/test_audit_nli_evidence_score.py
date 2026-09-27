"""Contract tests for the private NLI source screen."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.data.audit_nli_evidence_score import (
    Pair,
    overlap_screen,
    read_ocnli,
    read_protected,
    read_snli,
    source_summary,
)


def test_publisher_label_mapping_and_premise_grouping(tmp_path: Path) -> None:
    snli_file = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "premise": ["Same premise", "Same premise", "Unknown"],
                "hypothesis": ["A", "B", "C"],
                "label": [0, 2, -1],
            }
        ),
        snli_file,
    )
    snli, provenance = read_snli(snli_file)
    assert [row.label for row in snli] == ["entailment", "contradiction"]
    assert provenance["excluded_no_consensus"] == 1
    assert source_summary(snli)["independent_premise_groups"] == 1

    ocnli_file = tmp_path / "train.json"
    rows = [
        {
            "sentence1": "前提甲",
            "sentence2": "结论甲",
            "label": "neutral",
            "genre": "gov",
            "prem_id": "gov_1",
            "id": 1,
        },
        {
            "sentence1": "前提甲",
            "sentence2": "结论乙",
            "label": "-",
            "genre": "gov",
            "prem_id": "gov_1",
            "id": 2,
        },
        {
            "sentence1": "前提乙",
            "sentence2": "结论丙",
            "label": "neutral",
            "genre": None,
            "prem_id": None,
            "id": 3,
        },
        {
            "sentence1": "前提甲",
            "sentence2": "结论丁",
            "label": "entailment",
            "genre": "gov",
            "prem_id": "gov_2",
            "id": 4,
        },
        {
            "sentence1": "前提另",
            "sentence2": "结论戊",
            "label": "contradiction",
            "genre": "gov",
            "prem_id": "gov_2",
            "id": 5,
        },
    ]
    ocnli_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    ocnli, provenance = read_ocnli(ocnli_file)
    assert [row.label for row in ocnli] == ["neutral", "entailment", "contradiction"]
    assert provenance["excluded_no_consensus"] == 1
    assert provenance["excluded_missing_genre_or_premise_id"] == 1
    assert len({row.group for row in ocnli}) == 1
    assert provenance["premise_ids_with_multiple_texts"] == 1
    assert provenance["source_ids_sharing_premise_text"] == 2


def test_exact_group_overlap_and_protected_answer_refusal(tmp_path: Path) -> None:
    pairs = [
        Pair(
            "The dog is outdoors by a tall red building.",
            "The dog is outside.",
            "entailment",
            "premise-group-1",
            "caption",
            0,
        ),
        Pair(
            "The dog is outdoors by a tall red building.",
            "The dog is inside.",
            "contradiction",
            "premise-group-1",
            "caption",
            1,
        ),
    ]
    overlaps = overlap_screen(
        pairs,
        [("typed_final_goldfree", "The dog is outdoors by a tall red building.")],
    )
    assert overlaps["role_group_counts"]["typed_final_goldfree"]["exact"] == 1
    assert overlaps["matched_source_group_count"] == 1

    protected_file = tmp_path / "protected.jsonl"
    protected_file.write_text('{"state":"safe input","label":2}\n')
    from training.data.audit_nli_evidence_score import sha_file

    manifest = tmp_path / "inventory.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "role": "typed_final_goldfree",
                    "path": str(protected_file),
                    "sha256": sha_file(protected_file),
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="non-prompt"):
        read_protected(manifest, [])
