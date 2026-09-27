"""Contract checks for the aggregate-only Eos Score CPU admission audit."""

from __future__ import annotations

import json

import pytest

from training.data.eos08_score_only_admission import candidate, control, sha256


class TenTokens:
    def encode(self, text, *, add_special_tokens):
        assert text and not add_special_tokens
        return list(range(10))


def test_archived_replay_identity_and_type_tokens(tmp_path):
    rows = []
    for index in range(512):
        kind = "choice" if index < 160 else "noul" if index < 320 else "score"
        rows.append(
            {
                "id": f"r{index}",
                "source": "unit-test",
                "group_id": f"g{index}",
                "input_sha256": f"hash{index}",
                "task_type": kind,
                "state": "evidence",
                "instructions": "decide",
                "options": [
                    {"key": "0", "description": "no"},
                    {"key": "1", "description": "yes"},
                ],
            }
        )
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "replay_sha256": sha256(replay),
                "pool_roster": [
                    {"id": row["id"], "input_sha256": row["input_sha256"]}
                    for row in rows
                ],
                "replay_token_count": 512 * 40,
                "train_token_count": 123,
            }
        )
    )
    summary = control(replay, receipt, TenTokens())
    assert summary["counts_by_type"] == {"choice": 160, "noul": 160, "score": 192}
    assert summary["tokens_by_type"]["score"] == 192 * 40
    assert summary["total_tokens"] == 512 * 40 + 123

    receipt.write_text(receipt.read_text().replace("hash511", "changed"))
    with pytest.raises(ValueError, match="identity/order"):
        control(replay, receipt, TenTokens())


def test_ocnli_top_k_bound_is_deliberately_generous(tmp_path):
    source = tmp_path / "train.json"
    rows = [
        {
            "sentence1": "premise",
            "sentence2": "claim",
            "label": label,
            "genre": genre,
            "prem_id": str(index),
        }
        for index, (label, genre) in enumerate(
            [
                ("entailment", "fiction"),
                ("neutral", "fiction"),
                ("contradiction", "fiction"),
                ("entailment", "news"),
                ("-", "fiction"),
            ]
        )
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = candidate(source, sha256(source), TenTokens(), 2, 300, 1000, 0.01)
    assert summary["candidate_rows_before_group_and_overlap_filters"] == 3
    assert summary["excluded_counts"] == {"news_rights": 1, "no_consensus": 1}
    assert summary["candidate_native_tokens_top_k_upper_bound"] == 100
    assert summary["minimum_replacement_tokens"] == 290
    assert summary["status"] == "HOLD_TOKEN_UPPER_BOUND"
