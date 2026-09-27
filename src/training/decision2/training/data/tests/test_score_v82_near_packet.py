"""Keep v8.2 near-pair adjudication blind to source IDs and labels."""

from __future__ import annotations

import json

import pytest

from training.data import score_v82_near_packet as packet
from training.model.data import file_sha256


def test_eight_opaque_near_pairs_and_frozen_audit(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.write_bytes(b"s" * 32)
    seed.chmod(0o600)
    roster = {
        "train_vs_candidate_select": ("v82_train", "v82_select"),
        "train_vs_prior_v8_train": ("v82_train", "v8_train"),
        "train_vs_prior_v81_train": ("v82_train", "v81_train"),
        "select_vs_prior_v8_select_goldfree": ("v82_select", "v8_select_blind"),
    }
    sources = {}
    for name in {source for pair in roster.values() for source in pair} | {
        "v81_select_blind"
    }:
        path = tmp_path / f"{name}.jsonl"
        source_row = {
            "state": f"Rendered evidence from {name}",
            "instructions": "Choose according to the evidence",
            "options": [{"key": "0", "description": "No"}],
            "label": 0,
            "audit_metadata": {"secret": "must not leak"},
        }
        if name.endswith("_blind"):
            path.write_text(
                json.dumps(
                    {
                        "review_group": "opaque",
                        "items": [
                            dict(source_row, review_id=f"{name}-a"),
                            dict(source_row, review_id=f"{name}-b"),
                        ],
                    }
                )
                + "\n"
            )
        else:
            path.write_text(
                "".join(
                    json.dumps(dict(source_row, id=f"{name}-{suffix}")) + "\n"
                    for suffix in ("a", "b")
                )
            )
        sources[name] = path
    overlap = {}
    for comparison, (left, right) in roster.items():
        overlap[comparison] = {
            "near_context": {
                "examples": [
                    {
                        "left_id": f"{left}-a",
                        "right_id": f"{right}-a",
                        "similarity": 0.95,
                    }
                ]
            },
            "near_full": {
                "examples": [
                    {
                        "left_id": f"{left}-b",
                        "right_id": f"{right}-b",
                        "similarity": 0.96,
                    }
                ]
            },
        }
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "decision2-score-v8.2-pilot-audit/1",
                "status": "HOLD_OVERLAP",
                "flagged_comparisons": sorted(roster),
                "overlap": overlap,
            }
        )
    )
    output, mapping = tmp_path / "pairs.jsonl", tmp_path / "map.json"
    result = packet.write(
        audit_path=audit,
        audit_sha256=file_sha256(audit),
        seed_path=seed,
        sources=sources,
        output=output,
        mapping_output=mapping,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["pairs"] == len(rows) == 8
    assert all(set(row) == {"review_pair", "item_a", "item_b"} for row in rows)
    assert all(
        set(item) == {"state", "instructions", "options"}
        for row in rows
        for item in (row["item_a"], row["item_b"])
    )
    with pytest.raises(ValueError, match="changed"):
        packet.write(
            audit_path=audit,
            audit_sha256="0" * 64,
            seed_path=seed,
            sources=sources,
            output=tmp_path / "fresh.jsonl",
            mapping_output=tmp_path / "fresh-map.json",
        )
