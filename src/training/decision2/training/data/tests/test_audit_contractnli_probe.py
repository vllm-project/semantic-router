"""Contract tests for the aggregate-only publisher TRAIN audit."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from training.data.audit_contractnli_probe import audit_archive, audit_train


def _publisher_payload() -> dict:
    labels = {
        "hyp-a": {"hypothesis": "The secret clause applies."},
        "hyp-b": {"hypothesis": "The special clause applies."},
    }
    documents = []
    for index in range(15):
        text = f"Document {index} contains an exception and evidence."
        documents.append(
            {
                "id": f"private-doc-{index}",
                "text": text,
                "document_type": "sec-text",
                "spans": [[0, len(text)]],
                "annotation_sets": [
                    {
                        "annotations": {
                            "hyp-a": {
                                "choice": (
                                    "NotMentioned" if index % 3 == 0 else "Entailment"
                                ),
                                "spans": [] if index % 3 == 0 else [0],
                            },
                            "hyp-b": {"choice": "Contradiction", "spans": [0]},
                        }
                    }
                ],
            }
        )
    return {"documents": documents, "labels": labels}


def _archive(path: Path, payload: dict) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("contract-nli/train.json", json.dumps(payload))
        archive.writestr(
            "contract-nli/TERMS",
            "The data follow Creative Commons Attribution 4.0 International.",
        )
        archive.writestr("contract-nli/dev.json", "not valid json on purpose")
        archive.writestr("contract-nli/test.json", "not valid json on purpose")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_screen_never_reads_dev_test_or_emits_raw_text(tmp_path: Path) -> None:
    archive = tmp_path / "publisher.zip"
    digest = _archive(archive, _publisher_payload())
    result = audit_archive(archive, expected_sha256=digest, seed="test")
    assert result["documents"] == 15
    assert result["labeled_pairs"] == 30
    assert result["hypotheses"] == 2
    assert result["developer_test_labels_opened"] is False
    assert result["evidence_integrity_errors"] == {}
    split = result["document_disjoint_split"]
    assert split["train_documents"] + split["holdout_documents"] == 15
    assert split["holdout_pairs"] == split["holdout_documents"] * 2
    assert (
        0 <= split["shortcuts"]["hypothesis_only"]["correct"] <= split["holdout_pairs"]
    )
    serialized = json.dumps(result)
    assert "secret clause" not in serialized
    assert "private-doc" not in serialized


def test_archive_identity_and_terms_are_mandatory(tmp_path: Path) -> None:
    archive = tmp_path / "publisher.zip"
    digest = _archive(archive, _publisher_payload())
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_archive(archive, expected_sha256="0" * 64, seed="test")
    assert audit_archive(archive, expected_sha256=digest, seed="test")
    with zipfile.ZipFile(archive, "w") as publisher:
        publisher.writestr("contract-nli/train.json", json.dumps(_publisher_payload()))
        publisher.writestr("contract-nli/TERMS", "Unrecognized license wording")
    changed_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="terms differ"):
        audit_archive(archive, expected_sha256=changed_digest, seed="test")


def test_bad_annotations_fail_or_are_counted() -> None:
    payload = _publisher_payload()
    payload["documents"][0]["annotation_sets"][0]["annotations"]["hyp-a"]["spans"] = [
        99
    ]
    result = audit_train(payload, seed="test")
    assert result["evidence_integrity_errors"]["invalid_span_index"] == 1
    payload["documents"][1]["annotation_sets"][0]["annotations"].pop("hyp-b")
    with pytest.raises(ValueError, match="Incomplete"):
        audit_train(payload, seed="test")
