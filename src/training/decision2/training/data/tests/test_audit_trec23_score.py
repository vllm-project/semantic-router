"""Fail-closed source-screen contracts for public TREC 2023 Score diagnostics."""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest

from training.data import audit_trec23_score as audit


def test_qrels_sha_and_unjudged_are_not_zero(tmp_path, monkeypatch) -> None:
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 msmarco_passage_99_0 0\nq1 0 msmarco_passage_99_12 3\n")
    with pytest.raises(ValueError, match="publisher file"):
        audit.parse_qrels(path)
    monkeypatch.setattr(audit, "QRELS_SHA256", audit.sha_file(path))
    with pytest.raises(ValueError, match="qrels count changed"):
        audit.parse_qrels(path)
    # An unjudged passage must not be synthesized as a fourth row/grade 0.


def test_duplicate_classes_collapse_propagation_and_reject_conflict(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "dupes.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("msmarco_passage_99_0 msmarco_passage_99_0 snippet\n")
        stream.write("msmarco_passage_99_0 msmarco_passage_99_12 snippet\n")
        stream.write("msmarco_passage_99_25 msmarco_passage_99_25 snippet\n")
    monkeypatch.setattr(audit, "DUPES_SHA256", audit.sha_file(path))
    ids = {f"msmarco_passage_99_{offset}" for offset in (0, 12, 25)}
    mapping, count = audit.duplicate_representatives(path, ids)
    assert count == 3
    rows = [
        ("q1", "msmarco_passage_99_0", 2),
        ("q1", "msmarco_passage_99_12", 2),
        ("q1", "msmarco_passage_99_25", 0),
    ]
    assert audit.duplicate_summary(rows, mapping)["propagated_rows"] == 1
    conflicting = audit.duplicate_summary(
        rows[:1] + [("q1", "msmarco_passage_99_12", 3)], mapping
    )
    assert conflicting["conflicting_grade_groups"] == 1
    assert conflicting["conflicting_grade_rows"] == 2
    with pytest.raises(ValueError, match="cover every judgment"):
        audit.duplicate_representatives(path, ids | {"msmarco_passage_01_0"})


def test_exact_publisher_passage_byte_offset_and_native_rubric(tmp_path) -> None:
    member_dir = tmp_path / "members"
    member_dir.mkdir()
    a = {"pid": "msmarco_passage_99_0", "passage": "not answering", "docid": "d1"}
    first = (json.dumps(a) + "\n").encode()
    b = {"pid": f"msmarco_passage_99_{len(first)}", "passage": "answer", "docid": "d2"}
    with gzip.open(member_dir / "msmarco_passage_99.gz", "wb") as stream:
        stream.write(first)
        stream.write((json.dumps(b) + "\n").encode())
    passages, docids, hashes = audit.join_passages(member_dir, {b["pid"]})
    assert passages[b["pid"]] == "answer"
    assert docids[b["pid"]] == "d2"
    assert len(hashes["99"]) == 64
    row = audit.native_rows([("q1", b["pid"], 1)], {"q1": "question"}, passages)[0]
    assert row["task_type"] == "score"
    assert row["label"] == 1
    assert "does not answer" in row["options"][1]["description"]
    assert [option["key"] for option in row["options"]] == ["0", "1", "2", "3"]


def test_index_tar_requires_70_official_members() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for i in range(70):
            data = b"fake-gzip"
            member = tarfile.TarInfo(f"msmarco_v2_passage/msmarco_passage_{i:02d}.gz")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = buffer.getvalue()
    index = audit.index_tar(lambda start, end: payload[start : end + 1], len(payload))
    assert len(index) == 70
    assert index[0]["name"].endswith("00.gz")
    with pytest.raises(ValueError, match="70 passage members"):
        audit.index_tar(lambda start, end: payload[start : end + 1], 1024)


def test_missing_text_manifest_tokenizer_remain_hold(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        audit, "parse_qrels", lambda _: [("q1", "msmarco_passage_99_0", 0)]
    )
    monkeypatch.setattr(audit, "parse_queries", lambda _: {"q1": "question"})
    monkeypatch.setattr(
        audit,
        "duplicate_representatives",
        lambda *_: ({"msmarco_passage_99_0": "msmarco_passage_99_0"}, 1),
    )
    receipt = audit.audit(tmp_path, tmp_path, tmp_path, None, None, None, 1024)
    assert receipt["decision"] == "HOLD"
    assert "EXACT_PASSAGE_TEXT_NOT_JOINED" in receipt["blockers"]
    assert "NO_PINNED_PROTECTED_INVENTORY" in receipt["blockers"]
    assert "NO_PINNED_NATIVE_TOKENIZER" in receipt["blockers"]
