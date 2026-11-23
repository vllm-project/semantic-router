"""Insight reference tests (G2): accepted forms, verification, CLI wiring."""

from __future__ import annotations

import hashlib
import json

import pytest
from recipe_lab.cli import main
from recipe_lab.contract import ContractError
from recipe_lab.insights import (
    merge_insight_refs,
    parse_insight_refs,
    resolve_insight_refs,
)
from recipe_lab.ledger import Ledger, LedgerRecord

from tests.test_cli import _replay_record, _write_frozen_spec


def _file_ref(
    tmp_path,
    name="artifact.jsonl",
    payload=b'{"case_id":"c1"}\n',
    kind="attr",
    anchor=None,
) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    ref = f"{kind}:{path}@sha256:{digest}"
    if anchor:
        ref += f"#{anchor}"
    return ref


# ---- parsing ----


def test_parse_insight_refs_splits_and_skips_empty():
    assert parse_insight_refs(None) == []
    assert parse_insight_refs("") == []
    assert parse_insight_refs(" bench:r1 , ,rec-abcdef123456 ") == [
        "bench:r1",
        "rec-abcdef123456",
    ]


# ---- bench ----


def test_bench_ref_format_only():
    assert resolve_insight_refs(["bench:run-2026-10-29_1"]) == [
        "bench:run-2026-10-29_1"
    ]
    with pytest.raises(ContractError, match="bench:<run_id>"):
        resolve_insight_refs(["bench:"])


# ---- attr ----


def test_attr_ref_verifies_file_hash(tmp_path):
    ref = _file_ref(tmp_path)
    assert resolve_insight_refs([ref]) == [ref]


def test_attr_ref_hash_mismatch(tmp_path):
    ref = _file_ref(tmp_path, payload=b"one")
    path = ref.split("@", 1)[0][len("attr:") :]
    wrong = f"attr:{path}@sha256:" + "0" * 64
    with pytest.raises(ContractError, match="hash mismatch"):
        resolve_insight_refs([wrong])


def test_attr_ref_missing_file(tmp_path):
    wrong = f"attr:{tmp_path / 'ghost.jsonl'}@sha256:" + "0" * 64
    with pytest.raises(ContractError, match="cannot read referenced file"):
        resolve_insight_refs([wrong])


# ---- diag ----


def test_diag_ref_with_anchor_verifies_hash(tmp_path):
    ref = _file_ref(tmp_path, kind="diag", anchor="L42")
    assert resolve_insight_refs([ref]) == [ref]


def test_diag_ref_requires_anchor(tmp_path):
    with pytest.raises(ContractError, match="#<anchor>"):
        resolve_insight_refs([_file_ref(tmp_path, kind="diag")])


def test_unknown_form_rejected():
    with pytest.raises(ContractError, match="unsupported insight ref"):
        resolve_insight_refs(["wiki:page"])


# ---- rec- ----


def _seed_ledger(tmp_path) -> tuple[Ledger, str]:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    stored = ledger.append(
        LedgerRecord.from_dict(_replay_record("a.yaml", "routed_accuracy", 0.9))
    )
    return ledger, stored["record_id"]


def test_rec_ref_resolves_against_ledger(tmp_path):
    ledger, record_id = _seed_ledger(tmp_path)
    assert resolve_insight_refs([record_id], ledger) == [record_id]


def test_rec_ref_unknown_in_ledger(tmp_path):
    ledger, _ = _seed_ledger(tmp_path)
    with pytest.raises(ContractError, match="not found in ledger"):
        resolve_insight_refs(["rec-" + "0" * 12], ledger)


def test_rec_ref_malformed(tmp_path):
    ledger, _ = _seed_ledger(tmp_path)
    with pytest.raises(ContractError, match="rec-<12 hex>"):
        resolve_insight_refs(["rec-XYZ"], ledger)


def test_rec_ref_without_ledger(tmp_path):
    with pytest.raises(ContractError, match="needs a ledger"):
        resolve_insight_refs(["rec-" + "0" * 12], None)


# ---- dedupe + merge ----


def test_resolve_dedupes_preserving_order(tmp_path):
    ledger, record_id = _seed_ledger(tmp_path)
    refs = [record_id, "bench:r1", record_id, "bench:r1", "bench:r2"]
    assert resolve_insight_refs(refs, ledger) == [record_id, "bench:r1", "bench:r2"]


def test_merge_into_provenance_dedupes():
    provenance = {"insight_refs": ["bench:r1"]}
    merge_insight_refs(provenance, ["bench:r1", "bench:r2"])
    assert provenance["insight_refs"] == ["bench:r1", "bench:r2"]


def test_merge_creates_list_when_missing():
    provenance = {}
    merge_insight_refs(provenance, ["bench:r1"])
    assert provenance["insight_refs"] == ["bench:r1"]


def test_merge_rejects_non_list():
    with pytest.raises(ContractError, match="must be a list"):
        merge_insight_refs({"insight_refs": "bench:r1"}, ["bench:r2"])


# ---- CLI wiring ----


def _write_candidate(tmp_path) -> None:
    cands = tmp_path / "cands"
    cands.mkdir(exist_ok=True)
    (cands / "c.yaml").write_text("routing: {}\n", encoding="utf-8")


def _run_cmd(tmp_path, spec_path, cases, ledger: str, extra: list[str]) -> list[str]:
    return [
        "run",
        "--spec",
        spec_path,
        "--cases",
        str(cases),
        "--bank",
        str(tmp_path / "bank.json"),
        "--ledger",
        ledger,
        "--candidates",
        str(tmp_path / "cands"),
        "--router-bin",
        "/bin/true",
        "--mock-backend",
        str(tmp_path / "mock.py"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def test_run_merges_insight_refs_into_ledger(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    (tmp_path / "bank.json").write_text("", encoding="utf-8")
    _write_candidate(tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"
    stored = Ledger(ledger_path).append(
        LedgerRecord.from_dict(_replay_record("seed.yaml", "routed_accuracy", 0.9))
    )
    attr = _file_ref(
        tmp_path, name="attr.jsonl", payload=b'{"case_id":"c1","truth":"math"}\n'
    )

    def fake_evaluate(**kwargs):
        return {"summary": {"routed_accuracy": 0.5}, "bank_stats": {}, "statuses": {}}

    monkeypatch.setattr(cli, "evaluate_candidate", fake_evaluate)
    assert (
        main(
            _run_cmd(
                tmp_path,
                spec_path,
                cases,
                str(ledger_path),
                ["--insight-refs", f"{stored['record_id']}, bench:r7, {attr}"],
            )
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2  # seed + the candidate record
    assert rows[-1]["candidate"]["provenance"]["insight_refs"] == [
        stored["record_id"],
        "bench:r7",
        attr,
    ]


def test_run_rejects_unverifiable_insight_ref(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    (tmp_path / "bank.json").write_text("", encoding="utf-8")
    _write_candidate(tmp_path)
    assert (
        main(
            _run_cmd(
                tmp_path,
                spec_path,
                cases,
                str(tmp_path / "ledger.jsonl"),
                ["--insight-refs", "rec-000000000000"],
            )
        )
        == 2
    )
    assert "not found in ledger" in capsys.readouterr().err


def test_ledger_append_merges_insight_refs(tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    stored = Ledger(ledger_path).append(
        LedgerRecord.from_dict(_replay_record("a.yaml", "routed_accuracy", 0.9))
    )
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(_replay_record("b.yaml", "routed_accuracy", 0.8)), encoding="utf-8"
    )
    assert (
        main(
            [
                "ledger",
                "append",
                "--file",
                str(ledger_path),
                "--record",
                str(record_path),
                "--insight-refs",
                f"{stored['record_id']},bench:r9",
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["candidate"]["provenance"]["insight_refs"] == [
        stored["record_id"],
        "bench:r9",
    ]


def test_ledger_append_rejects_unknown_rec_ref(tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    Ledger(ledger_path).append(
        LedgerRecord.from_dict(_replay_record("a.yaml", "routed_accuracy", 0.9))
    )
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(_replay_record("b.yaml", "routed_accuracy", 0.8)), encoding="utf-8"
    )
    assert (
        main(
            [
                "ledger",
                "append",
                "--file",
                str(ledger_path),
                "--record",
                str(record_path),
                "--insight-refs",
                "rec-111111111111",
            ]
        )
        == 2
    )
    assert "not found in ledger" in capsys.readouterr().err
