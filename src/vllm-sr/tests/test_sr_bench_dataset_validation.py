"""Every dataset consumer verifies the same frozen rows before using them."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from cli.sr_bench import datasets
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.sources import _write_dataset


def _prepared(
    root, *, profile="smoke", answer="A", metadata=None, benchmark="mmlu-pro"
):
    case = {
        "id": "fixture-case",
        "benchmark": benchmark,
        "messages": [{"role": "user", "content": "Choose A or B."}],
        "answer": answer,
        "metadata": metadata or {},
    }
    return _write_dataset(
        root, [case], profile, 7, {benchmark: {"revision": "fixture-v1"}}
    )


def _consume(reader, manifest, operation):
    if operation == "selection":
        return reader.selection("smoke")
    if operation == "detail":
        return reader.detail(manifest["id"])
    return reader.compose([manifest["id"]], ["mmlu-pro"])


@pytest.mark.parametrize("operation", ["selection", "detail", "compose"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        {"name": {}},
        {"profile": {}},
        {"profile": "unknown"},
        {"split": "unknown"},
        {"split": "holdout"},
        {"case_count": True},
        {"case_count": "1"},
        {"custom_subset": []},
        {"sources": {"mmlu-pro": "invalid"}},
    ],
)
def test_consumers_reject_malformed_headers(tmp_path, operation, change):
    manifest = _prepared(tmp_path)
    header = [] if change is None else {**manifest, **change}
    Path(manifest["path"]).with_name("manifest.json").write_text(json.dumps(header))
    with pytest.raises(ValueError):
        _consume(DatasetReader(tmp_path), manifest, operation)


@pytest.mark.parametrize("operation", ["selection", "detail", "compose"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        {"id": ""},
        {"benchmark": []},
        {"metadata": []},
        {"messages": {}},
        {"metadata": {"split": "holdout"}},
        {"metadata": {"split": "unknown"}},
        {"metadata": {"stratum": "x" * 201}},
    ],
)
def test_consumers_reject_invalid_frozen_rows(tmp_path, operation, change):
    manifest = _prepared(tmp_path)
    data_path = Path(manifest["path"])
    row = json.loads(data_path.read_text())
    content = json.dumps([] if change is None else {**row, **change}) + "\n"
    data_path.write_text(content)
    manifest["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    data_path.with_name("manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _consume(DatasetReader(tmp_path), manifest, operation)


def test_holdout_requires_case_evidence_and_dev_missing_split_remains_valid(tmp_path):
    dev = _prepared(tmp_path / "dev")
    reader = DatasetReader(tmp_path / "dev")
    assert reader.detail(dev["id"])["case_count"] == 1
    assert reader.compose([dev["id"]], ["mmlu-pro"])["id"] == dev["id"]
    assert next(
        b for b in reader.selection("smoke")["benchmarks"] if b["id"] == "mmlu-pro"
    )["eligible"]

    holdout = _prepared(tmp_path / "holdout", profile="standard")
    reader = DatasetReader(tmp_path / "holdout")
    for operation in (
        lambda: reader.detail(holdout["id"]),
        lambda: reader.selection("standard"),
        lambda: reader.compose([holdout["id"]], ["mmlu-pro"]),
    ):
        with pytest.raises(ValueError, match="case split"):
            operation()
    verified = _prepared(
        tmp_path / "valid", profile="standard", metadata={"split": "holdout"}
    )
    assert DatasetReader(tmp_path / "valid").detail(verified["id"])["case_count"] == 1


def test_selection_checks_answers_without_exposing_or_mutating_them(tmp_path):
    first = _prepared(tmp_path, answer="reference-first")
    second = _prepared(tmp_path, answer="reference-second")
    paths = [Path(row["path"]) for row in (first, second)]
    before = [p.read_bytes() for p in paths]
    reader = DatasetReader(tmp_path)
    result = reader.selection("smoke")
    selected = next(b for b in result["benchmarks"] if b["id"] == "mmlu-pro")
    assert not selected["eligible"] and not selected["source_ids"]
    assert "Multiple frozen versions" in selected["reason"]
    assert "reference-" not in json.dumps(result)
    with pytest.raises(ValueError, match="conflicting"):
        reader.compose([first["id"], second["id"]], ["mmlu-pro"])
    assert [p.read_bytes() for p in paths] == before


def test_selection_skips_unrelated_profile_case_files_and_bounds_selected_bytes(
    tmp_path, monkeypatch
):
    smoke = _prepared(tmp_path)
    standard = _prepared(
        tmp_path, profile="standard", metadata={"split": "holdout", "extra": "x" * 1000}
    )
    smoke_size = Path(smoke["path"]).stat().st_size
    assert Path(standard["path"]).stat().st_size > smoke_size
    monkeypatch.setattr(datasets, "MAX_ROW_BYTES", smoke_size)
    reader = DatasetReader(tmp_path)
    assert next(
        b for b in reader.selection("smoke")["benchmarks"] if b["id"] == "mmlu-pro"
    )["eligible"]
    blocked = next(
        b for b in reader.selection("standard")["benchmarks"] if b["id"] == "mmlu-pro"
    )
    assert not blocked["eligible"] and blocked["reason_code"] == "source_size_limit"


def test_oversized_source_blocks_its_family_even_with_valid_duplicate(
    tmp_path, monkeypatch
):
    _prepared(tmp_path, metadata={"extra": "x" * 4000})
    _prepared(tmp_path)
    _prepared(tmp_path, benchmark="gpqa-diamond")
    monkeypatch.setattr(datasets, "MAX_ROW_BYTES", 1000)
    result = {
        b["id"]: b for b in DatasetReader(tmp_path).selection("smoke")["benchmarks"]
    }
    assert result["gpqa-diamond"]["eligible"]
    assert not result["mmlu-pro"]["eligible"]
    assert not result["mmlu-pro"]["source_ids"]
    assert result["mmlu-pro"]["reason_code"] == "source_size_limit"


def test_scan_budget_never_selects_unverified_alternative(tmp_path, monkeypatch):
    manifests = [_prepared(tmp_path, answer=answer) for answer in ("A", "B")]
    size = max(Path(m["path"]).stat().st_size for m in manifests)
    monkeypatch.setattr(datasets, "MAX_FINGERPRINT_SCAN_BYTES", size)
    result = next(
        b
        for b in DatasetReader(tmp_path).selection("smoke")["benchmarks"]
        if b["id"] == "mmlu-pro"
    )
    assert not result["eligible"] and not result["source_ids"]
    assert result["reason_code"] == "scan_budget_exhausted"


@pytest.mark.parametrize(
    "declared", [None, [], ["unknown"], ["mmlu-pro", "mmlu-pro"], ["gpqa-diamond"]]
)
def test_oversized_source_without_trustworthy_scope_fails_globally(
    tmp_path, monkeypatch, declared
):
    manifest = _prepared(tmp_path, metadata={"extra": "x" * 4000})
    manifest["benchmarks"] = declared
    Path(manifest["path"]).with_name("manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(datasets, "MAX_ROW_BYTES", 1000)
    with pytest.raises(ValueError, match="valid benchmark scope"):
        DatasetReader(tmp_path).selection("smoke")


def test_composition_and_selection_do_not_require_harness_input_projection(
    tmp_path, monkeypatch
):
    manifest = _prepared(tmp_path)
    original = copy.deepcopy(manifest)

    def reject_projection(*_args):
        raise AssertionError(
            "Source-backed input display is not needed to validate frozen rows"
        )

    monkeypatch.setattr(datasets, "_public_case", reject_projection)
    reader = DatasetReader(tmp_path)
    assert reader.selection("smoke")["model_requests"] == 0
    assert reader.compose([manifest["id"]], ["mmlu-pro"])["id"] == manifest["id"]
    assert manifest == original
