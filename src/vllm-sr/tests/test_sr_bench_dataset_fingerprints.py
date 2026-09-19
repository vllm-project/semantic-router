"""Default-source fingerprints stream full content and retain only bounded proofs."""

import json
from pathlib import Path

import pytest
from cli.sr_bench import datasets
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.sources import _write_dataset


def _row(identity, benchmark, answer="A"):
    return {
        "id": identity,
        "benchmark": benchmark,
        "messages": [{"role": "user", "content": "Choose A or B."}],
        "answer": answer,
        "metadata": {"split": "dev"},
    }


def _save(root, rows):
    return _write_dataset(
        root,
        rows,
        "quick",
        7,
        {r["benchmark"]: {"revision": "fixture-v1"} for r in rows},
    )


def _choices(reader):
    return {b["id"]: b for b in reader.selection("quick")["benchmarks"]}


def test_verified_large_bundle_does_not_block_small_equivalent_standalone(
    tmp_path, monkeypatch
):
    small = [_row("small", "mmlu-pro")]
    large = [_row(f"large-{i}", "livecodebench") for i in range(12)]
    standalone = _save(tmp_path, small)
    _save(tmp_path, large)
    bundle = _save(tmp_path, small + large)
    monkeypatch.setattr(datasets, "MAX_DATA_BYTES", 1024)
    reader = DatasetReader(tmp_path)
    result = _choices(reader)
    assert Path(bundle["path"]).stat().st_size > datasets.MAX_DATA_BYTES
    assert result["mmlu-pro"]["eligible"]
    assert result["mmlu-pro"]["source_ids"] == [standalone["id"]]
    assert not result["livecodebench"]["eligible"]
    assert result["livecodebench"]["reason_code"] == "source_size_limit"
    assert "Full content was verified" in result["livecodebench"]["reason"]
    assert reader.compose([standalone["id"]], ["mmlu-pro"])["id"] == standalone["id"]
    with pytest.raises(ValueError, match="size limit"):
        reader.compose([bundle["id"]], ["mmlu-pro"])


def test_streaming_proof_includes_answers_and_ignores_row_order(tmp_path):
    rows = [_row("first", "mmlu-pro"), _row("second", "mmlu-pro", "B")]
    original = _save(tmp_path, rows)
    reordered = _save(tmp_path, list(reversed(rows)))
    assert original["id"] != reordered["id"]
    reader = DatasetReader(tmp_path)
    assert _choices(reader)["mmlu-pro"]["eligible"]
    changed = [dict(rows[0], answer="B"), rows[1]]
    _save(tmp_path, changed)
    result = _choices(reader)["mmlu-pro"]
    assert not result["eligible"] and result["reason_code"] == "source_conflict"


def test_stat_keyed_fingerprint_cache_rechecks_tampering(tmp_path, monkeypatch):
    saved = _save(tmp_path, [_row("first", "mmlu-pro")])
    reader = DatasetReader(tmp_path)
    checked = []
    validate = datasets._validate_frozen_case

    def count_validation(case, manifest, identities):
        checked.append(case["id"])
        return validate(case, manifest, identities)

    monkeypatch.setattr(datasets, "_validate_frozen_case", count_validation)
    assert _choices(reader)["mmlu-pro"]["eligible"]
    assert _choices(reader)["mmlu-pro"]["eligible"]
    assert checked == ["first"]
    path = Path(saved["path"])
    path.write_bytes(path.read_bytes().replace(b'"answer":"A"', b'"answer":"B"'))
    with pytest.raises(ValueError, match="digest"):
        _choices(reader)
    assert checked == ["first", "first"]


def test_fingerprint_cache_is_bounded_and_contains_no_case_content(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(datasets, "MAX_FINGERPRINT_CACHE_ENTRIES", 1)
    _save(tmp_path, [_row("first", "mmlu-pro", "hidden-reference-first")])
    _save(tmp_path, [_row("second", "gpqa-diamond", "hidden-reference-second")])
    reader = DatasetReader(tmp_path)
    result = _choices(reader)
    assert result["mmlu-pro"]["eligible"] and result["gpqa-diamond"]["eligible"]
    assert len(reader.fingerprint_cache) == 1
    assert reader.fingerprint_cache_bytes <= datasets.MAX_FINGERPRINT_CACHE_BYTES
    assert "hidden-reference" not in json.dumps(list(reader.fingerprint_cache.values()))


def test_fingerprint_index_limit_is_explicit_and_blocks_unverified_family(
    tmp_path, monkeypatch
):
    _save(tmp_path, [_row("first", "mmlu-pro"), _row("second", "mmlu-pro")])
    monkeypatch.setattr(datasets, "MAX_FINGERPRINT_INDEX_BYTES", 300)
    result = _choices(DatasetReader(tmp_path))["mmlu-pro"]
    assert not result["eligible"] and not result["source_ids"]
    assert "fingerprint index limit" in result["reason"]


def test_sources_rejected_early_still_consume_scan_budget(tmp_path, monkeypatch):
    manifests = [
        _save(tmp_path, [_row(identity, "mmlu-pro", "x" * 2048)])
        for identity in ("first", "second")
    ]
    maximum = max(Path(manifest["path"]).stat().st_size for manifest in manifests)
    monkeypatch.setattr(datasets, "MAX_DATA_BYTES", 1024)
    monkeypatch.setattr(datasets, "MAX_FINGERPRINT_SCAN_BYTES", maximum)
    result = _choices(DatasetReader(tmp_path))["mmlu-pro"]
    assert not result["eligible"] and not result["source_ids"]
    assert result["reason_code"] == "scan_budget_exhausted"
