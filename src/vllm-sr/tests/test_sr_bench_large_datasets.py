"""Large immutable sources remain usable without loading grading assets en masse."""

import hashlib
import json
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
from cli.sr_bench import dataset_io, datasets, sources
from cli.sr_bench.contracts import canonical, digest
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset, combine_datasets
from cli.sr_bench.store import Store


def _rows(benchmark, count, asset_bytes=4096):
    for index in range(count):
        yield {
            "id": f"{benchmark}/{index:04d}",
            "benchmark": benchmark,
            "messages": [{"role": "user", "content": f"Question {index}"}],
            "answer": "private-reference",
            "metadata": {
                "split": "dev",
                "stratum": "coding",
                "tests": "x" * asset_bytes,
            },
        }


def _write(root, benchmark, count, asset_bytes=4096):
    return _write_dataset(
        root,
        _rows(benchmark, count, asset_bytes),
        "quick",
        7,
        {benchmark: {"revision": "fixture-v1"}},
    )


@pytest.mark.parametrize("benchmark", ["livecodebench", "mmlu-pro"])
def test_large_source_selection_compose_and_paginated_browse_through_service(
    tmp_path, monkeypatch, benchmark
):
    store = Store(tmp_path / "store")
    first = _write(store.root, benchmark, 12)
    second = _write(store.root, "gpqa-diamond", 2)
    monkeypatch.setattr(datasets, "MAX_ROW_BYTES", 8192)
    assert Path(first["path"]).stat().st_size > datasets.MAX_ROW_BYTES
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine, "start", lambda *a, **k: pytest.fail("No model run")
    )
    worker = threading.Thread(target=service.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {"Authorization": "Bearer fixture-token"}
    before = {
        manifest["id"]: Path(manifest["path"]).read_bytes()
        for manifest in (first, second)
    }
    try:
        selection = requests.get(
            base + "/datasets/selection?profile=quick", headers=headers, timeout=5
        )
        assert selection.status_code == 200, selection.text
        choices = {row["id"]: row for row in selection.json()["benchmarks"]}
        assert choices[benchmark]["eligible"]
        assert choices[benchmark]["case_count"] == 12
        composed = requests.post(
            base + "/datasets/compose",
            headers=headers,
            json={
                "dataset_ids": [first["id"], second["id"]],
                "benchmarks": [benchmark, "gpqa-diamond"],
            },
            timeout=5,
        )
        assert composed.status_code == 200, composed.text
        result = composed.json()["dataset"]
        expected = b"".join(
            (canonical(case) + "\n").encode()
            for benchmark, count in (("gpqa-diamond", 2), (benchmark, 12))
            for case in _rows(benchmark, count)
        )
        assert Path(result["path"]).read_bytes() == expected
        assert result["sha256"] == hashlib.sha256(expected).hexdigest()
        assert result["id"] == digest(
            {
                "cases_sha256": result["sha256"],
                "sources": result["sources"],
                "profile": "quick",
                "seed": 7,
                "custom_subset": False,
                "selection": "stratified-hash-v1",
            }
        )
        url = base + "/datasets/" + result["id"]
        detail = requests.get(url, headers=headers, timeout=5).json()
        assert detail["case_count"] == 14
        assert [b["count"] for b in detail["benchmarks"]] == [2, 12]
        cursor, seen = "0", []
        while cursor is not None:
            response = requests.get(
                url + "/cases",
                headers=headers,
                params={"benchmark": benchmark, "limit": 5, "cursor": cursor},
                timeout=5,
            )
            assert response.status_code == 200, response.text
            page = response.json()
            assert page["total"] == 12 and page["dataset_total"] == 14
            assert (
                "private-reference" not in response.text
                and '"tests"' not in response.text
            )
            seen.extend(case["id"] for case in page["cases"])
            cursor = page["next_cursor"]
        assert seen == [case["id"] for case in _rows(benchmark, 12)]
        assert (
            requests.post(
                base + "/datasets/compose",
                headers=headers,
                json={
                    "dataset_ids": [first["id"], second["id"]],
                    "benchmarks": ["gpqa-diamond", benchmark],
                },
                timeout=5,
            ).json()["dataset"]
            == result
        )
        if benchmark == "mmlu-pro":
            # Basic-adapter planning needs no external harness or model calls.
            # The same grading-heavy dataset reference stays compact over HTTP.
            reviewed = requests.post(
                base + "/plans",
                headers=headers,
                json={
                    "manifest": {
                        "version": "sr-bench-1.0",
                        "profile": "quick",
                        "seed": 7,
                        "dataset": result,
                        "targets": [
                            {
                                "id": "fixture-single",
                                "kind": "single",
                                "model": "fixture",
                                "base_url": "http://127.0.0.1:1/v1",
                                "prices": {
                                    "fixture": {
                                        "input": 1,
                                        "cached_input": 0.1,
                                        "cache_write": 1.25,
                                        "output": 3,
                                    }
                                },
                            }
                        ],
                    }
                },
                timeout=5,
            )
            assert reviewed.status_code == 200, reviewed.text
            review = reviewed.json()
            assert review["status"] == "validated" and review["model_requests"] == 0
            assert review["total"] == 14
            assert review["manifest"]["dataset"]["case_count"] == 14
            assert review["manifest"]["dataset"]["sha256"] == result["sha256"]
            assert "cases" not in review["manifest"]
            expected_rows = [
                case
                for family, count in (("gpqa-diamond", 2), (benchmark, 12))
                for case in _rows(family, count)
            ]
            assert review["manifest"]["case_sha256"] == digest(expected_rows)
            assert (
                '"tests"' not in reviewed.text
                and "private-reference" not in reviewed.text
            )
            assert len(reviewed.content) < 8192
        assert store.list() == []
        assert all(
            Path(manifest["path"]).read_bytes() == before[manifest["id"]]
            for manifest in (first, second)
        )
    finally:
        service.shutdown()
        service.server_close()
        worker.join(timeout=5)


def test_composition_and_public_pages_retain_bounded_memory(tmp_path, monkeypatch):
    first = _write(tmp_path, "livecodebench", 64, 128 * 1024)
    second = _write(tmp_path, "gpqa-diamond", 2)
    reader = DatasetReader(tmp_path)
    original_read = Path.read_bytes

    def reject_eager(path):
        if path.name == "cases.jsonl":
            pytest.fail("Dataset files must be streamed")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", reject_eager)
    tracemalloc.start()
    try:
        result = reader.compose(
            [first["id"], second["id"]], ["livecodebench", "gpqa-diamond"]
        )
        assert reader.page(result["id"], limit="3")["total"] == 66
        assert reader.detail(result["id"])["case_count"] == 66
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Eight MiB of private assets must not become an in-memory case list.
    assert peak < 4 * 1024 * 1024


@pytest.mark.parametrize("operation", ["page", "detail", "compose"])
def test_changed_source_cannot_publish_a_page_or_composed_output(
    tmp_path, monkeypatch, operation
):
    saved = _write(tmp_path, "livecodebench", 3)
    other = _write(tmp_path, "gpqa-diamond", 1)
    before = set((tmp_path / "datasets").iterdir())
    validate = datasets._validate_frozen_case
    changed = False

    def mutate(case, manifest, ids):
        nonlocal changed
        validate(case, manifest, ids)
        if not changed:
            changed = True
            path = Path(saved["path"])
            path.write_bytes(
                path.read_bytes().replace(b"private-reference", b"changed-reference")
            )

    monkeypatch.setattr(datasets, "_validate_frozen_case", mutate)
    reader = DatasetReader(tmp_path)
    with pytest.raises(ValueError, match=r"changed|digest"):
        if operation == "compose":
            reader.compose(
                [saved["id"], other["id"]], ["livecodebench", "gpqa-diamond"]
            )
        else:
            getattr(reader, operation)(saved["id"])
    assert set((tmp_path / "datasets").iterdir()) == before


def test_checked_file_cannot_be_swapped_for_symlink_before_open(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    external = tmp_path / "external"
    external.write_bytes(b"must-not-read")
    original_open = dataset_io.os.open

    def replace(name, flags, *args, **kwargs):
        if Path(name) == path:
            path.unlink()
            path.symlink_to(external)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(dataset_io.os, "open", replace)
    with pytest.raises(ValueError, match="regular file"):
        dataset_io.file_digest(path)


def test_failed_stream_never_publishes_partial_dataset(tmp_path):
    def broken():
        yield from _rows("livecodebench", 2)
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        _write_dataset(
            tmp_path, broken(), "quick", 7, {"livecodebench": {"revision": "v1"}}
        )
    assert list((tmp_path / "datasets").iterdir()) == []


def test_concurrent_identical_stream_writes_converge_without_replacing_files(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as workers:
        outputs = list(
            workers.map(lambda _: _write(tmp_path, "livecodebench", 12), range(2))
        )
    assert outputs[0] == outputs[1]
    path = Path(outputs[0]["path"])
    before = path.stat()
    assert _write(tmp_path, "livecodebench", 12) == outputs[0]
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_combine_stream_preserves_input_order_and_all_private_assets(
    tmp_path, monkeypatch
):
    first = _write(tmp_path, "livecodebench", 12)
    second = _write(tmp_path, "gpqa-diamond", 2)
    expected = Path(first["path"]).read_bytes() + Path(second["path"]).read_bytes()
    combined = combine_datasets([first, second], tmp_path)
    assert Path(combined["path"]).read_bytes() == expected
    monkeypatch.setattr(dataset_io, "CHUNK_BYTES", 97)
    Path(first["path"]).write_bytes(
        Path(first["path"])
        .read_bytes()
        .replace(b"private-reference", b"changed-reference")
    )
    before = set((tmp_path / "datasets").iterdir())
    with pytest.raises(ValueError, match="digest"):
        combine_datasets([first, second], tmp_path)
    assert set((tmp_path / "datasets").iterdir()) == before


def test_scan_limits_apply_to_browsing_and_composition(tmp_path, monkeypatch):
    saved = _write(tmp_path, "livecodebench", 3)
    monkeypatch.setattr(datasets, "MAX_FINGERPRINT_SCAN_BYTES", 1024)
    reader = DatasetReader(tmp_path)
    for action in (
        lambda: reader.page(saved["id"]),
        lambda: reader.detail(saved["id"]),
        lambda: reader.compose([saved["id"]], ["livecodebench"]),
    ):
        with pytest.raises(ValueError, match="size limit"):
            action()


def test_prepared_stage_is_not_visible_in_dataset_inventory(tmp_path, monkeypatch):
    original = sources.publish_dataset

    def inspect(directory, staged_data, sha, staged_manifest, rendered):
        assert not list((tmp_path / "datasets").glob("*/manifest.json"))
        assert DatasetReader(tmp_path).selection("quick")["model_requests"] == 0
        return original(directory, staged_data, sha, staged_manifest, rendered)

    monkeypatch.setattr(sources, "publish_dataset", inspect)
    _write(tmp_path, "livecodebench", 2)


def test_publication_uses_opened_directory_when_its_path_is_replaced(
    tmp_path, monkeypatch
):
    root = tmp_path / "store"
    external = tmp_path / "external"
    external.mkdir()
    original_link = dataset_io.os.link
    replaced = False

    def replace(source, destination, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            directory = next((root / "datasets").iterdir())
            directory.rename(root / "moved-dataset")
            directory.symlink_to(external, target_is_directory=True)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(dataset_io.os, "link", replace)
    with pytest.raises(ValueError, match="directory changed"):
        _write(root, "livecodebench", 2)
    assert list(external.iterdir()) == []


def test_page_byte_budget_advances_without_skipping_matching_cases(
    tmp_path, monkeypatch
):
    saved = _write(tmp_path, "livecodebench", 6)
    reader = DatasetReader(tmp_path)
    first = reader.page(saved["id"], limit="1")["cases"][0]
    monkeypatch.setattr(datasets, "MAX_PAGE_BYTES", len(json.dumps(first).encode()) * 2)
    cursor, seen = "0", []
    while cursor is not None:
        page = reader.page(saved["id"], cursor=cursor, limit="6")
        assert len(page["cases"]) == 2
        seen.extend(case["id"] for case in page["cases"])
        cursor = page["next_cursor"]
    assert seen == [row["id"] for row in _rows("livecodebench", 6)]
