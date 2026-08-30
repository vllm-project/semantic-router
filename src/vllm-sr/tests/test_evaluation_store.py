from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from cli.evaluation.store import LocalArtifactStore, StoreError


def test_cas_deduplicates_and_verifies_content(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first = store.put_bytes(b"same evidence", "text/plain")
    second = store.put_bytes(b"same evidence", "text/plain")

    assert first == second
    assert store.read_bytes(first) == b"same evidence"
    assert len(list(store.objects.iterdir())) == 1


def test_store_rejects_traversal_and_symlinked_run_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(StoreError, match="invalid run id"):
        store.write_run_json("../escape", "status.json", {"status": "bad"})

    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store.runs / "linked-run")
    with pytest.raises(StoreError, match=r"symlink|escapes store root"):
        store.write_run_json("linked-run", "status.json", {"status": "bad"})
    with pytest.raises(StoreError, match="invalid run id"):
        store.read_run_text("../../escape", "report.md")


def test_store_rejects_symlinked_run_artifacts_inside_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.set_status("run-1", {"status": "running"})
    run_dir = store.runs / "run-1"
    os.symlink(run_dir / "status.json", run_dir / "metrics.json")
    with pytest.raises(StoreError, match="symlink"):
        store.write_run_json("run-1", "metrics.json", {"value": 1})
    with pytest.raises(StoreError, match="symlink"):
        store.read_run_json("run-1", "metrics.json")

    os.symlink(run_dir / "status.json", run_dir / "events.jsonl")
    with pytest.raises(StoreError, match="symlink"):
        store.append_event("run-1", {"type": "bad"})


def test_atomic_writes_leave_no_temporary_files_and_final_artifacts_are_immutable(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.write_run_json("run-1", "metrics.json", {"value": 1})
    store.write_run_json("run-1", "metrics.json", {"value": 1})
    assert not list((store.runs / "run-1").glob(".*"))

    with pytest.raises(StoreError, match="immutable"):
        store.write_run_json("run-1", "metrics.json", {"value": 2})


def test_status_is_atomic_mutable_control_state_for_standalone_runs(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.set_status("run-1", {"status": "running"})
    store.set_status("run-1", {"status": "completed"})
    assert store.read_run_json("run-1", "status.json") == {"status": "completed"}


def test_store_and_run_directories_are_private(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.set_status("run-1", {"status": "running"})
    directories = (
        store.root,
        store.root / "objects",
        store.objects,
        store.runs,
        store.index,
        store.runs / "run-1",
    )
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)


def test_store_rejects_preexisting_non_private_directory(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir(mode=0o755)
    with pytest.raises(StoreError, match="mode 0700"):
        LocalArtifactStore(root)


def test_store_rejects_a_symlinked_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    os.symlink(real, linked)

    with pytest.raises(StoreError, match="root must not be a symlink"):
        LocalArtifactStore(linked)
