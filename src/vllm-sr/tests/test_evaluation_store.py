from __future__ import annotations

import json
import os
import stat
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from cli.evaluation import private_filesystem_mutation as filesystem_mutation_module
from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.bundle import PRIVATE_RECEIPT_PREFIX_NAMES, checksum_bytes
from cli.evaluation.canonical import (
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_digest,
)
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import RunManifest
from cli.evaluation.report_transaction import (
    ReportAlreadyPublishedError,
    ReportBundleTransaction,
)
from cli.evaluation.store import LocalArtifactStore, WorkerArtifactStore
from cli.evaluation.worker_report import WorkerEvent, WorkerRunProgress, WorkerRunState

_RUN_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "vllm-sr-evaluation:store-run"))
_LINKED_RUN_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "vllm-sr-evaluation:linked"))


def _run(
    status: Literal[
        "pending", "running", "sealing", "completed", "failed", "cancelled"
    ],
    run_id: str = _RUN_ID,
) -> WorkerRunState:
    return WorkerRunState(
        schema_version=SCHEMA_VERSION,
        id=run_id,
        client_request_id=run_id,
        name="Run 1",
        description="Store contract fixture",
        status=status,
        mode="replay",
        evidence_level="E0",
        target_id="fixture",
        change_profile="schema_adapter",
        suite_ids=("evaluation-smoke",),
        track_ids=("routing",),
        sample_limit=1,
        concurrency=1,
        seed=1,
        progress=WorkerRunProgress(percent=0, completed=0, total=1),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _event(sequence: int | str) -> WorkerEvent:
    return WorkerEvent(type="progress", message=f"event {sequence}")


def _write_minimal_report_bundle(
    transaction: ReportBundleTransaction,
    manifest: RunManifest,
    *,
    private_receipt: bytes | None = None,
) -> None:
    manifest_bytes = pretty_json_bytes(manifest)
    artifact_rows = [
        (
            "run-manifest.json",
            ArtifactRef(
                digest=sha256_digest(manifest_bytes),
                media_type="application/json",
                size_bytes=len(manifest_bytes),
            ),
        )
    ]
    for name in PRIVATE_RECEIPT_PREFIX_NAMES[1:]:
        if name.endswith(".jsonl"):
            reference = transaction.write_jsonl(name, ())
        else:
            reference = transaction.write_json(name, {"artifact": name})
        artifact_rows.append((name, reference))
    checksum_ref = transaction.write_bytes("checksums.sha256", b"")
    artifact_rows.append(("checksums.sha256", checksum_ref))
    transaction.write_bytes(
        "private-checksums.sha256",
        checksum_bytes(artifact_rows) if private_receipt is None else private_receipt,
    )
    transaction.write_json("report.json", {"artifact": "report.json"})


def _stage_manifest(store: LocalArtifactStore) -> RunManifest:
    fixture = Path(__file__).parent / "fixtures" / "evaluation" / "manifest.json"
    manifest = RunManifest.model_validate(json.loads(fixture.read_bytes()))
    manifest = manifest.with_semantic_updates(run_id=_RUN_ID)
    store.stage_run_manifest(manifest)
    return manifest


def test_cas_deduplicates_and_verifies_content(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first = store.put_bytes(b"same evidence", "text/plain")
    second = store.put_bytes(b"same evidence", "text/plain")

    assert first == second
    assert store.read_bytes(first) == b"same evidence"
    assert len(list(store.objects.iterdir())) == 1


def test_store_rejects_traversal_and_symlinked_run_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    for invalid_run_id in ("../escape", "portable-but-not-a-uuid", "run.with.dot"):
        with pytest.raises(StoreError, match="invalid run id"):
            store.append_event(invalid_run_id, _event("invalid"))

    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store.runs / _LINKED_RUN_ID)
    with pytest.raises(StoreError, match=r"symlink|escapes store root"):
        store.append_event(_LINKED_RUN_ID, _event("linked"))
    with pytest.raises(StoreError, match="invalid run id"):
        store.read_run_text("../../escape", "metrics.json")


def test_store_rejects_symlinked_run_artifacts_inside_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.write_run_status(_run("running"))
    run_dir = store.runs / _RUN_ID
    os.symlink(run_dir / "status.json", run_dir / "metrics.json")
    with pytest.raises(StoreError, match="symlink"):
        store.read_run_json(_RUN_ID, "metrics.json")
    with pytest.raises(StoreError, match="symlink"):
        store.read_optional_run_json(_RUN_ID, "metrics.json")

    os.symlink(run_dir / "status.json", run_dir / "events.jsonl")
    with pytest.raises(StoreError, match="symlink"):
        store.append_event(_RUN_ID, _event("bad"))


def test_optional_run_read_distinguishes_absence_from_corruption(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    assert store.read_optional_run_json(_RUN_ID, "report.json") is None

    run_dir = store._run_dir(_RUN_ID, create=True)
    report = run_dir / "report.json"
    report.write_bytes(
        b'{"schema_version":"evaluation.v1","duplicate":1,"duplicate":2}'
    )
    report.chmod(0o600)
    with pytest.raises(ValueError):
        store.read_optional_run_json(_RUN_ID, "report.json")


def test_published_snapshot_of_missing_or_unlocked_run_has_no_side_effects(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store.runs / _RUN_ID

    assert store.snapshot_published_report_bundle(_RUN_ID) is None
    assert not run_dir.exists()

    run_dir.mkdir(mode=0o700)
    assert store.snapshot_published_report_bundle(_RUN_ID) is None
    assert tuple(run_dir.iterdir()) == ()


def test_atomic_writes_leave_no_temporary_files_and_manifest_is_immutable(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)
    store.stage_run_manifest(manifest)
    assert not list((store.runs / _RUN_ID).glob(".*"))

    with pytest.raises(StoreError, match="immutable"):
        store.stage_run_manifest(manifest.with_semantic_updates(name="Changed run"))


def test_atomic_write_does_not_unlink_a_colliding_temporary_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store._run_dir(_RUN_ID, create=True)
    collision = run_dir / ".status.json.deadbeef"
    collision.write_bytes(b"another writer")
    collision.chmod(0o600)
    monkeypatch.setattr(
        filesystem_mutation_module.secrets,
        "token_hex",
        lambda _: "deadbeef",
    )

    with pytest.raises(StoreError, match="allocate an artifact temporary file"):
        store.write_run_status(_run("running"))

    assert collision.read_bytes() == b"another writer"


def test_report_transaction_requires_commit_and_cleans_owned_preparation(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)

    with (
        pytest.raises(StoreError, match="was not committed"),
        store.report_bundle_transaction(manifest),
    ):
        pass

    run_dir = store.runs / _RUN_ID
    assert not (run_dir / ".report-preparing").exists()

    preparing = run_dir / ".report-preparing"
    preparing.mkdir(mode=0o700)
    interrupted_temporary = preparing / ".metrics.json.abcdef12"
    interrupted_temporary.write_bytes(b"incomplete")
    interrupted_temporary.chmod(0o600)

    manifest = _stage_manifest(store)
    assert not store.recover_report_bundle(manifest)
    assert not preparing.exists()


def test_report_transaction_unknown_cleanup_is_zero_side_effect(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store._run_dir(_RUN_ID, create=True)
    preparing = run_dir / ".report-preparing"
    preparing.mkdir(mode=0o700)
    owned = preparing / "metrics.json"
    owned.write_bytes(b"owned\n")
    owned.chmod(0o600)
    unknown = preparing / "foreign-state"
    unknown.write_bytes(b"foreign\n")
    unknown.chmod(0o600)

    manifest = _stage_manifest(store)

    with pytest.raises(StoreError, match="unknown entry"):
        store.recover_report_bundle(manifest)

    assert owned.read_bytes() == b"owned\n"
    assert unknown.read_bytes() == b"foreign\n"


def test_report_transaction_preflights_every_mode_before_cleanup(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store._run_dir(_RUN_ID, create=True)
    preparing = run_dir / ".report-preparing"
    preparing.mkdir(mode=0o700)
    first = preparing / "cases.jsonl"
    first.write_bytes(b"owned\n")
    first.chmod(0o600)
    invalid = preparing / "metrics.json"
    invalid.write_bytes(b"insecure\n")
    invalid.chmod(0o644)

    manifest = _stage_manifest(store)

    with pytest.raises(StoreError, match="mode 0600"):
        store.recover_report_bundle(manifest)

    assert first.read_bytes() == b"owned\n"
    assert invalid.read_bytes() == b"insecure\n"


def test_report_transaction_rejects_an_existing_report_before_staging(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)
    report = store.runs / _RUN_ID / "report.json"
    report.write_bytes(pretty_json_bytes({"schema_version": "evaluation.v1"}))
    report.chmod(0o600)

    with (
        pytest.raises(ReportAlreadyPublishedError, match="already published"),
        store.report_bundle_transaction(manifest),
    ):
        raise AssertionError("existing report must reject transaction construction")

    assert not (store.runs / _RUN_ID / ".report-preparing").exists()


def test_report_transaction_does_not_classify_a_corrupt_report_as_published(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)
    run_dir = store._run_dir(_RUN_ID, create=True)
    report = run_dir / "report.json"
    report.write_bytes(b'{"duplicate":1,"duplicate":2}')
    report.chmod(0o600)

    with (
        pytest.raises(StoreError, match="report bundle is corrupt"),
        store.report_bundle_transaction(manifest),
    ):
        raise AssertionError("corrupt report must reject transaction construction")

    assert not (run_dir / ".report-preparing").exists()


def test_report_transaction_rejects_an_incomplete_bundle(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)

    with (
        pytest.raises(ValueError, match="artifact set is incomplete"),
        store.report_bundle_transaction(manifest) as transaction,
    ):
        transaction.write_json("report.json", {"schema_version": "evaluation.v1"})
        transaction.commit()

    assert not (store.runs / _RUN_ID / ".report-preparing").exists()


def test_report_transaction_rejects_a_noncanonical_private_receipt(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)

    with (
        pytest.raises(StoreError, match="sealed report bundle"),
        store.report_bundle_transaction(manifest) as transaction,
    ):
        _write_minimal_report_bundle(
            transaction,
            manifest,
            private_receipt=b"0" * 64 + b"  run-manifest.json\n",
        )
        transaction.commit()

    assert not (store.runs / _RUN_ID / ".report-preparing").exists()


@pytest.mark.parametrize(
    ("manifest_mutation", "message"),
    (
        ("missing", "valid staged run manifest"),
        ("different", "another staged run manifest"),
    ),
)
def test_report_recovery_validates_the_full_staged_manifest_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_mutation: str,
    message: str,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _stage_manifest(store)
    filesystem = store._filesystem
    rename = filesystem.rename_private_directory

    def interrupt_promotion(source: Path, target: Path) -> None:
        if source.name == ".report-preparing":
            raise OSError("simulated stop before transaction promotion")
        rename(source, target)

    monkeypatch.setattr(filesystem, "rename_private_directory", interrupt_promotion)
    with (
        pytest.raises(OSError, match="transaction promotion"),
        store.report_bundle_transaction(manifest) as transaction,
    ):
        _write_minimal_report_bundle(transaction, manifest)
        transaction.commit()
    monkeypatch.setattr(filesystem, "rename_private_directory", rename)

    run_dir = store.runs / _RUN_ID
    preparing = run_dir / ".report-preparing"
    before = {path.name: path.read_bytes() for path in preparing.iterdir()}
    manifest_path = run_dir / "run-manifest.json"
    if manifest_mutation == "missing":
        manifest_path.unlink()
    else:
        changed = manifest.with_semantic_updates(code_revision="sha256:" + "2" * 64)
        manifest_path.write_bytes(pretty_json_bytes(changed))
        manifest_path.chmod(0o600)

    with pytest.raises(StoreError, match=message):
        store.recover_report_bundle(manifest)

    assert {path.name: path.read_bytes() for path in preparing.iterdir()} == before
    assert not (run_dir / ".report-transaction").exists()
    assert not (run_dir / "report.json").exists()


def test_report_publication_syncs_destination_before_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    filesystem = store._filesystem
    manifest = _stage_manifest(store)
    events: list[tuple[str, str]] = []
    active_target: str | None = None
    replace_private_file = filesystem.replace_private_file
    sync_directory = filesystem.sync_directory

    def observe_replace(
        source: Path,
        target: Path,
        *,
        expected_data: bytes,
    ) -> bool:
        nonlocal active_target
        moved = replace_private_file(
            source,
            target,
            expected_data=expected_data,
        )
        if moved:
            active_target = target.name
            events.append(("replace", target.name))
        return moved

    def observe_sync(path: Path) -> None:
        nonlocal active_target
        if active_target is not None:
            events.append(("sync", path.name))
            if path.name == ".report-transaction":
                active_target = None
        sync_directory(path)

    monkeypatch.setattr(filesystem, "replace_private_file", observe_replace)
    monkeypatch.setattr(filesystem, "sync_directory", observe_sync)
    with store.report_bundle_transaction(manifest) as transaction:
        _write_minimal_report_bundle(transaction, manifest)
        transaction.commit()

    publication_events = iter(events)
    published_names: list[str] = []
    for event in publication_events:
        assert event[0] == "replace"
        published_names.append(event[1])
        assert next(publication_events) == ("sync", _RUN_ID)
        assert next(publication_events) == ("sync", ".report-transaction")
    assert published_names[-1] == "report.json"


def test_report_publication_recovers_when_source_sync_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    filesystem = store._filesystem
    run_dir = store._run_dir(_RUN_ID, create=True)
    replace_private_file = filesystem.replace_private_file
    sync_directory = filesystem.sync_directory
    active_target: str | None = None
    destination_synced = False
    interrupted = False

    def observe_replace(
        source: Path,
        target: Path,
        *,
        expected_data: bytes,
    ) -> bool:
        nonlocal active_target
        moved = replace_private_file(
            source,
            target,
            expected_data=expected_data,
        )
        if moved:
            active_target = target.name
        return moved

    def interrupt_source_sync(path: Path) -> None:
        nonlocal destination_synced, interrupted
        if active_target is not None and path == run_dir:
            sync_directory(path)
            destination_synced = True
            return
        if (
            active_target is not None
            and destination_synced
            and path.name == ".report-transaction"
            and not interrupted
        ):
            interrupted = True
            raise OSError("simulated stop before source directory sync")
        sync_directory(path)

    monkeypatch.setattr(filesystem, "replace_private_file", observe_replace)
    monkeypatch.setattr(filesystem, "sync_directory", interrupt_source_sync)
    manifest = _stage_manifest(store)
    with (
        pytest.raises(OSError, match="source directory sync"),
        store.report_bundle_transaction(manifest) as transaction,
    ):
        _write_minimal_report_bundle(transaction, manifest)
        transaction.commit()

    assert interrupted and destination_synced
    assert (run_dir / ".report-transaction" / "transaction.json").is_file()
    monkeypatch.setattr(filesystem, "replace_private_file", replace_private_file)
    monkeypatch.setattr(filesystem, "sync_directory", sync_directory)

    assert store.recover_report_bundle(manifest)
    assert (run_dir / "report.json").is_file()
    assert not (run_dir / ".report-transaction").exists()


def test_report_transaction_never_discards_uncommitted_artifacts_after_publication(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store._run_dir(_RUN_ID, create=True)
    report = run_dir / "report.json"
    report.write_bytes(pretty_json_bytes({"schema_version": "evaluation.v1"}))
    report.chmod(0o600)
    transaction = run_dir / ".report-transaction"
    transaction.mkdir(mode=0o700)
    stranded = transaction / "metrics.json"
    stranded.write_bytes(b"{}\n")
    stranded.chmod(0o600)

    manifest = _stage_manifest(store)

    with pytest.raises(StoreError, match="retains artifacts"):
        store.recover_report_bundle(manifest)

    assert stranded.is_file()


def test_status_is_atomic_mutable_control_state_for_standalone_runs(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.write_run_status(_run("running"))
    store.write_run_status(_run("completed"))
    assert store.read_run_json(_RUN_ID, "status.json")["status"] == "completed"


def test_concurrent_store_initialization_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = tuple(pool.map(lambda _: LocalArtifactStore(root), range(32)))

    assert all(store.root == stores[0].root for store in stores)


def test_concurrent_control_writes_preserve_atomic_status_and_event_lines(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")

    def write_status(value: int) -> None:
        store.write_run_status(
            _run("running").model_copy(
                update={
                    "progress": WorkerRunProgress(
                        percent=value,
                        completed=0,
                        total=1,
                    )
                }
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(write_status, range(8)))
        tuple(
            pool.map(
                lambda index: store.append_event(_RUN_ID, _event(index)),
                range(64),
            )
        )

    status = store.read_run_json(_RUN_ID, "status.json")
    assert status["progress"]["percent"] in range(8)
    events = store.read_run_text(_RUN_ID, "events.jsonl").splitlines()
    assert len(events) == 64
    assert {json.loads(row)["message"] for row in events} == {
        f"event {index}" for index in range(64)
    }


def test_worker_store_exposes_no_control_plane_mutations(
    tmp_path: Path,
) -> None:
    store = WorkerArtifactStore(tmp_path / "store")
    assert not hasattr(store, "stage_run_manifest")
    assert not hasattr(store, "write_run_status")
    assert not hasattr(store, "append_event")
    assert not hasattr(store, "append_event_if_changed")


def test_idempotent_event_append_reads_a_bounded_large_ledger_tail(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    run_dir = store.runs / _RUN_ID
    run_dir.mkdir(mode=0o700)
    ledger = run_dir / "events.jsonl"
    terminal = _event("terminal")
    prefix = canonical_json_bytes(_event(0)) + b"\n"
    prefix *= 2_000
    tail = canonical_json_bytes(terminal) + b"\n"
    ledger.write_bytes(prefix + tail)
    ledger.chmod(0o600)
    assert not store.append_event_if_changed(_RUN_ID, terminal)
    assert ledger.read_bytes() == prefix + tail
    next_event = _event("next")
    assert store.append_event_if_changed(_RUN_ID, next_event)
    assert ledger.read_bytes().endswith(canonical_json_bytes(next_event) + b"\n")


def test_store_and_run_directories_are_private(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.write_run_status(_run("running"))
    directories = (
        store.root,
        store.root / "objects",
        store.objects,
        store.runs,
        store.runs / _RUN_ID,
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
