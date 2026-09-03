"""Path-safe local CAS and append-only run evidence store."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from cli.evaluation.artifact_store_error import StoreError as _StoreError
from cli.evaluation.bundle import (
    REPORT_BUNDLE_OPTIONAL_NAMES,
    REPORT_BUNDLE_REQUIRED_NAMES,
    REPORT_TRANSACTION_ARTIFACT_NAMES,
    RUN_ARTIFACT_NAMES,
    artifact_media_type,
)
from cli.evaluation.canonical import (
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_digest,
    strict_json_loads,
)
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contract_validation import validate_canonical_uuid
from cli.evaluation.contracts import RunManifest
from cli.evaluation.private_filesystem import (
    DurablePrivateFilesystem,
)
from cli.evaluation.report_transaction import (
    ReportBundleTransaction,
    recover_report_transaction,
    report_transaction_pending,
)

if TYPE_CHECKING:
    from cli.evaluation.worker_report import WorkerEvent, WorkerRunState

_MAX_EVENT_TAIL_BYTES = 16 * 1024
_PROCESS_LOCK_STRIPES = 64


class _FilesystemRunLock:
    def __init__(self, filesystem: DurablePrivateFilesystem) -> None:
        self._filesystem = filesystem

    @contextmanager
    def acquire(self, run_dir: Path) -> Iterator[None]:
        with self._filesystem.exclusive_lock(
            run_dir / "artifacts.lock",
            "run artifact lock",
        ):
            yield

    @contextmanager
    def acquire_existing(self, run_dir: Path) -> Iterator[None]:
        with self._filesystem.exclusive_existing_lock(
            run_dir / "artifacts.lock",
            "existing run artifact lock",
        ):
            yield


class _ProcessRunLock:
    def __init__(self, filesystem: DurablePrivateFilesystem) -> None:
        del filesystem
        self._locks = tuple(threading.RLock() for _ in range(_PROCESS_LOCK_STRIPES))

    @contextmanager
    def acquire(self, run_dir: Path) -> Iterator[None]:
        lock = self._locks[hash(run_dir.name) % len(self._locks)]
        with lock:
            yield

    @contextmanager
    def acquire_existing(self, run_dir: Path) -> Iterator[None]:
        with self.acquire(run_dir):
            yield


RunLockStrategy = _FilesystemRunLock | _ProcessRunLock


class ArtifactStore:
    def __init__(
        self,
        root: str | Path,
        lock_strategy_type: type[RunLockStrategy],
    ) -> None:
        self._filesystem = DurablePrivateFilesystem.create(root)
        self.root = self._filesystem.root
        object_root = self.root / "objects"
        self._filesystem.ensure_private_directory(object_root, parents=True)
        self.objects = object_root / "sha256"
        self.runs = self.root / "runs"
        for directory in (self.objects, self.runs):
            self._filesystem.ensure_private_directory(directory, parents=True)
        self._run_lock_strategy = lock_strategy_type(self._filesystem)

    def _run_dir(self, run_id: str, *, create: bool = False) -> Path:
        try:
            validate_canonical_uuid(run_id)
        except ValueError as exc:
            raise _StoreError("invalid run id") from exc

        path = self._filesystem.within_root(self.runs / run_id)
        if create:
            self._filesystem.ensure_private_directory(path, parents=False)
        else:
            self._filesystem.directory_exists(path)
        return path

    @staticmethod
    def _object_hex(digest: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise _StoreError("invalid sha256 digest")
        return digest.removeprefix("sha256:")

    def _object_path(self, digest: str) -> Path:
        return self._filesystem.within_root(self.objects / self._object_hex(digest))

    @contextmanager
    def _run_lock(self, run_id: str) -> Iterator[Path]:
        run_dir = self._run_dir(run_id, create=True)
        with self._run_lock_strategy.acquire(run_dir):
            yield run_dir

    @contextmanager
    def _existing_run_lock(self, run_id: str) -> Iterator[Path | None]:
        run_dir = self._run_dir(run_id)
        if not self._filesystem.directory_exists(run_dir):
            yield None
            return
        with self._run_lock_strategy.acquire_existing(run_dir):
            yield run_dir

    def put_bytes(self, data: bytes, media_type: str) -> ArtifactRef:
        digest = sha256_digest(data)
        target = self._object_path(digest)
        existing = self._filesystem.read_private_file(target, missing_ok=True)
        if existing is not None:
            if existing != data:
                raise _StoreError(f"CAS object does not match digest {digest}")
        else:
            self._filesystem.atomic_write(target, data)
        return ArtifactRef(digest=digest, media_type=media_type, size_bytes=len(data))

    def put_json(self, value: Any) -> ArtifactRef:
        return self.put_bytes(canonical_json_bytes(value), "application/json")

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        data = self._filesystem.read_private_file(
            self._object_path(ref.digest),
            missing_ok=True,
        )
        if data is None:
            raise _StoreError(f"missing CAS object {ref.digest}")
        if len(data) != ref.size_bytes or sha256_digest(data) != ref.digest:
            raise _StoreError(f"corrupt CAS object {ref.digest}")
        return data

    def read_json(self, ref: ArtifactRef) -> Any:
        return strict_json_loads(self.read_bytes(ref))

    def read_run_json(self, run_id: str, name: str) -> Any:
        return strict_json_loads(self.read_run_bytes(run_id, name))

    def read_optional_run_json(self, run_id: str, name: str) -> Any | None:
        """Return None only when an allowed run artifact is genuinely absent."""

        if name not in RUN_ARTIFACT_NAMES:
            raise _StoreError(f"unsupported run artifact name: {name}")
        data = self._filesystem.read_private_file(
            self._run_dir(run_id) / name,
            missing_ok=True,
        )
        return None if data is None else strict_json_loads(data)

    def read_run_bytes(self, run_id: str, name: str) -> bytes:
        if name not in RUN_ARTIFACT_NAMES:
            raise _StoreError(f"unsupported run artifact name: {name}")
        data = self._filesystem.read_private_file(
            self._run_dir(run_id) / name,
            missing_ok=True,
        )
        if data is None:
            raise _StoreError(f"run artifact does not exist: {run_id}/{name}")
        return data

    def read_run_text(self, run_id: str, name: str) -> str:
        return self.read_run_bytes(run_id, name).decode("utf-8")

    def snapshot_published_report_bundle(
        self,
        run_id: str,
    ) -> Mapping[str, bytes] | None:
        """Atomically read one complete published bundle without repairing it."""

        with self._existing_run_lock(run_id) as run_dir:
            if run_dir is None:
                return None
            if report_transaction_pending(self._filesystem, run_dir):
                raise _StoreError("published report retains an incomplete transaction")
            entries = self._filesystem.directory_entries(run_dir)
            allowed = set(RUN_ARTIFACT_NAMES) | {"artifacts.lock", "execution.lock"}
            unknown = set(entries).difference(allowed)
            if unknown:
                raise _StoreError("published report directory contains unknown state")
            report_data = self._filesystem.read_private_file(
                run_dir / "report.json",
                missing_ok=True,
            )
            if report_data is None:
                if set(entries).intersection(REPORT_TRANSACTION_ARTIFACT_NAMES):
                    raise _StoreError(
                        "report bundle artifacts exist without the commit point"
                    )
                if self._filesystem.directory_entries(run_dir) != entries:
                    raise _StoreError("published report bundle changed during snapshot")
                return None
            names = REPORT_BUNDLE_REQUIRED_NAMES + REPORT_BUNDLE_OPTIONAL_NAMES
            files = {"report.json": report_data}
            for name in names:
                if name == "report.json":
                    continue
                data = self._filesystem.read_private_file(
                    run_dir / name,
                    missing_ok=True,
                )
                if data is not None:
                    files[name] = data
            missing = set(REPORT_BUNDLE_REQUIRED_NAMES).difference(files)
            if missing:
                raise _StoreError("published report bundle is incomplete")
            if self._filesystem.directory_entries(run_dir) != entries:
                raise _StoreError("published report bundle changed during snapshot")
            return MappingProxyType(files)

    def reference_staged_manifest(self, run_id: str) -> ArtifactRef:
        """Import the server-staged immutable manifest into CAS without rewriting it."""

        with self._run_lock(run_id) as run_dir:
            data = self._filesystem.read_private_file(
                run_dir / "run-manifest.json",
                missing_ok=True,
            )
            if data is None:
                raise _StoreError("staged run manifest does not exist")
        return self.put_bytes(data, "application/json")

    def _retain_transaction_artifact(self, name: str, data: bytes) -> ArtifactRef:
        return self.put_bytes(data, artifact_media_type(name))

    def _require_staged_manifest_bytes(
        self,
        run_dir: Path,
        manifest: RunManifest,
        operation: str,
    ) -> bytes:
        try:
            staged_data = self._filesystem.read_private_file(
                run_dir / "run-manifest.json"
            )
            if staged_data is None:
                raise _StoreError("staged run manifest is unavailable")
            staged = RunManifest.model_validate(strict_json_loads(staged_data))
        except (_StoreError, TypeError, ValueError) as exc:
            raise _StoreError(
                f"{operation} requires a valid staged run manifest"
            ) from exc
        if staged != manifest:
            raise _StoreError(f"{operation} belongs to another staged run manifest")
        return staged_data

    @contextmanager
    def report_bundle_transaction(
        self,
        manifest: RunManifest,
    ) -> Iterator[ReportBundleTransaction]:
        """Build and publish one immutable bundle under the run lock."""

        if not isinstance(manifest, RunManifest):
            raise TypeError("report publication requires a RunManifest")
        with self._run_lock(manifest.run_id) as run_dir:
            staged_data = self._require_staged_manifest_bytes(
                run_dir,
                manifest,
                "report publication",
            )
            transaction = ReportBundleTransaction(
                self._filesystem,
                self._retain_transaction_artifact,
                run_dir,
                manifest.run_id,
                manifest.manifest_digest,
                staged_data,
            )
            try:
                yield transaction
            except BaseException:
                transaction.close()
                raise
            if not transaction.committed:
                transaction.close()
                raise _StoreError("report bundle transaction was not committed")

    def _recover_report_bundle(self, manifest: RunManifest) -> bool:
        """Finish only staged publication owned by the exact durable manifest."""

        if not isinstance(manifest, RunManifest):
            raise TypeError("report recovery requires a RunManifest")
        with self._run_lock(manifest.run_id) as run_dir:
            if report_transaction_pending(self._filesystem, run_dir):
                staged_data = self._require_staged_manifest_bytes(
                    run_dir,
                    manifest,
                    "report recovery",
                )
            else:
                staged_data = b""
            return recover_report_transaction(
                self._filesystem,
                run_dir,
                manifest.run_id,
                manifest.manifest_digest,
                staged_data,
            )


class LocalArtifactStore(ArtifactStore):
    """Cross-process standalone store with explicit execution ownership."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root, _FilesystemRunLock)

    def stage_run_manifest(self, manifest: RunManifest) -> ArtifactRef:
        """Bind one immutable standalone manifest and retain its CAS identity."""

        data = pretty_json_bytes(manifest)
        with self._run_lock(manifest.run_id) as run_dir:
            target = self._filesystem.within_root(run_dir / "run-manifest.json")
            existing = self._filesystem.read_private_file(target, missing_ok=True)
            if existing is not None and existing != data:
                raise _StoreError(
                    "immutable run artifact already exists: run-manifest.json"
                )
            if existing is None:
                self._filesystem.atomic_write(target, data)
        return self.put_bytes(data, "application/json")

    def write_run_status(self, run: WorkerRunState) -> None:
        """Atomically replace standalone lifecycle state."""

        with self._run_lock(run.id) as run_dir:
            self._filesystem.atomic_write(
                self._filesystem.within_root(run_dir / "status.json"),
                pretty_json_bytes(run),
            )

    def append_event(self, run_id: str, event: WorkerEvent) -> None:
        """Append one typed standalone event to the durable ledger."""

        with self._run_lock(run_id) as run_dir:
            self._filesystem.append_private_file(
                run_dir / "events.jsonl",
                canonical_json_bytes(event) + b"\n",
            )

    def append_event_if_changed(self, run_id: str, event: WorkerEvent) -> bool:
        """Append one typed event unless it is already the ledger tail."""

        with self._run_lock(run_id) as run_dir:
            target = run_dir / "events.jsonl"
            data = canonical_json_bytes(event)
            if self._read_event_tail(target, missing_ok=True) == data:
                return False
            self._filesystem.append_private_file(target, data + b"\n")
        return True

    def _read_event_tail(
        self,
        path: Path,
        *,
        missing_ok: bool = False,
    ) -> bytes | None:
        snapshot = self._filesystem.read_private_file_tail(
            path,
            _MAX_EVENT_TAIL_BYTES,
            missing_ok=missing_ok,
        )
        if snapshot is None:
            return None
        size, tail = snapshot
        if not tail:
            return b""
        if not tail.endswith(b"\n"):
            raise _StoreError("evaluation event ledger framing is invalid")
        framed = tail[:-1]
        separator = framed.rfind(b"\n")
        if separator < 0:
            if size > _MAX_EVENT_TAIL_BYTES:
                raise _StoreError("evaluation event ledger tail exceeds its limit")
            return framed
        return framed[separator + 1 :]

    @contextmanager
    def execution_lease(self, run_id: str) -> Iterator[None]:
        """Serialize one complete standalone execution for a run identity."""

        run_dir = self._run_dir(run_id, create=True)
        with self._filesystem.exclusive_lock(
            run_dir / "execution.lock",
            "run execution lease",
        ):
            yield

    def recover_report_bundle(self, manifest: RunManifest) -> bool:
        return self._recover_report_bundle(manifest)


class WorkerArtifactStore(ArtifactStore):
    """Process-owned Dashboard staging store without control-plane lock files."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root, _ProcessRunLock)
