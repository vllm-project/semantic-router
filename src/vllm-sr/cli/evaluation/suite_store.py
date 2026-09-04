"""Private, immutable storage for normalized external benchmark suites."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from pydantic import Field, ValidationError

from cli.evaluation.benchmark_registry import (
    BenchmarkAdapterDescriptor,
    get_benchmark_adapter,
)
from cli.evaluation.benchmark_sources import (
    SourceVerificationError,
    require_verified_benchmark_source,
)
from cli.evaluation.canonical import canonical_json_bytes, digest_value, sha256_digest
from cli.evaluation.catalog import CatalogSuite
from cli.evaluation.contracts import (
    ArtifactRef,
    CaseGrading,
    CaseVisible,
    StrictModel,
)
from cli.evaluation.suite_contract import (
    BenchmarkSourceReceipt,
    BenchmarkSuiteManifest,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedFault,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPerturbation,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
    SuiteArtifactSet,
)
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    BenchmarkSuiteInstallRequest,
    NormalizedMediaEntry,
    SuiteArtifactInstall,
    SuiteArtifactRole,
    SuiteLicenseManifest,
)

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_CHUNK_BYTES = 1024 * 1024
_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_MAX_LICENSE_BYTES = 1024 * 1024
_PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

JSONLRecord: TypeAlias = (
    CaseVisible
    | CaseGrading
    | NormalizedCapacityObservation
    | NormalizedDecision
    | NormalizedOutcome
    | NormalizedMultimodalObservation
    | NormalizedPreference
    | NormalizedSafetyObservation
    | NormalizedTrajectoryStep
    | NormalizedPerturbation
    | NormalizedFault
    | NormalizedMediaEntry
)

_JSONL_MODELS: dict[SuiteArtifactRole, type[StrictModel]] = {
    "visible_cases": CaseVisible,
    "grading_cases": CaseGrading,
    "outcomes": NormalizedOutcome,
    "decisions": NormalizedDecision,
    "preferences": NormalizedPreference,
    "trajectories": NormalizedTrajectoryStep,
    "perturbations": NormalizedPerturbation,
    "faults": NormalizedFault,
    "multimodal_observations": NormalizedMultimodalObservation,
    "safety_observations": NormalizedSafetyObservation,
    "capacity_observations": NormalizedCapacityObservation,
    "media_manifest": NormalizedMediaEntry,
}

_EVIDENCE_ORDER = {f"E{level}": level for level in range(6)}


class SuiteStoreError(ValueError):
    """A normalized suite was unsafe, corrupt, or inconsistent."""


class _SuiteIndexRecord(StrictModel):
    id: str
    revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_size_bytes: int = Field(ge=0, strict=True)


def _suite_identity(manifest: BenchmarkSuiteManifest) -> dict[str, Any]:
    """Return the non-self-referential identity embedded by ``revision``."""

    return manifest.model_dump(mode="json", exclude={"revision"}, exclude_none=True)


class NormalizedSuiteStore:
    """Install and stream normalized suites without exposing raw data publicly.

    Each suite ID is immutable in a store.  A changed suite must use a new ID.
    Artifacts are content-addressed independently in visible, grading, and
    metadata domains; manifests themselves are addressed by the SHA256 of their
    canonical bytes.
    """

    def __init__(self, root: str | Path):
        expanded = Path(root).expanduser()
        if expanded.is_symlink():
            raise SuiteStoreError("suite store root must not be a symlink")
        self.root = expanded.absolute()
        self._ensure_private_dir(self.root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests" / "sha256"
        self.index = self.root / "index"
        self._ensure_private_dir(self.objects)
        for domain in ("visible", "grading", "metadata"):
            self._ensure_private_dir(self.objects / domain)
            self._ensure_private_dir(self.objects / domain / "sha256")
        self._ensure_private_dir(self.root / "manifests")
        self._ensure_private_dir(self.manifests)
        self._ensure_private_dir(self.index)

    @staticmethod
    def _ensure_private_dir(path: Path) -> None:
        if path.exists() or path.is_symlink():
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise SuiteStoreError("suite store directory is not readable") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise SuiteStoreError("symlink is not allowed in suite store")
            if not stat.S_ISDIR(metadata.st_mode):
                raise SuiteStoreError("suite store path is not a directory")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode != _PRIVATE_DIR_MODE:
                raise SuiteStoreError(
                    f"suite store directory must have mode 0700, got {mode:04o}"
                )
            return
        try:
            path.mkdir(parents=True, mode=_PRIVATE_DIR_MODE)
            os.chmod(path, _PRIVATE_DIR_MODE)
        except OSError as exc:
            raise SuiteStoreError("could not create private suite store") from exc
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != _PRIVATE_DIR_MODE:
            raise SuiteStoreError("could not enforce suite directory mode 0700")

    @staticmethod
    def _validate_private_file(path: Path) -> os.stat_result:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SuiteStoreError("suite store file is missing or unreadable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SuiteStoreError("symlink is not allowed in suite store")
        if not stat.S_ISREG(metadata.st_mode):
            raise SuiteStoreError("suite store object is not a regular file")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != _PRIVATE_FILE_MODE:
            raise SuiteStoreError(
                f"suite store file must have mode 0600, got {mode:04o}"
            )
        return metadata

    @staticmethod
    def _portable_id(value: str) -> str:
        if not _PORTABLE_ID_RE.fullmatch(value):
            raise SuiteStoreError("invalid suite id")
        return value

    @staticmethod
    def _digest_hex(digest: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise SuiteStoreError("invalid SHA256 digest")
        return digest.removeprefix("sha256:")

    def _object_path(self, role: SuiteArtifactRole, digest: str) -> Path:
        _, _, domain = ARTIFACT_ROLE_LAYOUT[role]
        return self.objects / domain / "sha256" / self._digest_hex(digest)

    def _index_path(self, suite_id: str) -> Path:
        return self.index / f"{self._portable_id(suite_id)}.json"

    def _manifest_path(self, digest: str) -> Path:
        return self.manifests / self._digest_hex(digest)

    @staticmethod
    def _open_readonly(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if not no_follow:
            raise SuiteStoreError("this platform cannot enforce no-follow file access")
        try:
            descriptor = os.open(path, flags | no_follow)
        except OSError as exc:
            raise SuiteStoreError("could not safely open suite file") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise SuiteStoreError("suite input is not a regular file")
        return descriptor

    @staticmethod
    def _safe_bundle_root(bundle_root: str | Path) -> Path:
        root = Path(bundle_root).expanduser()
        if root.is_symlink():
            raise SuiteStoreError("bundle root must not be a symlink")
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise SuiteStoreError("bundle root is missing") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise SuiteStoreError("bundle root is not a directory")
        return root.resolve(strict=True)

    @staticmethod
    def _bundle_file(root: Path, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise SuiteStoreError("bundle artifact path is not a safe relative path")
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts[:-1]:
            current /= part
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise SuiteStoreError("bundle artifact parent is missing") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise SuiteStoreError("bundle artifact parent must not be a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise SuiteStoreError("bundle artifact parent is not a directory")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise SuiteStoreError(
                "bundle artifact is missing or escapes its root"
            ) from exc
        if candidate.is_symlink():
            raise SuiteStoreError("bundle artifact must not be a symlink")
        return candidate

    @staticmethod
    def _stream_digest_file(path: Path) -> tuple[str, int]:
        descriptor = NormalizedSuiteStore._open_readonly(path)
        digest = hashlib.sha256()
        total = 0
        try:
            while True:
                chunk = os.read(descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        finally:
            os.close(descriptor)
        return "sha256:" + digest.hexdigest(), total

    @classmethod
    def _verify_ref(cls, path: Path, ref: ArtifactRef) -> None:
        cls._validate_private_file(path)
        digest, size = cls._stream_digest_file(path)
        if digest != ref.digest or size != ref.size_bytes:
            raise SuiteStoreError(f"corrupt suite object {ref.digest}")

    @staticmethod
    def _validate_source_receipt(
        descriptor: BenchmarkAdapterDescriptor, receipt: BenchmarkSourceReceipt
    ) -> None:
        source_exact = (
            receipt.adapter_id == descriptor.id
            and receipt.verified
            and receipt.source_clean
            and receipt.expected_source_revision == descriptor.source_revision
            and receipt.observed_source_revision == descriptor.source_revision
        )
        if descriptor.dataset_revision is None:
            dataset_exact = (
                receipt.expected_dataset_revision is None
                and receipt.observed_dataset_revision is None
                and receipt.dataset_clean is None
            )
        else:
            dataset_exact = (
                receipt.expected_dataset_revision == descriptor.dataset_revision
                and receipt.observed_dataset_revision == descriptor.dataset_revision
                and receipt.dataset_clean is True
            )
        if not source_exact or not dataset_exact:
            raise SuiteStoreError(
                "suite source receipt is dirty or does not match the registry exact pin"
            )

    @staticmethod
    def _validate_request_against_adapter(
        request: BenchmarkSuiteInstallRequest,
        descriptor: BenchmarkAdapterDescriptor,
    ) -> None:
        if request.decision_unit != descriptor.decision_unit:
            raise SuiteStoreError("suite decision unit does not match its adapter")
        if request.action_space != descriptor.action_space:
            raise SuiteStoreError("suite action space does not match its adapter")
        if not set(request.track_ids).issubset(descriptor.track_ids):
            raise SuiteStoreError("suite declares a track not supported by its adapter")
        maximum_evidence = max(
            _EVIDENCE_ORDER[level] for level in descriptor.evidence_levels
        )
        if _EVIDENCE_ORDER[request.evidence_level_ceiling] > maximum_evidence:
            raise SuiteStoreError("suite evidence ceiling exceeds its adapter contract")

    @staticmethod
    def _validate_jsonl_descriptor(descriptor: int, role: SuiteArtifactRole) -> int:
        model = _JSONL_MODELS[role]
        os.lseek(descriptor, 0, os.SEEK_SET)
        count = 0
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            while True:
                line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > _MAX_JSONL_LINE_BYTES:
                    raise SuiteStoreError("normalized JSONL record is too large")
                if not line.endswith(b"\n"):
                    raise SuiteStoreError(
                        "normalized JSONL must end every record with LF"
                    )
                if not line.strip():
                    raise SuiteStoreError(
                        "normalized JSONL cannot contain blank records"
                    )
                try:
                    value = json.loads(line)
                    model.model_validate(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SuiteStoreError(
                        f"invalid normalized {role} record at line {count + 1}"
                    ) from exc
                count += 1
        if count == 0:
            raise SuiteStoreError(f"normalized {role} artifact cannot be empty")
        return count

    @staticmethod
    def _validate_license_descriptor(descriptor: int, size: int) -> int:
        if size > _MAX_LICENSE_BYTES:
            raise SuiteStoreError("license manifest exceeds its fixed size limit")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            try:
                value = json.load(handle)
                SuiteLicenseManifest.model_validate(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise SuiteStoreError("invalid normalized license manifest") from exc
        return 1

    def _publish_temp_object(
        self, temporary: Path, target: Path, ref: ArtifactRef
    ) -> None:
        if target.exists() or target.is_symlink():
            self._verify_ref(target, ref)
            temporary.unlink()
            return
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            self._verify_ref(target, ref)
        except OSError as exc:
            raise SuiteStoreError("could not publish suite CAS object") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        self._verify_ref(target, ref)

    def _stage_artifact(
        self,
        root: Path,
        artifact: SuiteArtifactInstall,
    ) -> tuple[ArtifactRef, int]:
        source = self._bundle_file(root, artifact.relative_path)
        source_descriptor = self._open_readonly(source)
        initial = os.fstat(source_descriptor)
        if initial.st_size != artifact.size_bytes:
            os.close(source_descriptor)
            raise SuiteStoreError(
                f"bundle artifact {artifact.role!r} does not match its declared size"
            )
        _, expected_media_type, domain = ARTIFACT_ROLE_LAYOUT[artifact.role]
        object_directory = self.objects / domain / "sha256"
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".install-", dir=object_directory
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            os.fchmod(temporary_descriptor, _PRIVATE_FILE_MODE)
            while True:
                chunk = os.read(source_descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(temporary_descriptor, view)
                    if written <= 0:
                        raise SuiteStoreError(
                            "short write while staging suite artifact"
                        )
                    view = view[written:]
            os.fsync(temporary_descriptor)
            final = os.fstat(source_descriptor)
            if (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
            ) != (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
            ):
                raise SuiteStoreError("bundle artifact changed while it was installed")
            observed_digest = "sha256:" + digest.hexdigest()
            if observed_digest != artifact.digest or size != artifact.size_bytes:
                raise SuiteStoreError(
                    f"bundle artifact {artifact.role!r} does not match its digest and size"
                )
            ref = ArtifactRef(
                digest=observed_digest,
                size_bytes=size,
                media_type=expected_media_type,
            )
            if expected_media_type == "application/x-ndjson":
                record_count = self._validate_jsonl_descriptor(
                    temporary_descriptor, artifact.role
                )
            else:
                record_count = self._validate_license_descriptor(
                    temporary_descriptor, size
                )
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            os.close(source_descriptor)
            os.close(temporary_descriptor)
        target = self._object_path(artifact.role, ref.digest)
        self._publish_temp_object(temporary, target, ref)
        return ref, record_count

    @staticmethod
    def _write_immutable(path: Path, data: bytes) -> None:
        if path.exists() or path.is_symlink():
            NormalizedSuiteStore._validate_private_file(path)
            digest, size = NormalizedSuiteStore._stream_digest_file(path)
            if digest != sha256_digest(data) or size != len(data):
                raise SuiteStoreError(
                    "immutable suite metadata already has other content"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise SuiteStoreError("short write while publishing suite metadata")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                NormalizedSuiteStore._validate_private_file(path)
                digest, size = NormalizedSuiteStore._stream_digest_file(path)
                if digest != sha256_digest(data) or size != len(data):
                    raise SuiteStoreError(
                        "immutable suite metadata already has other content"
                    ) from None
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary.exists():
                temporary.unlink()

    def install(
        self,
        request: BenchmarkSuiteInstallRequest,
        bundle_root: str | Path,
        *,
        source_root: str | Path,
    ) -> BenchmarkSuiteManifest:
        """Validate and atomically publish one normalized suite.

        Reinstalling exactly the same suite is idempotent.  Reusing its ID for
        different content is rejected rather than silently moving a catalog
        alias.
        """

        # ``model_copy(update=...)`` intentionally skips Pydantic validation.
        # Re-parse at the trust boundary so a preconstructed object cannot
        # bypass the closed role/path/media contract.
        try:
            request = BenchmarkSuiteInstallRequest.model_validate(
                request.model_dump(mode="json")
            )
        except (AttributeError, ValidationError) as exc:
            raise SuiteStoreError("invalid suite install request") from exc
        descriptor = get_benchmark_adapter(request.adapter_id)
        try:
            source_receipt = require_verified_benchmark_source(descriptor, source_root)
        except SourceVerificationError as exc:
            raise SuiteStoreError(str(exc)) from exc
        request = request.model_copy(update={"source_receipt": source_receipt})
        self._validate_source_receipt(descriptor, request.source_receipt)
        self._validate_request_against_adapter(request, descriptor)
        root = self._safe_bundle_root(bundle_root)

        refs: dict[SuiteArtifactRole, ArtifactRef] = {}
        counts: dict[SuiteArtifactRole, int] = {}
        for artifact in sorted(request.artifacts, key=lambda item: item.role):
            ref, count = self._stage_artifact(root, artifact)
            refs[artifact.role] = ref
            counts[artifact.role] = count
        if counts["visible_cases"] != request.case_count:
            raise SuiteStoreError("visible case count does not match install metadata")
        if counts["grading_cases"] != request.case_count:
            raise SuiteStoreError("grading case count does not match install metadata")

        artifact_set = SuiteArtifactSet(**refs)
        manifest_fields: dict[str, Any] = {
            "id": request.id,
            "name": request.name,
            "adapter_id": request.adapter_id,
            "source_receipt": request.source_receipt,
            "decision_unit": request.decision_unit,
            "action_space": request.action_space,
            "track_ids": request.track_ids,
            "evidence_level_ceiling": request.evidence_level_ceiling,
            "split_protocol": request.split_protocol,
            "case_count": request.case_count,
            "arm_ids": request.arm_ids,
            "data_classification": request.data_classification,
            "redistribution": request.redistribution,
            "artifacts": artifact_set,
            "limitations": request.limitations,
        }
        manifest_seed = BenchmarkSuiteManifest(
            revision="sha256:" + "0" * 64, **manifest_fields
        )
        revision = digest_value(_suite_identity(manifest_seed))
        manifest = manifest_seed.model_copy(update={"revision": revision})
        if digest_value(_suite_identity(manifest)) != manifest.revision:
            raise SuiteStoreError(
                "suite revision does not match its immutable identity"
            )

        manifest_bytes = canonical_json_bytes(manifest)
        manifest_digest = sha256_digest(manifest_bytes)
        manifest_path = self._manifest_path(manifest_digest)
        index_record = _SuiteIndexRecord(
            id=manifest.id,
            revision=manifest.revision,
            manifest_digest=manifest_digest,
            manifest_size_bytes=len(manifest_bytes),
        )
        index_bytes = canonical_json_bytes(index_record)
        index_path = self._index_path(manifest.id)

        if index_path.exists() or index_path.is_symlink():
            existing = self.get_suite_manifest(manifest.id)
            if existing != manifest:
                raise SuiteStoreError(
                    "suite id is immutable and already refers to different content"
                )
            return existing

        self._write_immutable(manifest_path, manifest_bytes)
        self._write_immutable(index_path, index_bytes)
        installed = self.get_suite_manifest(manifest.id)
        if installed != manifest:
            raise SuiteStoreError("installed suite manifest did not round trip")
        return installed

    def _read_index(self, suite_id: str) -> _SuiteIndexRecord:
        path = self._index_path(suite_id)
        self._validate_private_file(path)
        descriptor = self._open_readonly(path)
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SuiteStoreError("invalid suite index record") from exc
        finally:
            os.close(descriptor)
        try:
            record = _SuiteIndexRecord.model_validate(value)
        except ValueError as exc:
            raise SuiteStoreError("invalid suite index record") from exc
        if record.id != suite_id:
            raise SuiteStoreError("suite index record belongs to another ID")
        return record

    def get_suite_manifest(self, suite_id: str) -> BenchmarkSuiteManifest:
        record = self._read_index(suite_id)
        path = self._manifest_path(record.manifest_digest)
        ref = ArtifactRef(
            digest=record.manifest_digest,
            size_bytes=record.manifest_size_bytes,
            media_type="application/json",
        )
        self._verify_ref(path, ref)
        descriptor = self._open_readonly(path)
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SuiteStoreError("invalid suite manifest") from exc
        finally:
            os.close(descriptor)
        try:
            manifest = BenchmarkSuiteManifest.model_validate(value)
        except ValueError as exc:
            raise SuiteStoreError("invalid suite manifest") from exc
        if manifest.id != suite_id or manifest.revision != record.revision:
            raise SuiteStoreError("suite manifest does not match its index")
        if digest_value(_suite_identity(manifest)) != manifest.revision:
            raise SuiteStoreError("suite manifest revision is corrupt")
        return manifest

    @staticmethod
    def _catalog_suite(manifest: BenchmarkSuiteManifest) -> CatalogSuite:
        descriptor = get_benchmark_adapter(manifest.adapter_id)
        return CatalogSuite(
            id=manifest.id,
            name=manifest.name,
            description=(
                f"Pinned, operator-normalized {descriptor.name} recorded replay. "
                "Raw cases, labels, outcomes, and artifact references remain "
                "private; E1-E5 require adapter execution attestation."
            ),
            track_ids=manifest.track_ids,
            modes=("replay",),
            evidence_level="E0",
            case_count=manifest.case_count,
            revision=manifest.revision,
            tags=(
                "external",
                "pinned",
                "recorded-policy",
                "normalization-unattested",
                f"adapter:{manifest.adapter_id}",
                f"classification:{manifest.data_classification}",
                f"redistribution:{manifest.redistribution}",
            ),
        )

    def get_catalog_suite(self, suite_id: str) -> CatalogSuite:
        """Return browser-safe metadata, never artifact references or records."""

        return self._catalog_suite(self.get_suite_manifest(suite_id))

    def list_catalog_suites(self) -> tuple[CatalogSuite, ...]:
        """List browser-safe metadata in stable ID order."""

        suites: list[CatalogSuite] = []
        for path in sorted(self.index.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise SuiteStoreError("symlink is not allowed in suite index")
            if path.suffix != ".json":
                raise SuiteStoreError("unexpected file in suite index")
            suites.append(self.get_catalog_suite(path.stem))
        return tuple(suites)

    @staticmethod
    def _artifact_ref(
        manifest: BenchmarkSuiteManifest, role: SuiteArtifactRole
    ) -> ArtifactRef:
        ref = getattr(manifest.artifacts, role)
        if ref is None:
            raise SuiteStoreError(f"suite has no {role!r} artifact")
        return ref

    @classmethod
    def _iter_jsonl_path(
        cls, path: Path, ref: ArtifactRef, role: SuiteArtifactRole
    ) -> Iterator[JSONLRecord]:
        cls._verify_ref(path, ref)
        model = _JSONL_MODELS[role]
        descriptor = cls._open_readonly(path)
        count = 0
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                while True:
                    line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
                    if not line:
                        break
                    if len(line) > _MAX_JSONL_LINE_BYTES or not line.endswith(b"\n"):
                        raise SuiteStoreError("corrupt normalized JSONL record framing")
                    try:
                        value = json.loads(line)
                        yield model.model_validate(value)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise SuiteStoreError(
                            f"corrupt normalized {role} record at line {count + 1}"
                        ) from exc
                    count += 1
        finally:
            os.close(descriptor)

    def load_jsonl(
        self, suite_id: str, role: SuiteArtifactRole
    ) -> Iterator[JSONLRecord]:
        """Stream strict private records for a trusted executor or grader.

        The license manifest is JSON rather than JSONL and is intentionally not
        accepted here.  Callers must explicitly select a role; no path is ever
        accepted by this API.
        """

        if role not in ARTIFACT_ROLE_LAYOUT:
            raise SuiteStoreError("unknown suite artifact role")
        if role == "license_manifest":
            raise SuiteStoreError("license_manifest is JSON, not JSONL")
        manifest = self.get_suite_manifest(suite_id)
        ref = self._artifact_ref(manifest, role)
        path = self._object_path(role, ref.digest)
        yield from self._iter_jsonl_path(path, ref, role)
