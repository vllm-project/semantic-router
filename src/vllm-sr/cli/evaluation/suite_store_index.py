"""Immutable normalized-suite manifest index and integrity validation."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from cli.evaluation.canonical import (
    canonical_json_bytes,
    digest_value,
    sha256_digest,
    strict_json_load,
)
from cli.evaluation.contract_primitives import ArtifactRef, StrictModel
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store_cas import SuiteCAS
from cli.evaluation.suite_store_error import SuiteStoreError


class SuiteIndexRecord(StrictModel):
    id: str
    revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_size_bytes: int = Field(ge=0, strict=True)


def suite_identity(manifest: BenchmarkSuiteManifest) -> dict[str, Any]:
    """Return the non-self-referential identity embedded by ``revision``."""

    return manifest.model_dump(mode="json", exclude={"revision"}, exclude_none=True)


class SuiteManifestIndex:
    """Own suite-ID immutability and manifest-to-CAS index integrity."""

    def __init__(self, cas: SuiteCAS):
        self._cas = cas

    def _read_index(self, suite_id: str) -> SuiteIndexRecord:
        path = self._cas.index_path(suite_id)
        self._cas.validate_private_file(path)
        descriptor = self._cas.open_readonly(path)
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                value = strict_json_load(handle)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SuiteStoreError("invalid suite index record") from exc
        finally:
            os.close(descriptor)
        try:
            record = SuiteIndexRecord.model_validate(value)
        except ValueError as exc:
            raise SuiteStoreError("invalid suite index record") from exc
        if record.id != suite_id:
            raise SuiteStoreError("suite index record belongs to another ID")
        return record

    def get(self, suite_id: str) -> BenchmarkSuiteManifest:
        record = self._read_index(suite_id)
        path = self._cas.manifest_path(record.manifest_digest)
        ref = ArtifactRef(
            digest=record.manifest_digest,
            size_bytes=record.manifest_size_bytes,
            media_type="application/json",
        )
        self._cas.verify_ref(path, ref)
        descriptor = self._cas.open_readonly(path)
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                value = strict_json_load(handle)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SuiteStoreError("invalid suite manifest") from exc
        finally:
            os.close(descriptor)
        try:
            manifest = BenchmarkSuiteManifest.model_validate(value)
        except ValueError as exc:
            raise SuiteStoreError("invalid suite manifest") from exc
        if manifest.id != suite_id or manifest.revision != record.revision:
            raise SuiteStoreError("suite manifest does not match its index")
        if digest_value(suite_identity(manifest)) != manifest.revision:
            raise SuiteStoreError("suite manifest revision is corrupt")
        return manifest

    def list(self) -> tuple[BenchmarkSuiteManifest, ...]:
        manifests: list[BenchmarkSuiteManifest] = []
        for path in sorted(self._cas.index.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise SuiteStoreError("symlink is not allowed in suite index")
            if path.suffix != ".json":
                raise SuiteStoreError("unexpected file in suite index")
            manifests.append(self.get(path.stem))
        return tuple(manifests)

    def publish(self, manifest: BenchmarkSuiteManifest) -> BenchmarkSuiteManifest:
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_digest = sha256_digest(manifest_bytes)
        manifest_path = self._cas.manifest_path(manifest_digest)
        index_record = SuiteIndexRecord(
            id=manifest.id,
            revision=manifest.revision,
            manifest_digest=manifest_digest,
            manifest_size_bytes=len(manifest_bytes),
        )
        index_bytes = canonical_json_bytes(index_record)
        index_path = self._cas.index_path(manifest.id)

        if index_path.exists() or index_path.is_symlink():
            existing = self.get(manifest.id)
            if existing != manifest:
                raise SuiteStoreError(
                    "suite id is immutable and already refers to different content"
                )
            return existing

        self._cas.write_immutable(manifest_path, manifest_bytes)
        self._cas.write_immutable(index_path, index_bytes)
        installed = self.get(manifest.id)
        if installed != manifest:
            raise SuiteStoreError("installed suite manifest did not round trip")
        return installed
