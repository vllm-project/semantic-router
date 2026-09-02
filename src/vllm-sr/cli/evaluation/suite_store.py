"""Public facade for the private normalized benchmark suite store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_install_contract import (
    BenchmarkSuiteInstallRequest,
    SuiteArtifactRole,
)
from cli.evaluation.suite_store_cas import SuiteCAS
from cli.evaluation.suite_store_index import SuiteManifestIndex
from cli.evaluation.suite_store_install import SuiteInstaller
from cli.evaluation.suite_store_records import JSONLRecord, SuiteRecordReader


class NormalizedSuiteStore:
    """Trusted facade over CAS, manifest-index, installer, and record readers."""

    def __init__(self, root: str | Path):
        self._initialize(root, create=True)

    @classmethod
    def open_read_only(cls, root: str | Path) -> NormalizedSuiteStore:
        store = cls.__new__(cls)
        store._initialize(root, create=False)
        return store

    def _initialize(self, root: str | Path, *, create: bool) -> None:
        self._cas = SuiteCAS(root, create=create)
        self._index = SuiteManifestIndex(self._cas)
        self._records = SuiteRecordReader(self._cas)
        self._installer = SuiteInstaller(self._cas, self._index, self._records)

        # Filesystem roots are exposed for operational inspection and backup;
        # every read/write still goes through the security-enforcing owners.
        self.root = self._cas.root
        self.objects = self._cas.objects
        self.manifests = self._cas.manifests
        self.index = self._cas.index

    def install(
        self,
        request: BenchmarkSuiteInstallRequest,
        bundle_root: str | Path,
        *,
        source_root: str | Path,
        native_export_root: str | Path | None = None,
    ) -> BenchmarkSuiteManifest:
        return self._installer.install(
            request,
            bundle_root,
            source_root=source_root,
            native_export_root=native_export_root,
        )

    def get_suite_manifest(self, suite_id: str) -> BenchmarkSuiteManifest:
        return self._index.get(suite_id)

    def list_suite_manifests(self) -> tuple[BenchmarkSuiteManifest, ...]:
        """List validated private manifests in stable ID order."""

        return self._index.list()

    def load_jsonl(
        self, suite_id: str, role: SuiteArtifactRole
    ) -> Iterator[JSONLRecord]:
        """Stream strict private records for a trusted executor or grader."""

        manifest = self._index.get(suite_id)
        yield from self._records.load(manifest, role)
