"""Browser-safe projection of private normalized suite manifests."""

from __future__ import annotations

from collections.abc import Mapping

from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.catalog_suites import CatalogSuite
from cli.evaluation.catalog_tracks import CatalogMethod, CatalogMethodEvidenceSource
from cli.evaluation.executor_contracts import Mode
from cli.evaluation.executor_registry import ExecutorRegistry
from cli.evaluation.method_contract_v2 import EvaluationMethodPlugin
from cli.evaluation.method_registry_v2 import method_plugin_for_benchmark
from cli.evaluation.normalized_suite_live_admission import (
    NormalizedSuiteLiveAdmission,
    normalized_suite_live_admissions,
)
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore


class NormalizedSuiteCatalog:
    """Project suite metadata through explicit executor capability contracts."""

    def __init__(
        self,
        store: NormalizedSuiteStore,
        executor_registry: ExecutorRegistry,
        executor_ids: Mapping[Mode, str],
    ):
        if set(executor_ids) != {"replay", "live"}:
            raise ValueError(
                "normalized suite catalog requires explicit replay and live executors"
            )
        contracts = {
            mode: executor_registry.contract(executor_ids[mode])
            for mode in ("replay", "live")
        }
        for mode, contract in contracts.items():
            if contract.mode != mode or not contract.normalized_suite:
                raise ValueError(
                    f"executor {contract.id} cannot project normalized {mode} suites"
                )
        self._store = store
        self._contracts = contracts

    def get(self, suite_id: str) -> CatalogSuite:
        return self._project(self._store.get_suite_manifest(suite_id))

    def list(self) -> tuple[CatalogSuite, ...]:
        return tuple(
            self._project(manifest) for manifest in self._store.list_suite_manifests()
        )

    def _project(self, manifest: BenchmarkSuiteManifest) -> CatalogSuite:
        descriptor = get_benchmark_adapter(manifest.adapter_id)
        replay = self._contracts["replay"]
        plugin = method_plugin_for_benchmark(manifest.adapter_id)
        executors: dict[Mode, str] = {"replay": replay.id}
        live_admissions = normalized_suite_live_admissions(self._store, manifest)
        if live_admissions:
            live = self._contracts["live"]
            unsupported = sorted(
                admission.track_id
                for admission in live_admissions
                if admission.track_id not in live.track_ids
            )
            if unsupported:
                raise ValueError(
                    "normalized live method is unsupported by its executor: "
                    + ", ".join(unsupported)
                )
        import_evidence = manifest.qualification_receipt.qualification
        import_summary = (
            "The registered parser was rerun and produced the same records."
            if import_evidence.parser_verified
            else "The imported records passed the required data checks."
        )
        if live_admissions:
            executors["live"] = live.id
        return CatalogSuite(
            id=manifest.id,
            name=manifest.name,
            description=(
                f"Imported {descriptor.name} benchmark workload pinned to a specific "
                f"revision. {import_summary} Imported results are exploratory and do not "
                "attest the original benchmark run; release evidence requires a separately "
                "supported live evaluation. Raw cases, labels, outcomes, and artifact "
                "references remain private."
            ),
            track_ids=manifest.track_ids,
            modes=tuple(executors),
            evidence_level=manifest.qualification_receipt.evidence_level,
            executors=executors,
            case_count=manifest.case_count,
            revision=manifest.revision,
            tags=(
                "external",
                "pinned",
                "exploratory-e0",
                "normalized-replay",
                (
                    "parser-verified"
                    if import_evidence.parser_verified
                    else "user-provided-import"
                ),
                "native-run-unattested",
                f"research-method:{plugin.status}",
                *(("target-live",) if live_admissions else ()),
                f"adapter:{manifest.adapter_id}",
                f"classification:{manifest.data_classification}",
                f"redistribution:{manifest.redistribution}",
            ),
            methods=_installed_catalog_methods(
                manifest,
                plugin=plugin,
                live_admissions=live_admissions,
            ),
        )


def _installed_catalog_methods(
    manifest: BenchmarkSuiteManifest,
    *,
    plugin: EvaluationMethodPlugin,
    live_admissions: tuple[NormalizedSuiteLiveAdmission, ...],
) -> tuple[CatalogMethod, ...]:
    """Keep imports E0 while projecting independent exact live methods.

    A parser-verified or user-provided normalized import is configured for
    replay once installed. It never inherits native or gate qualification from
    the research inventory. Exact immutable first-party source contracts may
    additionally configure server-owned live methods; only fresh broker
    evidence can later earn a non-E0 level.
    """

    methods = [
        CatalogMethod(
            id=f"{plugin.id}.{track_id}",
            track_id=track_id,
            qualified_gate_ids=(),
            evidence_source=CatalogMethodEvidenceSource.NORMALIZED_IMPORT,
            status="configured",
        )
        for track_id in manifest.track_ids
    ]
    for admission in live_admissions:
        methods.append(
            CatalogMethod(
                id=admission.method_id,
                track_id=admission.track_id,
                qualified_gate_ids=admission.qualified_gate_ids,
                evidence_source=admission.evidence_source,
                status="configured",
            )
        )
    return tuple(methods)
