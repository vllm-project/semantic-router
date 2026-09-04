"""Resolve workload inputs and dispatch the selected evidence executor."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.contracts import ArtifactRef, RunManifest
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.fixture_executor import execute_fixture
from cli.evaluation.fixtures import FixtureInputs, fixture_inputs
from cli.evaluation.live_executor import execute_live_raw, grade_live_execution
from cli.evaluation.normalized_suite_executor import (
    NormalizedSuiteInputs,
    execute_normalized_suites,
)
from cli.evaluation.resolution import live_grading, sample_fixture
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import BenchmarkSuiteManifest
from cli.evaluation.suite_store import NormalizedSuiteStore


@dataclass(frozen=True)
class CollectedEvidence:
    inputs: FixtureInputs | NormalizedSuiteInputs
    visible_ref: ArtifactRef
    grading_ref: ArtifactRef
    fixture_ref: ArtifactRef | None
    records: list[ExecutionRecord]
    discovered_entrypoints: tuple[str, ...]
    routing_traces: tuple[RoutingDiagnostic, ...]
    benchmark_revisions: dict[str, str]
    private_lineage: dict[str, object] | None


def collect_evidence(
    manifest: RunManifest,
    store: LocalArtifactStore,
    *,
    suite_store: NormalizedSuiteStore | None = None,
    external_suites: tuple[BenchmarkSuiteManifest, ...] = (),
) -> CollectedEvidence:
    if external_suites:
        if suite_store is None:
            raise ValueError("external suite execution requires a trusted suite store")
        replay = execute_normalized_suites(
            store=suite_store,
            manifests=external_suites,
            track_ids=manifest.track_ids,
            sample_limit=manifest.sample_limit,
            seed=manifest.seed,
        )
        inputs = replay.inputs
        if inputs.suite_revisions != manifest.suite_revisions:
            raise ValueError(
                "executed external suite revisions differ from the frozen manifest"
            )
        return CollectedEvidence(
            inputs=inputs,
            visible_ref=store.put_json(inputs.visible),
            grading_ref=store.put_json(inputs.grading),
            fixture_ref=None,
            records=replay.records,
            discovered_entrypoints=(),
            routing_traces=(),
            benchmark_revisions=inputs.suite_revisions,
            private_lineage=inputs.private_lineage,
        )

    inputs = sample_fixture(fixture_inputs(), manifest.sample_limit, manifest.seed)
    grading = (
        inputs.grading if manifest.mode == "replay" else live_grading(inputs.grading)
    )
    visible_ref = store.put_json(inputs.visible)
    grading_ref = store.put_json(grading)
    fixture_ref = store.put_json(inputs.fixture) if manifest.mode == "replay" else None
    if manifest.mode == "replay":
        records = execute_fixture(
            inputs.visible, grading, inputs.fixture, manifest.track_ids
        )
        records = [
            row.model_copy(
                update={
                    "evaluation_cost": 0.00005,
                    "evidence_kind": "synthetic-contract-fixture",
                }
            )
            for row in records
        ]
        entrypoints: tuple[str, ...] = ()
        routing_traces: tuple[RoutingDiagnostic, ...] = ()
    else:
        raw = execute_live_raw(
            inputs.visible,
            track_ids=manifest.track_ids,
            router_api_url=manifest.target.router_api_url,
            envoy_url=manifest.target.envoy_url,
            concurrency=manifest.concurrency,
            model_arms=manifest.target.model_arms,
            router_api_key_env=(
                manifest.target.router_api_key.env
                if manifest.target.router_api_key
                else None
            ),
            envoy_api_key_env=(
                manifest.target.envoy_api_key.env
                if manifest.target.envoy_api_key
                else None
            ),
        )
        result = grade_live_execution(
            raw,
            inputs.visible,
            grading,
            track_ids=manifest.track_ids,
            model_arms=manifest.target.model_arms,
        )
        records = result.records
        entrypoints = result.discovered_entrypoints
        routing_traces = result.routing_traces
    return CollectedEvidence(
        inputs=inputs,
        visible_ref=visible_ref,
        grading_ref=grading_ref,
        fixture_ref=fixture_ref,
        records=records,
        discovered_entrypoints=entrypoints,
        routing_traces=routing_traces,
        benchmark_revisions=dict(manifest.suite_revisions),
        private_lineage=None,
    )
