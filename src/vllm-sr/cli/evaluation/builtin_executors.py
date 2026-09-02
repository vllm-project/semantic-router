"""Built-in evidence executor implementations for the current catalog."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from cli.evaluation.case_plan import project_visible_case_set
from cli.evaluation.contracts import RunManifest
from cli.evaluation.dense_pool_oracle import grade_routing_with_dense_pool_oracle
from cli.evaluation.execution_contract import (
    FIXTURE_REPLAY_EXECUTOR_ID,
    LIVE_RUNTIME_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
    NORMALIZED_LIVE_EXECUTOR_ID,
    NORMALIZED_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.execution_plan import ExecutionPlan
from cli.evaluation.executor_contracts import BUILTIN_EXECUTOR_CONTRACTS
from cli.evaluation.executor_registry import CollectedEvidence, ExecutorRegistry
from cli.evaluation.fixture_executor import execute_fixture
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.live_executor import execute_live_raw
from cli.evaluation.live_mom_cases import live_mom_case_sets
from cli.evaluation.live_runtime_collection import collect_live_runtime_evidence
from cli.evaluation.mom_replay_executor import mom_replay_fixture
from cli.evaluation.normalized_suite_executor import execute_normalized_suites
from cli.evaluation.normalized_suite_live_executor import (
    execute_normalized_suite_live,
)
from cli.evaluation.resolution import sample_case_sets, sample_fixture
from cli.evaluation.runtime_factors import runtime_factors
from cli.evaluation.store import ArtifactStore
from cli.evaluation.suite_store import NormalizedSuiteStore

_CONTRACTS = MappingProxyType(
    {contract.id: contract for contract in BUILTIN_EXECUTOR_CONTRACTS}
)


class FixtureReplayExecutor:
    contract = _CONTRACTS[FIXTURE_REPLAY_EXECUTOR_ID]

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence:
        del plan, suite_store
        inputs = sample_fixture(fixture_inputs(), manifest.sample_limit, manifest.seed)
        inputs = replace(
            inputs,
            visible=project_visible_case_set(inputs.visible, manifest.track_ids),
        )
        if inputs.fixture is None:
            raise ValueError("fixture executor received no replay evidence")
        records = execute_fixture(
            inputs.visible, inputs.grading, inputs.fixture, manifest.track_ids
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
        return CollectedEvidence(
            inputs=inputs,
            visible_ref=store.put_json(inputs.visible),
            grading_ref=store.put_json(inputs.grading),
            fixture_ref=store.put_json(inputs.fixture),
            records=records,
            discovered_entrypoints=(),
            routing_traces=(),
        )


class MoMReplayExecutor:
    contract = _CONTRACTS[MOM_REPLAY_EXECUTOR_ID]

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence:
        del plan, suite_store
        visible, grading = live_mom_case_sets()
        visible, grading = sample_case_sets(
            visible, grading, manifest.sample_limit, manifest.seed
        )
        visible = project_visible_case_set(visible, manifest.track_ids)
        mixture = manifest.target.mixture
        if mixture is None:
            raise ValueError("MoM replay executor requires a frozen target mixture")
        fixture = mom_replay_fixture(manifest, visible, grading)
        factors = runtime_factors(manifest)
        inputs = replace(
            fixture_inputs(),
            visible=visible,
            grading=grading,
            fixture=fixture,
            policy=factors.policy,
            pool=factors.pool,
            arms=factors.arms,
            binding=factors.binding,
            environment=factors.environment,
            suite_revisions=dict(manifest.suite_revisions),
            suite_executors=dict(manifest.suite_executors),
            executor_ids=dict.fromkeys(manifest.track_ids, self.contract.id),
        )
        fixture_records = execute_fixture(
            inputs.visible, inputs.grading, fixture, manifest.track_ids
        )
        fixture_records = grade_routing_with_dense_pool_oracle(
            fixture_records,
            tuple(arm.id for arm in mixture.model_arms),
        )
        records = [
            row.model_copy(
                update={
                    "evaluation_cost": 0.00005,
                    "evidence_kind": "mom-frozen-counterfactual-v1",
                }
            )
            for row in fixture_records
        ]
        return CollectedEvidence(
            inputs=inputs,
            visible_ref=store.put_json(inputs.visible),
            grading_ref=store.put_json(inputs.grading),
            fixture_ref=store.put_json(fixture),
            records=records,
            discovered_entrypoints=(),
            routing_traces=(),
        )


class LiveRuntimeExecutor:
    contract = _CONTRACTS[LIVE_RUNTIME_EXECUTOR_ID]

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence:
        del plan, suite_store
        return collect_live_runtime_evidence(
            manifest,
            store,
            executor_id=self.contract.id,
            execute_raw=execute_live_raw,
        )


class NormalizedReplayExecutor:
    contract = _CONTRACTS[NORMALIZED_REPLAY_EXECUTOR_ID]

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence:
        if suite_store is None:
            raise ValueError(
                "normalized replay executor requires a trusted suite store"
            )
        replay = execute_normalized_suites(
            store=suite_store,
            manifests=plan.suites,
            track_ids=manifest.track_ids,
            sample_limit=manifest.sample_limit,
            seed=manifest.seed,
            executor_id=self.contract.id,
            target_id=manifest.target.id,
        )
        if replay.inputs.suite_revisions != manifest.suite_revisions:
            raise ValueError("executed suite revisions differ from the frozen manifest")
        return CollectedEvidence(
            inputs=replay.inputs,
            visible_ref=store.put_json(replay.inputs.visible),
            grading_ref=store.put_json(replay.inputs.grading),
            fixture_ref=None,
            records=replay.records,
            discovered_entrypoints=(),
            routing_traces=(),
        )


class NormalizedLiveExecutor:
    contract = _CONTRACTS[NORMALIZED_LIVE_EXECUTOR_ID]

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence:
        if suite_store is None:
            raise ValueError("normalized live executor requires a trusted suite store")
        execution = execute_normalized_suite_live(
            manifest=manifest,
            store=suite_store,
            manifests=plan.suites,
            executor_id=self.contract.id,
        )
        if execution.inputs.suite_revisions != manifest.suite_revisions:
            raise ValueError("executed suite revisions differ from the frozen manifest")
        return CollectedEvidence(
            inputs=execution.inputs,
            visible_ref=store.put_json(execution.inputs.visible),
            grading_ref=store.put_json(execution.inputs.grading),
            fixture_ref=None,
            records=execution.records,
            discovered_entrypoints=execution.discovered_entrypoints,
            routing_traces=execution.routing_traces,
        )


DEFAULT_EXECUTOR_REGISTRY = ExecutorRegistry(
    (
        FixtureReplayExecutor(),
        MoMReplayExecutor(),
        LiveRuntimeExecutor(),
        NormalizedReplayExecutor(),
        NormalizedLiveExecutor(),
    )
)
