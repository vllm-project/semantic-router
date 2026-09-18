from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock

import cli.evaluation.orchestrator as evaluation_orchestrator
import pytest
from cli.evaluation.builtin_executors import (
    DEFAULT_EXECUTOR_REGISTRY,
    FixtureReplayExecutor,
)
from cli.evaluation.campaign_protocol import CAMPAIGN_COHORT_SCHEMA_VERSION
from cli.evaluation.catalog import get_catalog
from cli.evaluation.catalog_suites import CatalogSuite
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import Message
from cli.evaluation.contracts import (
    GradingCaseSet,
    RunManifest,
    VisibleCaseSet,
)
from cli.evaluation.errors import StoreError
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_collection import collect_evidence
from cli.evaluation.execution_contract import (
    FIXTURE_REPLAY_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.execution_plan import (
    DEFAULT_SUITE_REGISTRY,
    resolve_execution_plan,
)
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.executor_registry import CollectedEvidence, ExecutorRegistry
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.live_executor import LiveRawResult
from cli.evaluation.live_mom_cases import LIVE_MOM_CASE_COUNT, live_mom_case_sets
from cli.evaluation.metrics import compute_metrics
from cli.evaluation.mom_replay_executor import mom_replay_fixture
from cli.evaluation.orchestrator import run_evaluation, validate_manifest
from cli.evaluation.run_ownership import StandaloneRunOwnership
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.worker_report import WorkerRunState
from evaluation_contract_test_support import (
    _assert_fixture_bundle,
    _assert_fixture_metrics,
    _assert_fixture_run_summary,
    _ExecutorStub,
    _live_manifest,
    _manifest,
    _records,
    _uuid,
)


class _MissingFixtureExecutor:
    contract = FixtureReplayExecutor.contract

    def collect(self, *args: object, **kwargs: object) -> CollectedEvidence:
        collected = FixtureReplayExecutor().collect(*args, **kwargs)
        return replace(collected, fixture_ref=None)


class _UnexpectedFixtureExecutor:
    contract = ExecutorContract(
        id="unexpected-fixture.v1",
        mode="replay",
        suite_class="test-provider",
        target_profile="recorded-source",
        lineage_profile="runtime",
        track_ids=TRACK_IDS,
    )

    def collect(self, *args: object, **kwargs: object) -> CollectedEvidence:
        collected = FixtureReplayExecutor().collect(*args, **kwargs)
        return replace(
            collected,
            inputs=replace(
                collected.inputs,
                suite_executors={"evaluation-smoke": self.contract.id},
                executor_ids=dict.fromkeys(TRACK_IDS, self.contract.id),
            ),
        )


def _mom_replay_manifest(run_id: str = "mom-replay") -> RunManifest:
    live = _live_manifest(run_id)
    return live.with_semantic_updates(
        mode="replay",
        suite_executors={"live-mom-core": MOM_REPLAY_EXECUTOR_ID},
        track_ids=("routing", "model_pool", "joint"),
        sample_limit=LIVE_MOM_CASE_COUNT,
    )


def test_fixture_run_completes_all_tracks_with_rich_bundle(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)

    _assert_fixture_run_summary(report)
    _assert_fixture_metrics(report)
    _assert_fixture_bundle(report, store)
    assert "method_reports" not in store.read_run_json(report.run.id, "report.json")


def test_execution_lease_deduplicates_concurrent_standalone_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest("concurrent-standalone-run")
    root = tmp_path / "store"
    stores = (LocalArtifactStore(root), LocalArtifactStore(root))
    calls = 0
    calls_lock = Lock()

    def count_collection(*args: object, **kwargs: object) -> CollectedEvidence:
        nonlocal calls
        with calls_lock:
            calls += 1
        return collect_evidence(*args, **kwargs)

    monkeypatch.setattr(
        evaluation_orchestrator,
        "collect_evidence",
        count_collection,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = tuple(pool.map(lambda store: run_evaluation(manifest, store), stores))

    assert reports[0] == reports[1]
    assert calls == 1


@pytest.mark.parametrize(
    ("artifact_name", "failure"),
    (
        ("metrics.json", "public checksum receipt"),
        ("failure-summary.json", "contains invalid JSON"),
        ("lineage.json", "lineage.json has an invalid contract"),
        ("checksums.sha256", "public checksum receipt"),
        ("report.json", "immutable staged manifest"),
    ),
)
def test_cas_backed_bundle_tampering_cannot_reconcile_control_state(
    tmp_path: Path,
    artifact_name: str,
    failure: str,
) -> None:
    manifest = _manifest(f"published-bundle-tamper-{artifact_name}")
    store = LocalArtifactStore(tmp_path / "store")
    run_evaluation(manifest, store)
    run_dir = store.runs / manifest.run_id
    status_before = (run_dir / "status.json").read_bytes()
    events_before = (run_dir / "events.jsonl").read_bytes()
    artifact = run_dir / artifact_name
    if artifact_name == "metrics.json":
        tampered = artifact.read_bytes() + b" "
        media_type = "application/json"
    elif artifact_name == "failure-summary.json":
        tampered = artifact.read_bytes().replace(
            b'"total_records":',
            b'"total_records":0,"total_records":',
            1,
        )
        media_type = "application/json"
    elif artifact_name == "lineage.json":
        payload = json.loads(artifact.read_bytes())
        tampered = json.dumps(payload["resolved_snapshot"]).encode()
        media_type = "application/json"
    elif artifact_name == "checksums.sha256":
        tampered = b"\n".join(reversed(artifact.read_bytes().splitlines())) + b"\n"
        media_type = "text/plain"
    else:
        payload = json.loads(artifact.read_bytes())
        payload["run"]["name"] += " tampered"
        tampered = json.dumps(payload).encode()
        media_type = "application/json"
    artifact.write_bytes(tampered)
    artifact.chmod(0o600)
    store.put_bytes(tampered, media_type)

    with pytest.raises(StoreError, match=failure):
        run_evaluation(manifest, store)

    assert (run_dir / "status.json").read_bytes() == status_before
    assert (run_dir / "events.jsonl").read_bytes() == events_before


def test_incomplete_published_bundle_cannot_reconcile_control_state(
    tmp_path: Path,
) -> None:
    manifest = _manifest("published-bundle-missing-artifact")
    store = LocalArtifactStore(tmp_path / "store")
    run_evaluation(manifest, store)
    run_dir = store.runs / manifest.run_id
    status_before = (run_dir / "status.json").read_bytes()
    events_before = (run_dir / "events.jsonl").read_bytes()
    (run_dir / "lineage.json").unlink()

    with pytest.raises(StoreError, match="bundle is incomplete"):
        run_evaluation(manifest, store)

    assert (run_dir / "status.json").read_bytes() == status_before
    assert (run_dir / "events.jsonl").read_bytes() == events_before


def test_existing_report_repairs_standalone_terminal_control_state_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest("standalone-report-recovery")
    store = LocalArtifactStore(tmp_path / "store")
    record_state = evaluation_orchestrator.StandaloneRunOwnership.record_state

    def interrupt_terminal_status(
        ownership: StandaloneRunOwnership,
        value: WorkerRunState,
    ) -> None:
        if value.status == "completed":
            raise RuntimeError("simulated stop after immutable report publication")
        record_state(ownership, value)

    monkeypatch.setattr(
        evaluation_orchestrator.StandaloneRunOwnership,
        "record_state",
        interrupt_terminal_status,
    )
    with pytest.raises(RuntimeError, match="immutable report publication"):
        run_evaluation(manifest, store)
    assert (store.runs / manifest.run_id / "report.json").is_file()
    assert store.read_run_json(manifest.run_id, "status.json")["status"] == "failed"

    monkeypatch.setattr(
        evaluation_orchestrator.StandaloneRunOwnership,
        "record_state",
        record_state,
    )
    recovered = run_evaluation(manifest, store)
    rerun = run_evaluation(manifest, store)

    assert recovered == rerun
    assert recovered.run.status == "completed"
    assert store.read_run_json(manifest.run_id, "status.json")["status"] == "completed"
    events = [
        json.loads(line)
        for line in store.read_run_text(manifest.run_id, "events.jsonl").splitlines()
    ]
    assert events[-1]["type"] == "completed"
    assert sum(event["type"] == "completed" for event in events) == 1


@pytest.mark.parametrize("interrupted_artifact", ("metrics.json", "report.json"))
def test_sealed_report_transaction_recovers_without_reexecuting_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_artifact: str,
) -> None:
    manifest = _manifest(f"report-transaction-{interrupted_artifact}")
    store = LocalArtifactStore(tmp_path / "store")
    replace = store._filesystem.replace_private_file
    interrupted = False

    def interrupt_publication(
        source: Path,
        target: Path,
        *,
        expected_data: bytes,
    ) -> bool:
        nonlocal interrupted
        if (
            not interrupted
            and source.parent.name == ".report-transaction"
            and target.name == interrupted_artifact
        ):
            interrupted = True
            raise OSError("simulated report publication interruption")
        return replace(source, target, expected_data=expected_data)

    monkeypatch.setattr(
        store._filesystem,
        "replace_private_file",
        interrupt_publication,
    )
    with pytest.raises(OSError, match="report publication interruption"):
        run_evaluation(manifest, store)

    run_dir = store.runs / manifest.run_id
    assert interrupted
    assert (run_dir / ".report-transaction" / "transaction.json").is_file()
    assert not (run_dir / "report.json").exists()

    monkeypatch.setattr(store._filesystem, "replace_private_file", replace)

    def reject_reexecution(*args: object, **kwargs: object) -> object:
        raise AssertionError("sealed report recovery must not recollect evidence")

    monkeypatch.setattr(evaluation_orchestrator, "collect_evidence", reject_reexecution)
    recovered = run_evaluation(manifest, store)

    assert recovered.run.status == "completed"
    assert (run_dir / "report.json").is_file()
    assert not (run_dir / ".report-transaction").exists()
    assert not (run_dir / ".report-preparing").exists()


def test_sealed_report_transaction_is_bound_to_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest("report-transaction-manifest-binding")
    store = LocalArtifactStore(tmp_path / "store")
    replace = store._filesystem.replace_private_file

    def interrupt_report(
        source: Path,
        target: Path,
        *,
        expected_data: bytes,
    ) -> bool:
        if source.parent.name == ".report-transaction" and target.name == "report.json":
            raise OSError("simulated report publication interruption")
        return replace(source, target, expected_data=expected_data)

    monkeypatch.setattr(
        store._filesystem,
        "replace_private_file",
        interrupt_report,
    )
    with pytest.raises(OSError, match="report publication interruption"):
        run_evaluation(manifest, store)
    monkeypatch.setattr(store._filesystem, "replace_private_file", replace)

    changed = manifest.with_semantic_updates(code_revision="sha256:" + "2" * 64)
    with pytest.raises(StoreError, match="another run manifest"):
        run_evaluation(changed, store)


def test_sealed_report_preparation_recovers_before_directory_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest("report-transaction-pre-promotion")
    store = LocalArtifactStore(tmp_path / "store")
    rename = store._filesystem.rename_private_directory
    interrupted = False

    def interrupt_promotion(source: Path, target: Path) -> None:
        nonlocal interrupted
        if (
            not interrupted
            and source.name == ".report-preparing"
            and target.name == ".report-transaction"
        ):
            interrupted = True
            raise OSError("simulated report promotion interruption")
        rename(source, target)

    monkeypatch.setattr(
        store._filesystem,
        "rename_private_directory",
        interrupt_promotion,
    )
    with pytest.raises(OSError, match="report promotion interruption"):
        run_evaluation(manifest, store)

    run_dir = store.runs / manifest.run_id
    assert interrupted
    assert (run_dir / ".report-preparing" / "transaction.json").is_file()
    assert not (run_dir / ".report-transaction").exists()

    monkeypatch.setattr(store._filesystem, "rename_private_directory", rename)

    def reject_reexecution(*args: object, **kwargs: object) -> object:
        raise AssertionError("sealed preparation recovery must not recollect evidence")

    monkeypatch.setattr(evaluation_orchestrator, "collect_evidence", reject_reexecution)
    recovered = run_evaluation(manifest, store)

    assert recovered.run.status == "completed"
    assert (run_dir / "report.json").is_file()
    assert not (run_dir / ".report-preparing").exists()
    assert not (run_dir / ".report-transaction").exists()


def test_rich_pool_and_preference_reducers_preserve_decision_information() -> None:
    pool_records = [
        ExecutionRecord(
            id=f"pool-{case_id}-{arm_id}",
            track_id="model_pool",
            case_id=case_id,
            attempt_id=f"attempt-{case_id}-{arm_id}",
            status="succeeded",
            arm_id=arm_id,
            success=True,
            quality=quality,
            runtime_cost=runtime_cost,
        )
        for case_id, arm_id, quality, runtime_cost in (
            ("case-1", "arm-a", 0.5, 0.3),
            ("case-1", "arm-b", 0.6, 0.2),
            ("case-2", "arm-a", 0.4, 0.3),
            ("case-2", "arm-b", 0.4, 0.2),
        )
    ]
    preference_records = [
        ExecutionRecord(
            id=f"preference-{index}",
            track_id="preference",
            case_id=f"preference-case-{index}",
            attempt_id=f"preference-attempt-{index}",
            status="succeeded",
            success=True,
            preference_match=matched,
            behavior_propensity=propensity,
        )
        for index, (matched, propensity) in enumerate(
            ((True, 0.1), (False, 0.9)), start=1
        )
    ]

    metrics = {
        metric.id: metric.value
        for metric in compute_metrics(
            pool_records + preference_records,
            capacity_profile=None,
        )
    }

    assert metrics["model_pool.quality_dominated_arm_count"] == 1
    assert metrics["model_pool.pareto_evaluable_arm_count"] == 2
    assert metrics["model_pool.pareto_dominated_arm_count"] == 1
    assert metrics["model_pool.mean_pairwise_failure_jaccard"] == 0
    assert metrics["model_pool.all_arm_failure_rate"] == 0
    assert metrics["preference.effective_sample_size"] == pytest.approx(
        1.2195121951219512
    )
    assert metrics["preference.effective_sample_ratio"] == pytest.approx(
        0.6097560975609756
    )
    assert metrics["preference.self_normalized_ips_agreement"] == pytest.approx(0.9)


def test_executor_registry_is_explicit_and_rejects_ambiguous_ids() -> None:
    registry = ExecutorRegistry((_ExecutorStub("executor-a"),))
    assert registry.ids == ("executor-a",)
    assert registry.require("executor-a").contract.id == "executor-a"
    with pytest.raises(ValueError, match="unknown evaluation executor"):
        registry.require("missing")
    with pytest.raises(ValueError, match="duplicate evaluation executor"):
        ExecutorRegistry((_ExecutorStub("executor-a"), _ExecutorStub("executor-a")))

    validate_manifest(
        _manifest(),
        executor_registry=ExecutorRegistry(
            (_ExecutorStub(FIXTURE_REPLAY_EXECUTOR_ID),)
        ),
    )


def test_execution_entry_rejects_a_model_copy_that_bypassed_digest_validation() -> None:
    tampered = _manifest().model_copy(update={"name": "Tampered after validation"})

    with pytest.raises(ValueError, match="manifest_digest does not match"):
        validate_manifest(tampered)


def test_execution_identity_maps_are_immutable_after_resolution() -> None:
    manifest = _manifest()
    plan = resolve_execution_plan(
        manifest, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
    )
    inputs = fixture_inputs()

    with pytest.raises(TypeError):
        plan.suite_revisions["evaluation-smoke"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.suite_executors["evaluation-smoke"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        inputs.suite_executors["evaluation-smoke"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        inputs.executor_ids["routing"] = "changed"  # type: ignore[index]

    reversed_grading = GradingCaseSet(cases=tuple(reversed(inputs.grading.cases)))
    with pytest.raises(ValueError, match="identical ordering"):
        replace(inputs, grading=reversed_grading)


def test_catalog_suite_mode_executor_contract_is_exact_and_immutable() -> None:
    fields = {
        "id": "installed-routing",
        "name": "Installed routing",
        "description": "One workload with distinct replay and live strategies.",
        "track_ids": ("routing",),
        "modes": ("replay", "live"),
        "evidence_level": "E0",
        "executors": {
            "replay": "normalized-suite-replay.v1",
            "live": "normalized-suite-live.v1",
        },
        "revision": "sha256:" + "a" * 64,
        "methods": (
            {
                "id": "installed.routing.v1",
                "track_id": "routing",
                "qualified_gate_ids": (),
                "evidence_source": "normalized_import",
                "status": "configured",
            },
        ),
    }
    suite = CatalogSuite.model_validate(fields)
    with pytest.raises(TypeError):
        suite.executors["replay"] = "changed"  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly cover"):
        CatalogSuite.model_validate(
            {**fields, "executors": {"replay": "normalized-suite-replay.v1"}}
        )
    with pytest.raises(ValueError, match="canonical replay/live order"):
        CatalogSuite.model_validate({**fields, "modes": ("live", "replay")})
    with pytest.raises(ValueError):
        CatalogSuite.model_validate(
            {
                key: value
                for key, value in {
                    **fields,
                    "executor_id": "normalized-suite-replay.v1",
                }.items()
                if key != "executors"
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("modes", ("live",)),
        ("executors", {"replay": MOM_REPLAY_EXECUTOR_ID, "live": "wrong.v1"}),
        ("evidence_level", "E1"),
        ("case_count", 0),
        (
            "campaign_protocol",
            {
                "schema_version": "evaluation-campaign-cohort.v0",
                "minimum_cases": 59,
            },
        ),
        ("campaign_protocol", {"minimum_cases": 0}),
        (
            "campaign_protocol",
            {"minimum_cases": LIVE_MOM_CASE_COUNT + 1},
        ),
        ("track_ids", ("routing", "model_pool")),
    ),
)
def test_campaign_protocol_requires_a_consistent_mom_cohort_contract(
    field: str, value: object
) -> None:
    suite = next(
        suite
        for suite in get_catalog(generated_at=False).suites
        if suite.id == "live-mom-core"
    )
    payload = suite.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValueError):
        CatalogSuite.model_validate(payload)


def test_campaign_protocol_omission_and_clean_break() -> None:
    suites = get_catalog(generated_at=False).suites
    campaign = next(suite for suite in suites if suite.id == "live-mom-core")
    campaign_payload = campaign.model_dump(mode="json", exclude_none=False)
    assert campaign_payload["campaign_protocol"] == {
        "schema_version": CAMPAIGN_COHORT_SCHEMA_VERSION,
        "minimum_cases": 59,
    }
    smoke = next(suite for suite in suites if suite.id == "evaluation-smoke")
    assert "campaign_protocol" not in smoke.model_dump(mode="json", exclude_none=False)

    for legacy_field, value in (
        ("campaign_eligible", True),
        ("campaign_minimum_cases", 59),
    ):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            CatalogSuite.model_validate({**campaign_payload, legacy_field: value})


def test_executor_output_is_validated_against_the_resolved_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    plan = resolve_execution_plan(
        manifest, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
    )
    store = LocalArtifactStore(tmp_path / "store")

    drifted_plan = replace(
        plan, suite_revisions={"evaluation-smoke": "different-revision"}
    )
    with pytest.raises(ValueError, match="different suite revisions"):
        collect_evidence(manifest, store, drifted_plan)

    drifted_manifest = manifest.with_semantic_updates(
        suite_executors={"evaluation-smoke": "normalized-suite-replay.v1"}
    )
    with pytest.raises(ValueError, match="suite executors"):
        validate_manifest(drifted_manifest)

    monkeypatch.setattr(
        "cli.evaluation.builtin_executors.execute_fixture", lambda *args: []
    )
    with pytest.raises(ValueError, match="omitted a planned case-track cell"):
        collect_evidence(manifest, store, plan)


def test_executor_fixture_declaration_is_enforced_centrally(tmp_path: Path) -> None:
    manifest = _manifest()
    plan = resolve_execution_plan(
        manifest, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
    )
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError, match="omitted its required fixture reference"):
        collect_evidence(
            manifest,
            store,
            plan,
            registry=ExecutorRegistry((_MissingFixtureExecutor(),)),
        )

    unexpected = _UnexpectedFixtureExecutor()
    unexpected_plan = replace(
        plan,
        suite_executors={"evaluation-smoke": unexpected.contract.id},
    )
    with pytest.raises(ValueError, match="undeclared fixture reference"):
        collect_evidence(
            manifest,
            store,
            unexpected_plan,
            registry=ExecutorRegistry((unexpected,)),
        )


def test_mom_replay_executes_the_same_dense_campaign_cohort(tmp_path: Path) -> None:
    manifest = _mom_replay_manifest()
    store = LocalArtifactStore(tmp_path / "store")

    report = run_evaluation(manifest, store)
    records = _records(store, report.run.id)
    by_track = {
        track_id: [row for row in records if row.track_id == track_id]
        for track_id in ("routing", "model_pool", "joint")
    }

    assert report.run.status == "completed"
    assert report.run.evidence_level == "E0"
    assert {track.evidence_level for track in report.tracks} == {"E0"}
    assert len(by_track["routing"]) == LIVE_MOM_CASE_COUNT
    assert len(by_track["model_pool"]) == LIVE_MOM_CASE_COUNT * 2
    assert len(by_track["joint"]) == LIVE_MOM_CASE_COUNT
    assert len({row.case_id for row in records}) == LIVE_MOM_CASE_COUNT
    assert {row.grader for row in by_track["routing"] if row.quality is not None} == {
        "dense-pool-oracle.v1"
    }
    assert all(
        {row.arm_id for row in by_track["model_pool"] if row.case_id == case_id}
        == {arm.id for arm in manifest.target.mixture.model_arms}
        for case_id in {row.case_id for row in by_track["routing"]}
    )
    assert run_evaluation(manifest, store) == report


def test_mom_replay_and_live_freeze_identical_workload_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = _mom_replay_manifest("mom-workload-replay")
    live = replay.with_semantic_updates(
        run_id=_uuid("mom-workload-live"),
        name="Live workload bridge",
        mode="live",
        suite_executors={"live-mom-core": "live-runtime.v1"},
    )
    empty_raw = LiveRawResult(
        records=[],
        discovered_entrypoints=(live.target.mixture.entrypoint_model,),
        routing_traces=(),
        chat_results={},
        model_pool_results={},
        model_pool_arm_ids=tuple(arm.id for arm in live.target.mixture.model_arms),
        joint_results={},
    )
    monkeypatch.setattr(
        "cli.evaluation.builtin_executors.execute_live_raw",
        lambda *args, **kwargs: empty_raw,
    )
    store = LocalArtifactStore(tmp_path / "store")

    replay_evidence = DEFAULT_EXECUTOR_REGISTRY.require(MOM_REPLAY_EXECUTOR_ID).collect(
        replay,
        store,
        resolve_execution_plan(
            replay, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
        ),
        None,
    )
    live_evidence = DEFAULT_EXECUTOR_REGISTRY.require("live-runtime.v1").collect(
        live,
        store,
        resolve_execution_plan(
            live, None, DEFAULT_SUITE_REGISTRY, DEFAULT_EXECUTOR_REGISTRY
        ),
        None,
    )

    assert replay.suite_ids == live.suite_ids == ("live-mom-core",)
    assert replay.suite_revisions == live.suite_revisions
    assert replay_evidence.visible_ref.digest == live_evidence.visible_ref.digest
    assert replay_evidence.grading_ref.digest == live_evidence.grading_ref.digest


def test_mom_replay_randomness_binds_the_full_case_snapshot() -> None:
    manifest = _mom_replay_manifest("mom-case-snapshot")
    visible, grading = live_mom_case_sets()
    original = mom_replay_fixture(manifest, visible, grading)
    first = visible.cases[0]
    changed_visible = VisibleCaseSet(
        cases=(
            first.model_copy(
                update={
                    "messages": (
                        Message(
                            role="user",
                            content="Return exactly 43; this is a changed frozen case.",
                        ),
                    )
                }
            ),
            *visible.cases[1:],
        )
    )
    changed = mom_replay_fixture(manifest, changed_visible, grading)

    assert original.cases[0].model_dump(mode="json") != changed.cases[0].model_dump(
        mode="json"
    )
