from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.canonical import (
    digest_value,
    pretty_json_bytes,
    sha256_digest,
    strict_json_loads,
)
from cli.evaluation.catalog import get_catalog
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import RunManifest, WorkloadSnapshot
from cli.evaluation.execution_plan import SuiteRegistry, resolve_execution_plan
from cli.evaluation.executor_contracts import BUILTIN_EXECUTOR_CONTRACTS
from cli.evaluation.manifest_identity import seal_manifest_fields
from cli.evaluation.orchestrator import run_evaluation, run_worker_evaluation
from cli.evaluation.store import LocalArtifactStore, WorkerArtifactStore
from cli.evaluation.target_capabilities import DEFAULT_TARGET_REGISTRY
from cli.evaluation.target_contracts import EvaluationTarget
from cli.evaluation.worker_report import (
    WorkerReportDraft,
    WorkerRunProgress,
    WorkerRunState,
    worker_run_state_from_manifest,
)
from pydantic import ValidationError


def _golden(name: str) -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "evaluation" / name
    return strict_json_loads(path.read_bytes())


def _stage_server_manifest(store: WorkerArtifactStore, manifest: RunManifest) -> None:
    """Model the Dashboard-owned staging boundary used by the worker process."""

    run_dir = store.runs / manifest.run_id
    run_dir.mkdir(mode=0o700)
    target = run_dir / "run-manifest.json"
    target.write_bytes(pretty_json_bytes(manifest))
    target.chmod(0o600)


def test_runtime_catalog_tracks_are_capability_dependent() -> None:
    matrix = _golden("capability-matrix.json")
    assert matrix["schema_version"] == SCHEMA_VERSION
    for case in matrix["cases"]:
        if not case["valid"]:
            with pytest.raises(ValidationError):
                EvaluationTarget.model_validate(case["target"])
            continue
        target = EvaluationTarget.model_validate(case["target"])
        catalog = get_catalog(
            generated_at=False,
            router_api_url=target.router_api_url,
            envoy_url=target.envoy_url,
            agent_task_ledger=target.agent_task_ledger,
            fault_recovery_ledger=target.fault_recovery_ledger,
            hard_policy_ledger=target.hard_policy_ledger,
            production_experiment_ledger=target.production_experiment_ledger,
            mixture=target.mixture,
            backend_topology_digest=target.backend_topology_digest,
        )
        if target.mixture is None:
            assert tuple(item.id for item in catalog.targets) == (
                "fixture",
                "benchmark-source",
            )
            assert case["expected_tracks"] == []
            continue
        mixture_target = next(
            item for item in catalog.targets if item.id == target.mixture.id
        )
        assert mixture_target.track_ids == tuple(case["expected_tracks"]), case["name"]
        assert mixture_target.healthy is bool(case["expected_tracks"]), case["name"]
        assert mixture_target.mixture == target.mixture.public_summary()


def test_live_manifest_requires_the_current_runtime_endpoint_contract() -> None:
    payload = _golden("live-manifest.json")
    frozen_mixture = dict(dict(payload["target"])["mixture"])
    mixture_id = frozen_mixture["id"]
    payload = seal_manifest_fields(
        {
            **{
                key: value for key, value in payload.items() if key != "manifest_digest"
            },
            "mode": "live",
            "target": {
                "schema_version": SCHEMA_VERSION,
                "id": mixture_id,
                "kind": "mixture-of-models",
                "router_api_url": "http://router:8080",
                "envoy_url": "http://envoy:8801",
                "backend_topology_digest": sha256_digest(b"backend-topology"),
                "mixture": frozen_mixture,
            },
        }
    )
    parsed = RunManifest.model_validate(payload)
    live_executor = next(
        executor
        for executor in BUILTIN_EXECUTOR_CONTRACTS
        if executor.id == "live-runtime.v1"
    )
    DEFAULT_TARGET_REGISTRY.resolve(parsed, live_executor)
    scoped = parsed.with_semantic_updates(
        target=parsed.target.model_copy(update={"id": f"candidate--{mixture_id}"})
    )
    DEFAULT_TARGET_REGISTRY.resolve(scoped, live_executor)
    assert parsed.target.envoy_url == "http://envoy:8801"
    assert parsed.target.mixture is not None
    assert parsed.target.mixture.model_arms[0].id == "fast"
    missing_topology = dict(payload)
    missing_topology["target"] = {
        key: value
        for key, value in dict(payload["target"]).items()
        if key != "backend_topology_digest"
    }
    missing_topology_manifest = RunManifest.model_validate(
        seal_manifest_fields(
            {
                key: value
                for key, value in missing_topology.items()
                if key != "manifest_digest"
            }
        )
    )
    with pytest.raises(ValueError, match="brokered-runtime target is incomplete"):
        DEFAULT_TARGET_REGISTRY.resolve(missing_topology_manifest, live_executor)
    payload["target"] = {
        "schema_version": SCHEMA_VERSION,
        "id": mixture_id,
        "kind": "mixture-of-models",
        "mixture": frozen_mixture,
    }
    missing_endpoints = RunManifest.model_validate(
        seal_manifest_fields(
            {key: value for key, value in payload.items() if key != "manifest_digest"}
        )
    )
    with pytest.raises(ValueError, match="brokered-runtime target is incomplete"):
        DEFAULT_TARGET_REGISTRY.resolve(missing_endpoints, live_executor)


def test_mom_replay_admission_uses_frozen_executor_contract_not_suite_id() -> None:
    live = RunManifest.model_validate(_golden("live-manifest.json"))
    campaign_suite = next(
        suite
        for suite in get_catalog(generated_at=False).suites
        if suite.campaign_protocol
    )
    renamed_suite = campaign_suite.model_copy(
        update={"id": "new-mom-cohort", "revision": "new-mom-cohort-v1"}
    )
    mom_executor = next(
        contract
        for contract in BUILTIN_EXECUTOR_CONTRACTS
        if contract.suite_class == "mom-cohort"
    )
    replay = live.with_semantic_updates(
        mode="replay",
        suite_ids=("new-mom-cohort",),
        suite_revisions={"new-mom-cohort": "new-mom-cohort-v1"},
        suite_executors={"new-mom-cohort": mom_executor.id},
        track_ids=("routing", "model_pool", "joint"),
        concurrency=1,
        capacity_slo=None,
        capacity_load_protocol=None,
    )

    plan = resolve_execution_plan(
        replay,
        None,
        SuiteRegistry((renamed_suite,)),
        DEFAULT_EXECUTOR_REGISTRY,
    )
    state = worker_run_state_from_manifest(
        replay,
        status="pending",
        progress=WorkerRunProgress(
            percent=0,
            completed=0,
            total=len(replay.track_ids),
            message="Queued",
        ),
    )

    assert plan.executor_id == mom_executor.id
    assert state.suite_ids == ("new-mom-cohort",)
    assert state.mixture is not None


def test_replay_mixture_rejects_non_mom_executor_contract() -> None:
    live = RunManifest.model_validate(_golden("live-manifest.json"))
    fixture_suite = next(
        suite
        for suite in get_catalog(generated_at=False).suites
        if suite.id == "evaluation-smoke"
    ).model_copy(update={"id": "ordinary-replay", "revision": "ordinary-replay-v1"})
    fixture_executor = next(
        contract
        for contract in BUILTIN_EXECUTOR_CONTRACTS
        if contract.suite_class == "fixture"
    )
    forged = live.with_semantic_updates(
        mode="replay",
        suite_ids=("ordinary-replay",),
        suite_revisions={"ordinary-replay": "ordinary-replay-v1"},
        suite_executors={"ordinary-replay": fixture_executor.id},
        track_ids=("routing",),
        concurrency=1,
        capacity_slo=None,
        capacity_load_protocol=None,
    )

    with pytest.raises(ValueError, match="does not accept its frozen executor"):
        resolve_execution_plan(
            forged,
            None,
            SuiteRegistry((fixture_suite,)),
            DEFAULT_EXECUTOR_REGISTRY,
        )


def test_visible_and_grading_case_artifacts_must_be_physically_separate() -> None:
    ref = ArtifactRef(
        digest="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=10,
    )
    with pytest.raises(ValidationError, match="separate artifacts"):
        WorkloadSnapshot(id="hidden-label-check", visible_cases=ref, grading_cases=ref)


def test_canonical_digest_is_key_order_independent() -> None:
    assert digest_value({"b": 2, "a": [3, 1]}) == digest_value({"a": [3, 1], "b": 2})


@pytest.mark.parametrize(
    "payload",
    (
        b'{"track":"routing","track":"joint"}',
        b'{"metric":NaN}',
        b'{"metric":Infinity}',
    ),
)
def test_contract_json_rejects_ambiguous_or_non_finite_values(payload: bytes) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


def test_worker_report_json_is_exactly_the_server_draft_contract(
    tmp_path: Path,
) -> None:
    manifest = RunManifest.model_validate(_golden("manifest.json"))
    store = WorkerArtifactStore(tmp_path / "store")
    _stage_server_manifest(store, manifest)
    assert not hasattr(store, "recover_report_bundle")
    assert not hasattr(store, "execution_lease")

    report = run_worker_evaluation(manifest, store)
    worker_payload = store.read_run_json(manifest.run_id, "report.json")

    assert report.run.status == "completed"
    assert {
        "attestation_revision",
        "method_reports",
        "recommendations",
        "routing_recipe_report",
        "statistics",
    }.isdisjoint(worker_payload)
    assert {"track_evidence_levels", "controlled_pair"}.isdisjoint(
        worker_payload["run"]
    )
    status_path = store.runs / manifest.run_id / "status.json"
    server_status = b'{"owner":"go","status":"sealing"}\n'
    status_path.write_bytes(server_status)
    status_path.chmod(0o600)

    assert run_worker_evaluation(manifest, store) == report
    assert status_path.read_bytes() == server_status
    assert not (store.runs / manifest.run_id / "events.jsonl").exists()


def test_worker_rejects_a_published_report_with_pending_transaction_state(
    tmp_path: Path,
) -> None:
    manifest = RunManifest.model_validate(_golden("manifest.json"))
    store = WorkerArtifactStore(tmp_path / "store")
    _stage_server_manifest(store, manifest)
    run_worker_evaluation(manifest, store)
    preparing = store.runs / manifest.run_id / ".report-preparing"
    preparing.mkdir(mode=0o700)

    with pytest.raises(StoreError, match="incomplete transaction"):
        run_worker_evaluation(manifest, store)

    assert preparing.is_dir()


def test_worker_report_draft_excludes_server_publication_fields() -> None:
    payload = _golden("worker-report-draft.json")
    assert {
        "attestation_revision",
        "method_reports",
        "recommendations",
        "routing_recipe_report",
        "statistics",
    }.isdisjoint(payload)
    assert {"track_evidence_levels", "controlled_pair"}.isdisjoint(payload["run"])

    without_report_version = dict(payload)
    without_report_version.pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version"):
        WorkerReportDraft.model_validate(without_report_version)

    without_run_version = dict(payload["run"])
    without_run_version.pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version"):
        WorkerRunState.model_validate(without_run_version)

    without_provenance_version = dict(payload["provenance"])
    without_provenance_version.pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version"):
        WorkerReportDraft.model_validate(
            {**payload, "provenance": without_provenance_version}
        )

    for field, value in (
        ("attestation_revision", "evaluation-server-attestation.v2"),
        ("method_reports", []),
        ("routing_recipe_report", None),
        ("statistics", []),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkerReportDraft.model_validate({**payload, field: value})

    for field, value in (
        ("track_evidence_levels", {}),
        ("controlled_pair", {"pair_id": "forged", "role": "candidate"}),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkerRunState.model_validate({**payload["run"], field: value})

    for verdict in ("not_applicable", "waived"):
        invalid_summary = {**payload["summary"], "verdict": verdict}
        with pytest.raises(ValidationError, match="verdict"):
            WorkerReportDraft.model_validate({**payload, "summary": invalid_summary})


def test_worker_report_draft_requires_one_complete_terminal_run_state() -> None:
    payload = _golden("worker-report-draft.json")
    mutations = (
        (("run", "status"), "failed", "fully completed"),
        (("run", "started_at"), None, "fully completed"),
        (("run", "completed_at"), None, "fully completed"),
        (("run", "error"), "worker failed", "fully completed"),
        (("run", "progress", "percent"), 99, "fully completed"),
        (("run", "progress", "completed"), 7, "fully completed"),
        (("run", "progress", "total"), 7, "fully completed"),
        (("run", "progress", "current_track_id"), "capacity", "fully completed"),
        (
            ("run", "completed_at"),
            "2025-12-31T23:59:59Z",
            "cannot precede",
        ),
    )
    for path, replacement, message in mutations:
        invalid: dict[str, Any] = copy.deepcopy(payload)
        owner: dict[str, Any] = invalid
        for segment in path[:-1]:
            owner = owner[segment]
        owner[path[-1]] = replacement
        with pytest.raises(ValidationError, match=message):
            WorkerReportDraft.model_validate(invalid)


def test_existing_report_metadata_drift_cannot_mutate_standalone_control_state(
    tmp_path: Path,
) -> None:
    manifest = RunManifest.model_validate(_golden("manifest.json"))
    worker_store = WorkerArtifactStore(tmp_path / "store")
    _stage_server_manifest(worker_store, manifest)
    run_worker_evaluation(manifest, worker_store)
    store = LocalArtifactStore(tmp_path / "store")
    report_path = store.runs / manifest.run_id / "report.json"
    report = json.loads(report_path.read_bytes())
    foreign_run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "evaluation-foreign-run"))
    report["run"]["id"] = foreign_run_id
    report["run"]["client_request_id"] = foreign_run_id
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.chmod(0o600)
    store.put_bytes(report_path.read_bytes(), "application/json")
    status_path = store.runs / manifest.run_id / "status.json"
    sentinel = b'{"owner":"standalone-supervisor","status":"sealing"}\n'
    status_path.write_bytes(sentinel)
    status_path.chmod(0o600)

    with pytest.raises(StoreError, match="immutable staged manifest"):
        run_evaluation(manifest, store)

    assert status_path.read_bytes() == sentinel
    assert not (store.runs / foreign_run_id).exists()
