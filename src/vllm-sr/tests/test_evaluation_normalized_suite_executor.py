from __future__ import annotations

import base64
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from benchmark_normalization_fixtures import write_native_fixture
from cli.commands.eval import eval
from cli.evaluation.benchmark_normalization import normalize_benchmark_suite
from cli.evaluation.broker_client import BrokerProtocolError
from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.catalog import get_catalog
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contract_primitives import SecretRef
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.execution_contract import (
    NORMALIZED_LIVE_EXECUTOR_ID,
    NORMALIZED_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.http_client import HTTPResult
from cli.evaluation.live_executor import LiveRawResult
from cli.evaluation.normalized_suite_inputs import SelectedCase, evidence_kind
from cli.evaluation.normalized_suite_live_admission import (
    NORMALIZED_MULTIMODAL_LIVE_METHOD_ID,
)
from cli.evaluation.normalized_suite_live_executor import (
    execute_normalized_suite_live,
)
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import (
    NormalizedMultimodalObservation,
    NormalizedPerturbation,
)
from cli.evaluation.suite_install_contract import NormalizedMediaEntry
from cli.evaluation.suite_store import NormalizedSuiteStore
from click.testing import CliRunner
from evaluation_normalized_suite_test_support import (
    _PIXEL,
    _PRIVATE_MARKERS,
    _base_bundle,
    _catalog,
    _decision,
    _digest,
    _install_composite,
    _install_r2_suite,
    _live_manifest,
    _manifest,
    _qualification_cases,
    _receipt,
    _suite_request,
    _target_mixture,
    _trusted_source_verifier,
    _write_jsonl,
)

pytestmark = pytest.mark.usefixtures(_trusted_source_verifier.__name__)


def _strip_nondeterministic_report_fields(payload: dict[str, Any]) -> None:
    payload["run"]["started_at"] = None
    payload["run"]["completed_at"] = None
    payload["provenance"]["generated_at"] = None
    for artifact in payload["artifacts"]:
        artifact["digest"] = None
        artifact["size_bytes"] = None
    for gate in payload["gates"]:
        gate["evaluated_at"] = None
    for track in payload["tracks"]:
        for gate in track["gates"]:
            gate["evaluated_at"] = None


def _run_composite_cli(
    manifest_path: Path,
    first_store: Path,
    suite_store: NormalizedSuiteStore,
) -> Any:
    runner = CliRunner()
    validated = runner.invoke(
        eval,
        [
            "validate",
            "--manifest",
            str(manifest_path),
            "--suite-store",
            str(suite_store.root),
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"] is True
    executed = runner.invoke(
        eval,
        [
            "run",
            "--manifest",
            str(manifest_path),
            "--store",
            str(first_store),
            "--suite-store",
            str(suite_store.root),
        ],
    )
    assert executed.exit_code == 0, executed.output
    return executed


def test_installed_composite_executes_all_tracks_deterministically_without_leaks(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_ids = _install_composite(tmp_path / "bundles", suite_store)
    manifest = _manifest("normalized-composite", suite_ids, suite_store)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    first_store = tmp_path / "evaluation-a"
    executed = _run_composite_cli(manifest_path, first_store, suite_store)
    first = json.loads(executed.output)
    assert (
        run_evaluation(
            manifest,
            LocalArtifactStore(first_store),
            suite_store=suite_store,
        ).model_dump(mode="json", exclude_none=False)
        == first
    )
    second = run_evaluation(
        manifest,
        LocalArtifactStore(tmp_path / "evaluation-b"),
        suite_store=suite_store,
    )

    second_payload = second.model_dump(mode="json", exclude_none=False)
    for payload in (first, second_payload):
        _strip_nondeterministic_report_fields(payload)
    assert first == second_payload
    assert {track["track_id"] for track in first["tracks"]} == set(TRACK_IDS)
    assert all(track["status"] == "completed" for track in first["tracks"])
    assert first["run"]["evidence_level"] == "E0"
    assert all(
        _catalog(suite_store).get(suite_id).evidence_level == "E0"
        for suite_id in suite_ids
    )
    assert first["summary"]["verdict"] == "unavailable"
    expected_revisions = {
        suite_id: suite_store.get_suite_manifest(suite_id).revision
        for suite_id in suite_ids
    }
    assert first["provenance"]["benchmark_revisions"] == expected_revisions

    public_names = {artifact["name"] for artifact in first["artifacts"]}
    public_payload = executed.output + "".join(
        (first_store / "runs" / manifest.run_id / name).read_text()
        for name in public_names
        if name.endswith((".json", ".jsonl", ".md", ".html"))
    )
    assert all(marker not in public_payload for marker in _PRIVATE_MARKERS)
    assert "xroute-private-case" not in public_payload
    assert "ace-private-case" not in public_payload
    assert "r2-private-case" not in public_payload
    records = (first_store / "runs" / manifest.run_id / "records.jsonl").read_text()
    assert "normalized suite does not declare this track" not in records
    parsed_records = [
        ExecutionRecord.model_validate_json(row) for row in records.splitlines()
    ]
    assert all(
        record.method_id is None
        for record in parsed_records
        if record.track_id == "model_pool"
    )

    lineage_path = first_store / "runs" / manifest.run_id / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    assert lineage["schema_version"] == SCHEMA_VERSION
    assert lineage["resolved_snapshot"]["policy"]["id"] != "fixture-policy"
    assert (
        lineage["resolved_snapshot"]["environment"]["platform"]
        == "normalized-suite-replay"
    )
    assert any(
        row["source_id"] == "secret-arm-a"
        for row in lineage["normalized_suite_identities"]["arm_identities"]
    )
    identity_lineage = lineage["normalized_suite_identities"]
    arm_ids = {row["opaque_id"] for row in identity_lineage["arm_identities"]}
    action_ids = {row["opaque_id"] for row in identity_lineage["action_identities"]}
    selected_ids = {
        record.selected_arm_id
        for record in parsed_records
        if record.selected_arm_id is not None
    }
    assert arm_ids and action_ids and arm_ids.isdisjoint(action_ids)
    assert {record.arm_id for record in parsed_records if record.arm_id} <= arm_ids
    assert {
        record.action_id for record in parsed_records if record.action_id
    } <= action_ids
    assert selected_ids <= arm_ids | action_ids
    assert selected_ids & arm_ids
    assert selected_ids & action_ids
    assert stat.S_IMODE(lineage_path.stat().st_mode) == 0o600
    assert (
        "HIDDEN EXPECTED ANSWER"
        in (first_store / "runs" / manifest.run_id / "grading-cases.jsonl").read_text()
    )


def _install_registered_mmr(
    root: Path,
    store: NormalizedSuiteStore,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    source_root = root / "source"
    export_root = root / "native"
    source_root.mkdir(parents=True)
    write_native_fixture("mmr-bench", export_root)
    monkeypatch.setattr(
        "cli.evaluation.benchmark_normalization.require_verified_benchmark_source",
        lambda descriptor, _root: _receipt(descriptor.id),
    )
    result = normalize_benchmark_suite(
        adapter_id="mmr-bench",
        source_root=source_root,
        export_root=export_root,
        output_root=root / "normalized",
        suite_id="registered-mmr-live",
    )
    return store.install(
        result.request,
        result.bundle_path,
        source_root=source_root,
        native_export_root=export_root,
    ).id


def _install_user_provided_mmr(
    root: Path,
    store: NormalizedSuiteStore,
) -> str:
    case_id = "user-provided-mmr-case"
    _base_bundle(
        root,
        case_id,
        track_ids=("model_pool", "multimodal"),
        image=True,
        expected_answer="one",
    )
    _write_jsonl(
        root / "grading/multimodal-observations.jsonl",
        (
            NormalizedMultimodalObservation(
                case_id=case_id,
                modality="image",
                supported=True,
                quality=1.0,
                privacy_violations=0,
                source_record_digest=_digest("user-provided-mmr-observation"),
            ),
        ),
    )
    media_bytes = base64.b64decode(_PIXEL.partition(",")[2], validate=True)
    _write_jsonl(
        root / "metadata/media.jsonl",
        (
            NormalizedMediaEntry(
                id="user-provided-image",
                digest="sha256:" + hashlib.sha256(media_bytes).hexdigest(),
                media_type="image/png",
                size_bytes=len(media_bytes),
                modality="image",
                license_id="upstream",
            ),
        ),
    )
    request = _suite_request(
        root,
        adapter_id="mmr-bench",
        suite_id="user-provided-mmr",
        case_id=case_id,
        tracks=("model_pool", "multimodal"),
        optional_roles=("multimodal_observations", "media_manifest"),
    )
    return store.install(request, root, source_root=root.parent).id


def _target_executor(
    observed_case_ids: list[str],
    *,
    broker_bound: bool = True,
) -> Any:
    def execute(visible: Any, **kwargs: object) -> LiveRawResult:
        assert kwargs["track_ids"] == ("multimodal",)
        assert kwargs["mixture"] == _target_mixture()
        assert "router_api_key_env" not in kwargs
        assert "envoy_api_key_env" not in kwargs
        case = visible.cases[0]
        observed_case_ids.append(case.id)
        receipt = _digest(f"broker-{case.id}") if broker_bound else None
        response = HTTPResult(
            success=True,
            status_code=200,
            payload={
                "choices": [{"message": {"content": "  one  "}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
            latency_ms=8.0,
            headers={},
            broker_receipt=receipt,
        )
        return LiveRawResult(
            records=[
                ExecutionRecord(
                    id=f"multimodal-{case.id}",
                    track_id="multimodal",
                    case_id=case.id,
                    attempt_id=f"attempt-{case.id}",
                    status="succeeded",
                    success=True,
                    modality="image",
                    latency_ms=8.0,
                    broker_receipt=receipt,
                ),
            ],
            discovered_entrypoints=("entrypoint-a",),
            routing_traces=(
                RoutingDiagnostic(
                    case_id=case.id,
                    selected_model="provider-strong",
                    selection_status="selected",
                ),
            ),
            chat_results={case.id: response},
            model_pool_results={},
            model_pool_arm_ids=(),
            joint_results={},
        )

    return execute


def test_authenticated_normalized_live_executor_requires_dashboard_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = _install_registered_mmr(tmp_path / "bundles", suite_store, monkeypatch)
    manifest = _live_manifest(
        "authenticated-normalized-live",
        suite_id,
        suite_store,
        track_ids=("multimodal",),
    )
    manifest = manifest.with_semantic_updates(
        target=manifest.target.model_copy(
            update={"envoy_api_key": SecretRef(env="ENVOY_EVAL_KEY")}
        )
    )

    with pytest.raises(BrokerProtocolError, match="requires the Dashboard HTTP broker"):
        execute_normalized_suite_live(
            manifest=manifest,
            store=suite_store,
            manifests=(suite_store.get_suite_manifest(suite_id),),
            executor_id=NORMALIZED_LIVE_EXECUTOR_ID,
        )


def _assert_registered_live_catalog(
    suite_store: NormalizedSuiteStore,
    suite_id: str,
) -> None:
    source_catalog = _catalog(suite_store).get(suite_id)
    assert source_catalog.evidence_level == "E0"
    assert source_catalog.track_ids == ("model_pool", "multimodal")
    assert source_catalog.modes == ("replay", "live")
    live_methods = tuple(
        method
        for method in source_catalog.methods
        if method.evidence_source != "normalized_import"
    )
    assert len(live_methods) == 1
    assert live_methods[0].id == NORMALIZED_MULTIMODAL_LIVE_METHOD_ID
    assert live_methods[0].track_id == "multimodal"


def test_same_installed_workload_replays_history_or_executes_current_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = _install_registered_mmr(tmp_path / "bundles", suite_store, monkeypatch)
    _assert_registered_live_catalog(suite_store, suite_id)
    replay = _manifest("target-workload-replay", (suite_id,), suite_store)
    replay = replay.with_semantic_updates(track_ids=("multimodal",), sample_limit=1)
    replay_store = LocalArtifactStore(tmp_path / "replay-store")
    replay_report = run_evaluation(replay, replay_store, suite_store=suite_store)
    assert (
        run_evaluation(replay, replay_store, suite_store=suite_store) == replay_report
    )

    observed_case_ids: list[str] = []
    monkeypatch.setattr(
        "cli.evaluation.normalized_suite_live_executor.execute_live_raw",
        _target_executor(observed_case_ids),
    )
    live = _live_manifest(
        "target-workload-live", suite_id, suite_store, track_ids=("multimodal",)
    )
    live_store = LocalArtifactStore(tmp_path / "live-store")
    report = run_evaluation(live, live_store, suite_store=suite_store)

    replay_cases = replay_store.read_run_bytes(
        replay.run_id, "cases.jsonl"
    ).splitlines()
    live_cases = live_store.read_run_bytes(live.run_id, "cases.jsonl").splitlines()
    assert [json.loads(row)["id"] for row in replay_cases] == observed_case_ids
    assert [json.loads(row)["id"] for row in live_cases] == observed_case_ids

    records = [
        ExecutionRecord.model_validate_json(row)
        for row in live_store.read_run_bytes(live.run_id, "records.jsonl").splitlines()
    ]
    assert len(records) == 1
    multimodal = records[0]
    assert multimodal.track_id == "multimodal"
    assert multimodal.quality == 1.0
    assert multimodal.grader == "normalized-suite-hidden-answer-exact.v1"
    assert multimodal.evidence_kind == NORMALIZED_LIVE_EXECUTOR_ID
    assert multimodal.broker_receipt is not None

    lineage = live_store.read_run_json(live.run_id, "lineage.json")
    assert lineage["schema_version"] == SCHEMA_VERSION
    identities = lineage["normalized_suite_identities"]
    assert identities["arm_identities"] == []
    assert identities["action_identities"] == []
    assert (
        lineage["resolved_snapshot"]["environment"]["target_id"] == _target_mixture().id
    )
    assert "fixture_ref" not in lineage["resolved_snapshot"]
    assert report.run.evidence_level == "E4"
    observed_before_reload = tuple(observed_case_ids)
    assert run_evaluation(live, live_store, suite_store=suite_store) == report
    assert tuple(observed_case_ids) == observed_before_reload
    campaign = next(
        profile
        for profile in get_catalog(generated_at=False).change_profiles
        if profile.id == "agent_multimodal"
    )
    g5 = next(slot for slot in campaign.campaign_slots if slot.gate_id == "G5")
    assert g5.track_id == "multimodal"
    assert g5.minimum_evidence_level == report.run.evidence_level
    assert g5.accepted_executor_ids == (NORMALIZED_LIVE_EXECUTOR_ID,)

    model_pool_live = _live_manifest(
        "mmr-model-pool-live", suite_id, suite_store, track_ids=("model_pool",)
    )
    with pytest.raises(ValueError, match="no first-party normalized live method"):
        run_evaluation(
            model_pool_live,
            LocalArtifactStore(tmp_path / "blocked-model-pool"),
            suite_store=suite_store,
        )

    monkeypatch.setattr(
        "cli.evaluation.normalized_suite_live_executor.execute_live_raw",
        _target_executor([], broker_bound=False),
    )
    unbound = _live_manifest(
        "mmr-unbound-live", suite_id, suite_store, track_ids=("multimodal",)
    )
    unbound_report = run_evaluation(
        unbound,
        LocalArtifactStore(tmp_path / "unbound-live"),
        suite_store=suite_store,
    )
    assert unbound_report.run.evidence_level == "E0"


def test_non_admitted_normalized_imports_cannot_reach_live_tracks(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = _install_r2_suite(tmp_path / "bundles", suite_store)
    source_catalog = _catalog(suite_store).get(suite_id)
    assert source_catalog.evidence_level == "E0"
    assert source_catalog.modes == ("replay",)
    manifest = _live_manifest(
        "target-capacity-no-replay-qualification",
        suite_id,
        suite_store,
        track_ids=("capacity",),
    )

    with pytest.raises(ValueError, match="no first-party normalized live method"):
        run_evaluation(
            manifest,
            LocalArtifactStore(tmp_path / "evaluation"),
            suite_store=suite_store,
        )

    mmr_id = _install_user_provided_mmr(tmp_path / "bundles" / "user-mmr", suite_store)
    mmr_catalog = _catalog(suite_store).get(mmr_id)
    assert mmr_catalog.evidence_level == "E0"
    assert mmr_catalog.modes == ("replay",)
    assert all(
        method.id != NORMALIZED_MULTIMODAL_LIVE_METHOD_ID
        for method in mmr_catalog.methods
    )
    mmr_live = _live_manifest(
        "user-mmr-live", mmr_id, suite_store, track_ids=("multimodal",)
    )
    with pytest.raises(ValueError, match="no first-party normalized live method"):
        run_evaluation(
            mmr_live,
            LocalArtifactStore(tmp_path / "blocked-user-mmr"),
            suite_store=suite_store,
        )


def test_declared_track_without_qualification_artifact_is_unavailable(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    bundle = tmp_path / "missing-safety"
    _base_bundle(bundle, "missing-private-case", track_ids=("safety",))
    request = _suite_request(
        bundle,
        adapter_id="acebench",
        suite_id="missing-safety-suite",
        case_id="missing-private-case",
        tracks=("safety",),
        optional_roles=(),
    )
    installed = suite_store.install(request, bundle, source_root=bundle.parent)
    manifest = _manifest(
        "missing-safety-run", (installed.id,), suite_store
    ).with_semantic_updates(track_ids=("safety",))

    report = run_evaluation(
        manifest,
        LocalArtifactStore(tmp_path / "evaluation"),
        suite_store=suite_store,
    )

    assert report.run.evidence_level == "E0"
    assert report.tracks[0].status == "unavailable"
    assert report.tracks[0].coverage.evaluated == 0
    records = (
        tmp_path / "evaluation" / "runs" / manifest.run_id / "records.jsonl"
    ).read_text()
    assert '"status":"unavailable"' in records
    assert "lacks safety enforcement observations" in records
    assert "PRIVATE NORMALIZED PROMPT" not in records
    assert "missing-private-case" not in records


def test_composite_sampling_is_stratified_per_suite_and_preserves_track_union(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_ids = _install_composite(tmp_path / "bundles", suite_store)
    manifest = _manifest("normalized-stratified", suite_ids, suite_store)
    manifest = manifest.with_semantic_updates(sample_limit=1)
    artifact_store = LocalArtifactStore(tmp_path / "evaluation")

    report = run_evaluation(
        manifest,
        artifact_store,
        suite_store=suite_store,
    )

    assert tuple(track.track_id for track in report.tracks) == TRACK_IDS
    assert all(track.status == "completed" for track in report.tracks)
    assert report.run.evidence_level == "E0"
    records = [
        ExecutionRecord.model_validate_json(row)
        for row in artifact_store.read_run_bytes(
            manifest.run_id, "records.jsonl"
        ).splitlines()
    ]
    evidence_by_track = {
        track_id: {row.evidence_kind for row in records if row.track_id == track_id}
        for track_id in TRACK_IDS
    }
    assert evidence_by_track == {
        track_id: {"normalized-suite-replay.v1;ceiling=E0"} for track_id in TRACK_IDS
    }


def test_imported_record_evidence_is_always_e0(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    bundle = tmp_path / "bundles" / "routerarena"
    case_id = "routerarena-case"
    _qualification_cases(bundle, (case_id,), track_ids=("routing",))
    _write_jsonl(bundle / "grading/decisions.jsonl", (_decision(case_id),))
    suite_id = suite_store.install(
        _suite_request(
            bundle,
            adapter_id="routerarena",
            suite_id="imported-routerarena",
            case_id=case_id,
            tracks=("routing",),
            optional_roles=("decisions",),
        ),
        bundle,
        source_root=bundle.parent,
    ).id
    manifest = suite_store.get_suite_manifest(suite_id)
    visible = next(suite_store.load_jsonl(suite_id, "visible_cases"))
    grading = next(suite_store.load_jsonl(suite_id, "grading_cases"))
    case = SelectedCase(
        manifest=manifest,
        source_visible=visible,  # type: ignore[arg-type]
        source_grading=grading,  # type: ignore[arg-type]
        visible=visible,  # type: ignore[arg-type]
        grading=grading,  # type: ignore[arg-type]
        executor_id=NORMALIZED_REPLAY_EXECUTOR_ID,
    )

    assert manifest.qualification_receipt.evidence_level == "E0"
    assert manifest.qualification_receipt.qualified_gate_ids == ()
    assert evidence_kind(case, "routing") == "normalized-suite-replay.v1;ceiling=E0"


def test_imported_robustness_pairs_remain_e0_and_cannot_pass_g4(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "robustness"
    case_ids = ("source", "perturbed")
    _qualification_cases(bundle, case_ids, track_ids=("routing",))
    _write_jsonl(
        bundle / "grading/decisions.jsonl",
        (_decision(case_id) for case_id in case_ids),
    )
    _write_jsonl(
        bundle / "grading/perturbations.jsonl",
        (
            NormalizedPerturbation(
                pair_id="pair-1",
                source_case_id="source",
                perturbed_case_id="perturbed",
                relation="invariant",
                slice_ids=("routerarena:paraphrase",),
                native_pair_count=1,
                source_record_digest=_digest("pair-1"),
            ),
        ),
    )
    request = _suite_request(
        bundle,
        adapter_id="routerarena",
        suite_id="imported-robustness",
        case_id="source",
        tracks=("routing",),
        optional_roles=("decisions", "perturbations"),
        case_count=2,
    )

    suite_store = NormalizedSuiteStore(tmp_path / "store")
    manifest = suite_store.install(request, bundle, source_root=bundle.parent)
    assert manifest.qualification_receipt.evidence_level == "E0"
    assert manifest.qualification_receipt.qualified_gate_ids == ()
    run = _manifest(
        "method-only-g4", (manifest.id,), suite_store
    ).with_semantic_updates(track_ids=("routing",))
    report = run_evaluation(
        run, LocalArtifactStore(tmp_path / "evaluation"), suite_store=suite_store
    )
    assert report.run.evidence_level == "E0"
    assert (
        next(gate for gate in report.gates if gate.id == "G4").verdict == "unavailable"
    )
