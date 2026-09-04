from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cli.evaluation.canonical import digest_value
from cli.evaluation.compare import compare_reports
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contracts import EvaluationTarget, RunManifest
from cli.evaluation.fixture_executor import execute_fixture
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.gates import compute_gates
from cli.evaluation.metrics import compute_metrics
from cli.evaluation.orchestrator import run_evaluation, validate_manifest
from cli.evaluation.resolution import resolve_snapshot
from cli.evaluation.store import LocalArtifactStore, StoreError


def _manifest(
    run_id: str = "fixture-run",
    sample_limit: int = 4,
    *,
    baseline_run_id: str | None = None,
    code_revision: str = "sha256:" + "1" * 64,
) -> RunManifest:
    return RunManifest(
        manifest_digest="sha256:" + "0" * 64,
        run_id=run_id,
        mode="replay",
        target=EvaluationTarget(id="fixture", kind="builtin-fixture"),
        change_profile="schema_adapter",
        gate_contract_version="evaluation-release-gates.v1",
        suite_ids=("evaluation-smoke",),
        suite_revisions={"evaluation-smoke": "builtin-v1"},
        track_ids=TRACK_IDS,
        sample_limit=sample_limit,
        concurrency=2,
        seed=17,
        baseline_run_id=baseline_run_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision=code_revision,
        policy_snapshot_digest=fixture_inputs().policy.recipe_digest,
        config_digest="sha256:"
        + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        redaction_policy="public-safe-v1",
    )


def _live_manifest(
    run_id: str,
    *,
    envoy_url: str = "http://envoy:8801",
    price_delta: float = 0,
    topology_digest: str = "sha256:" + "b" * 64,
) -> RunManifest:
    arms = fixture_inputs().arms
    if price_delta:
        first = arms[0].model_copy(
            update={
                "input_cost_per_million_tokens_usd": (
                    arms[0].input_cost_per_million_tokens_usd + price_delta
                )
            }
        )
        arms = (first, *arms[1:])
    return RunManifest(
        manifest_digest="sha256:" + "0" * 64,
        run_id=run_id,
        mode="live",
        target=EvaluationTarget(
            id="runtime",
            kind="runtime",
            router_api_url="http://router:8080",
            envoy_url=envoy_url,
            backend_topology_digest=topology_digest,
            model_arms=arms,
        ),
        change_profile="recipe",
        gate_contract_version="evaluation-release-gates.v1",
        suite_ids=("live-joint",),
        suite_revisions={"live-joint": "executor-v1"},
        track_ids=("routing", "model_pool", "joint"),
        sample_limit=4,
        concurrency=2,
        seed=17,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value("live-policy"),
        config_digest="sha256:"
        + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        redaction_policy="public-safe-v1",
    )


def _resolved_live(manifest: RunManifest, store: LocalArtifactStore):
    inputs = fixture_inputs()
    return resolve_snapshot(
        manifest,
        inputs,
        store.put_json(inputs.visible),
        store.put_json(inputs.grading),
        None,
        (),
    )


def test_fixture_run_completes_all_tracks_with_rich_bundle(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)

    assert tuple(track.track_id for track in report.tracks) == TRACK_IDS
    assert all(track.status == "completed" for track in report.tracks)
    assert report.summary.coverage.evaluated == report.summary.coverage.total == 29
    assert report.summary.failed_gates == 0
    assert report.summary.quality_score is None
    assert report.summary.runtime_cost is None
    assert report.summary.capacity_tco is None
    assert report.costs.runtime.amount is not None
    assert report.costs.capacity_tco.amount is not None
    assert report.costs.evaluation_overhead.amount is not None
    assert report.recommendations[0].startswith("E0 diagnostic only")
    assert not any(row.startswith("[AF-") for row in report.recommendations)
    verdicts = {gate.id: gate.verdict for gate in report.gates}
    assert verdicts["G8"] == "not_applicable"
    assert verdicts["G9"] == "not_applicable"
    metrics = {metric.id: metric.value for metric in report.metrics}
    assert {
        "routing.abstention_rate",
        "routing.fallback_rate",
        "model_pool.best_single_quality",
        "model_pool.oracle_gain",
        "model_pool.unique_win_rate",
        "model_pool.selection_entropy_bits",
        "model_pool.selection_arm_coverage",
        "joint.normalized_regret",
        "joint.reliability",
        "safety.violation_upper_95",
        "capacity.cost_per_successful_request",
    } <= set(metrics)
    assert metrics["safety.violation_rate"] == 0
    assert metrics["safety.violation_upper_95"] > 0
    assert any(metric_id.endswith("marginal_contribution") for metric_id in metrics)

    names = {artifact.name for artifact in report.artifacts}
    assert names == {
        "metrics.json",
        "gates.json",
        "provenance.json",
        "failure-summary.json",
        "checksums.sha256",
    }
    assert "report.json" not in names
    assert all("/" not in (artifact.uri or "") for artifact in report.artifacts)

    checksum_lines = (
        (store.runs / report.run.id / "checksums.sha256").read_text().splitlines()
    )
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    assert set(checksums) == names - {"checksums.sha256"}
    for name, expected in checksums.items():
        actual = hashlib.sha256(
            (store.runs / report.run.id / name).read_bytes()
        ).hexdigest()
        assert actual == expected

    private_checksum_lines = (
        (store.runs / report.run.id / "private-checksums.sha256")
        .read_text()
        .splitlines()
    )
    private_checksums = dict(
        line.split("  ", 1)[::-1] for line in private_checksum_lines
    )
    assert {
        "run-manifest.json",
        "cases.jsonl",
        "grading-cases.jsonl",
        "records.jsonl",
        "lineage.json",
        "failure-cases.jsonl",
        "report.md",
        "report.html",
        "checksums.sha256",
    } <= set(private_checksums)
    assert "private-checksums.sha256" not in names


def test_fixture_report_is_deterministic_and_coverage_is_case_based(
    tmp_path: Path,
) -> None:
    first = run_evaluation(_manifest("same-run", 2), LocalArtifactStore(tmp_path / "a"))
    second = run_evaluation(
        _manifest("same-run", 2), LocalArtifactStore(tmp_path / "b")
    )

    first_payload = first.model_dump(mode="json", exclude_none=False)
    second_payload = second.model_dump(mode="json", exclude_none=False)
    for payload in (first_payload, second_payload):
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
    assert first_payload == second_payload
    assert first.summary.coverage.evaluated == 15
    assert first.summary.coverage.total == 15
    assert first.summary.coverage.fraction == 1.0


def test_existing_report_is_idempotent_only_for_the_same_frozen_manifest(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _manifest("idempotent-run")
    first = run_evaluation(manifest, store)
    assert run_evaluation(manifest, store) == first

    changed_code = manifest.model_copy(update={"code_revision": "sha256:" + "2" * 64})
    with pytest.raises(StoreError, match="different run manifest"):
        run_evaluation(changed_code, store)

    changed_suite = manifest.model_copy(
        update={"suite_revisions": {"evaluation-smoke": "builtin-v2"}}
    )
    with pytest.raises(ValueError, match="suite revisions"):
        run_evaluation(changed_suite, store)


def test_hidden_grading_is_not_written_to_policy_visible_cases(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)
    visible_payload = (store.runs / report.run.id / "cases.jsonl").read_text()

    assert "expected_answer" not in visible_payload
    assert "expected_route" not in visible_payload
    assert "preferred_arm_id" not in visible_payload
    assert "should_block" not in visible_payload
    assert "grading-cases.jsonl" not in {artifact.name for artifact in report.artifacts}
    assert store.read_run_text(report.run.id, "grading-cases.jsonl")


def test_real_regression_produces_a_failing_gate() -> None:
    inputs = fixture_inputs()
    records = execute_fixture(inputs.visible, inputs.grading, inputs.fixture, TRACK_IDS)
    records = [
        (
            row.model_copy(update={"safety_violations": 1})
            if row.track_id == "safety" and row.case_id == "safety-1"
            else row
        )
        for row in records
    ]
    gates = compute_gates(
        compute_metrics(records), has_records=True, cost_accounted=True
    )
    assert {gate.id: gate.verdict for gate in gates}["G2"] == "fail"


def test_compare_is_paired_by_metric_id(tmp_path: Path) -> None:
    baseline = run_evaluation(_manifest("baseline"), LocalArtifactStore(tmp_path / "a"))
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id="baseline",
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "b"),
    )
    comparison = compare_reports(baseline, candidate)

    assert comparison.baseline_run_id == "baseline"
    assert comparison.candidate_run_id == "candidate"
    assert all(
        metric.delta == 0 for metric in comparison.metrics if metric.value is not None
    )
    assert comparison.verdict == "unavailable"


def test_compare_rejects_different_workloads(tmp_path: Path) -> None:
    baseline = run_evaluation(
        _manifest("baseline", sample_limit=4), LocalArtifactStore(tmp_path / "a")
    )
    candidate = run_evaluation(
        _manifest(
            "candidate",
            sample_limit=2,
            baseline_run_id="baseline",
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "b"),
    )

    with pytest.raises(ValueError, match="workload_snapshot_digest"):
        compare_reports(baseline, candidate)


def test_compare_rejects_self_and_unlinked_candidate(tmp_path: Path) -> None:
    baseline = run_evaluation(_manifest("baseline"), LocalArtifactStore(tmp_path / "a"))
    unlinked = run_evaluation(
        _manifest("candidate", code_revision="sha256:" + "2" * 64),
        LocalArtifactStore(tmp_path / "b"),
    )

    with pytest.raises(ValueError, match="cannot be compared with itself"):
        compare_reports(baseline, baseline)
    with pytest.raises(ValueError, match="baseline_run_id"):
        compare_reports(baseline, unlinked)


def test_live_factor_snapshot_ids_are_stable_and_detect_pool_environment_drift(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first = _resolved_live(_live_manifest("first"), store)
    second = _resolved_live(_live_manifest("second"), store)

    assert first.manifest_digest == "sha256:" + "0" * 64
    assert first.manifest_digest != digest_value(_live_manifest("first"))
    assert first.policy.id == second.policy.id
    assert first.pool.id == second.pool.id
    assert first.binding.id == second.binding.id
    assert first.environment.id == second.environment.id

    changed_environment = _resolved_live(
        _live_manifest("changed-env", envoy_url="http://envoy:9901"), store
    )
    changed_pool = _resolved_live(
        _live_manifest("changed-pool", price_delta=0.5), store
    )
    changed_topology = _resolved_live(
        _live_manifest("changed-topology", topology_digest="sha256:" + "c" * 64),
        store,
    )
    assert changed_environment.environment.id != first.environment.id
    assert changed_environment.pool.id == first.pool.id
    assert changed_pool.pool.id != first.pool.id
    assert changed_pool.binding.id != first.binding.id
    assert changed_topology.environment.id != first.environment.id
    assert changed_topology.pool.id == first.pool.id


def test_live_suite_selection_rejects_unsupported_target_capabilities() -> None:
    manifest = _live_manifest("unsupported")
    manifest = manifest.model_copy(
        update={"target": manifest.target.model_copy(update={"router_api_url": None})}
    )

    with pytest.raises(ValueError, match="joint, model_pool, routing"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    ("pool_snapshot_digest", "environment_snapshot_digest"),
)
def test_compare_rejects_pool_or_environment_drift(
    tmp_path: Path,
    field: str,
) -> None:
    baseline = run_evaluation(_manifest("baseline"), LocalArtifactStore(tmp_path / "a"))
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id="baseline",
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "b"),
    )
    changed_provenance = candidate.provenance.model_copy(
        update={field: "sha256:" + "f" * 64}
    )
    candidate = candidate.model_copy(update={"provenance": changed_provenance})

    with pytest.raises(ValueError, match=field):
        compare_reports(baseline, candidate)


def test_failure_cases_remain_explicit_in_bundle(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)
    rows = [
        json.loads(line)
        for line in (store.runs / report.run.id / "failure-cases.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows
    assert all(row["status"] == "failed" for row in rows)
    assert "failure-cases.jsonl" not in {artifact.name for artifact in report.artifacts}
    summary = store.read_run_json(report.run.id, "failure-summary.json")
    assert summary["failed"] == len(rows)
    assert "case_id" not in json.dumps(summary)
