from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from cli.commands.eval import eval
from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.canonical import canonical_json_bytes, digest_value
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    EvaluationTarget,
    ImagePart,
    ImageURL,
    Message,
    RunManifest,
)
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import (
    BenchmarkSourceReceipt,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    LICENSE_CONTRACT_VERSION,
    BenchmarkSuiteInstallRequest,
    SuiteArtifactInstall,
    SuiteArtifactRole,
)
from cli.evaluation.suite_store import NormalizedSuiteStore
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _trusted_source_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(descriptor: Any, _source_root: Path) -> BenchmarkSourceReceipt:
        return BenchmarkSourceReceipt(
            adapter_id=descriptor.id,
            expected_source_revision=descriptor.source_revision,
            observed_source_revision=descriptor.source_revision,
            expected_dataset_revision=descriptor.dataset_revision,
            observed_dataset_revision=descriptor.dataset_revision,
            source_clean=True,
            dataset_clean=(True if descriptor.dataset_revision else None),
            verified=True,
        )

    monkeypatch.setattr(
        "cli.evaluation.suite_store.require_verified_benchmark_source", verified
    )


_PIXEL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)
_PRIVATE_MARKERS = (
    "PRIVATE NORMALIZED PROMPT",
    "HIDDEN EXPECTED ANSWER",
    "secret-arm-a",
    "secret-arm-b",
    "private-grader",
)


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _write_license(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": LICENSE_CONTRACT_VERSION,
                "licenses": [
                    {
                        "id": "upstream",
                        "name": "Pinned upstream fixture license",
                        "redistribution": "metadata_only",
                    }
                ],
            }
        )
    )


def _artifact(root: Path, role: SuiteArtifactRole) -> SuiteArtifactInstall:
    relative_path, media_type, _ = ARTIFACT_ROLE_LAYOUT[role]
    content = (root / relative_path).read_bytes()
    return SuiteArtifactInstall(
        role=role,
        relative_path=relative_path,
        digest="sha256:" + hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type=media_type,
    )


def _receipt(adapter_id: str) -> BenchmarkSourceReceipt:
    descriptor = get_benchmark_adapter(adapter_id)
    return BenchmarkSourceReceipt(
        adapter_id=adapter_id,
        expected_source_revision=descriptor.source_revision,
        observed_source_revision=descriptor.source_revision,
        expected_dataset_revision=descriptor.dataset_revision,
        observed_dataset_revision=descriptor.dataset_revision,
        source_clean=True,
        dataset_clean=True if descriptor.dataset_revision else None,
        verified=True,
    )


def _digest(label: str) -> str:
    return digest_value({"private_source_record": label})


def _base_bundle(root: Path, case_id: str, *, image: bool = False) -> None:
    message = Message(
        role="user",
        content=(
            (ImagePart(image_url=ImageURL(url=_PIXEL, detail="low")),)
            if image
            else f"PRIVATE NORMALIZED PROMPT {case_id}"
        ),
    )
    _write_jsonl(
        root / "visible/cases.jsonl",
        (
            CaseVisible(
                id=case_id,
                messages=(message,),
                modality="image" if image else "text",
                trajectory_id=f"private-trajectory-{case_id}",
            ),
        ),
    )
    _write_jsonl(
        root / "grading/cases.jsonl",
        (
            CaseGrading(
                case_id=case_id,
                expected_route="secret-arm-a",
                expected_answer="HIDDEN EXPECTED ANSWER",
                preferred_arm_id="secret-arm-a",
                should_block=False,
            ),
        ),
    )
    _write_license(root / "metadata/licenses.json")


def _decision(case_id: str) -> NormalizedDecision:
    return NormalizedDecision(
        case_id=case_id,
        selected_arm_id="secret-arm-a",
        selection_status="selected",
        success=True,
        latency_ms=2.5,
        source_record_digest=_digest(f"{case_id}-decision"),
    )


def _outcomes(case_id: str) -> tuple[NormalizedOutcome, ...]:
    return (
        NormalizedOutcome(
            case_id=case_id,
            arm_id="secret-arm-a",
            success=True,
            quality=0.9,
            latency_ms=18,
            input_tokens=11,
            output_tokens=7,
            runtime_cost_usd=0.003,
            grader_id="private-grader",
            grader_revision="private-grader-v1",
            split="frozen-test",
            source_record_digest=_digest(f"{case_id}-outcome-a"),
        ),
        NormalizedOutcome(
            case_id=case_id,
            arm_id="secret-arm-b",
            success=True,
            quality=0.6,
            latency_ms=9,
            input_tokens=11,
            output_tokens=5,
            runtime_cost_usd=0.001,
            grader_id="private-grader",
            grader_revision="private-grader-v1",
            split="frozen-test",
            source_record_digest=_digest(f"{case_id}-outcome-b"),
        ),
    )


def _write_common_observations(root: Path, case_id: str) -> list[SuiteArtifactRole]:
    _write_jsonl(root / "grading/decisions.jsonl", (_decision(case_id),))
    _write_jsonl(root / "grading/outcomes.jsonl", _outcomes(case_id))
    return ["decisions", "outcomes"]


def _suite_request(
    root: Path,
    *,
    adapter_id: str,
    suite_id: str,
    case_id: str,
    tracks: tuple[str, ...],
    optional_roles: Iterable[SuiteArtifactRole],
) -> BenchmarkSuiteInstallRequest:
    descriptor = get_benchmark_adapter(adapter_id)
    roles: tuple[SuiteArtifactRole, ...] = (
        "visible_cases",
        "grading_cases",
        *tuple(optional_roles),
        "license_manifest",
    )
    return BenchmarkSuiteInstallRequest(
        id=suite_id,
        name=f"Normalized {descriptor.name} integration suite",
        adapter_id=adapter_id,
        source_receipt=_receipt(adapter_id),
        decision_unit=descriptor.decision_unit,
        action_space=descriptor.action_space,
        track_ids=tracks,  # type: ignore[arg-type]
        evidence_level_ceiling="E5",
        split_protocol="fixed composite integration split",
        case_count=1,
        arm_ids=("secret-arm-a", "secret-arm-b"),
        data_classification="restricted",
        redistribution="metadata_only",
        artifacts=tuple(_artifact(root, role) for role in roles),
        limitations=("integration-normalized evidence only",),
    )


def _install_xroute_suite(root: Path, store: NormalizedSuiteStore) -> str:
    xroute = root / "xroute"
    _base_bundle(xroute, "xroute-private-case", image=True)
    xroute_roles = _write_common_observations(xroute, "xroute-private-case")
    _write_jsonl(
        xroute / "grading/multimodal-observations.jsonl",
        (
            NormalizedMultimodalObservation(
                case_id="xroute-private-case",
                modality="image",
                supported=True,
                quality=0.88,
                privacy_violations=0,
                source_record_digest=_digest("xroute-multimodal"),
            ),
        ),
    )
    _write_jsonl(
        xroute / "grading/preferences.jsonl",
        (
            NormalizedPreference(
                case_id="xroute-private-case",
                left_action_id="secret-arm-a",
                right_action_id="secret-arm-b",
                preference="left",
                chosen_action_id="secret-arm-a",
                reward=1.0,
                exposure_probability=0.5,
                behavior_propensity=0.5,
                participant_digest=_digest("private-participant"),
                source_record_digest=_digest("xroute-preference"),
            ),
        ),
    )
    xroute_roles.extend(("multimodal_observations", "preferences"))
    return store.install(
        _suite_request(
            xroute,
            adapter_id="xroutebench",
            suite_id="composite-xroute",
            case_id="xroute-private-case",
            tracks=("routing", "model_pool", "joint", "multimodal", "preference"),
            optional_roles=xroute_roles,
        ),
        xroute,
        source_root=xroute.parent,
    ).id


def _install_ace_suite(root: Path, store: NormalizedSuiteStore) -> str:
    ace = root / "ace"
    _base_bundle(ace, "ace-private-case")
    ace_roles = _write_common_observations(ace, "ace-private-case")
    _write_jsonl(
        ace / "grading/trajectories.jsonl",
        (
            NormalizedTrajectoryStep(
                trajectory_id="private-trajectory-ace-private-case",
                step_id="private-step-0",
                sequence=0,
                case_id="ace-private-case",
                selected_action_id="secret-arm-a",
                tool_name="private-tool",
                tool_call_valid=True,
                terminal=False,
                privacy_exposures=0,
                source_record_digest=_digest("ace-step-0"),
            ),
            NormalizedTrajectoryStep(
                trajectory_id="private-trajectory-ace-private-case",
                step_id="private-step-1",
                sequence=1,
                case_id="ace-private-case",
                selected_action_id="secret-arm-a",
                terminal=True,
                terminal_success=True,
                task_score=0.95,
                privacy_exposures=0,
                source_record_digest=_digest("ace-step-1"),
            ),
        ),
    )
    _write_jsonl(
        ace / "grading/safety-observations.jsonl",
        (
            NormalizedSafetyObservation(
                case_id="ace-private-case",
                violations=0,
                blocked=False,
                source_record_digest=_digest("ace-safety"),
            ),
        ),
    )
    ace_roles.extend(("trajectories", "safety_observations"))
    return store.install(
        _suite_request(
            ace,
            adapter_id="acebench",
            suite_id="composite-ace",
            case_id="ace-private-case",
            tracks=("routing", "joint", "agentic", "safety"),
            optional_roles=ace_roles,
        ),
        ace,
        source_root=ace.parent,
    ).id


def _install_r2_suite(root: Path, store: NormalizedSuiteStore) -> str:
    r2 = root / "r2"
    _base_bundle(r2, "r2-private-case")
    r2_roles = _write_common_observations(r2, "r2-private-case")
    _write_jsonl(
        r2 / "grading/capacity-observations.jsonl",
        (
            NormalizedCapacityObservation(
                case_id="r2-private-case",
                concurrency=1,
                success=True,
                latency_ms=12,
                throughput_rps=8,
                runtime_cost_usd=0.002,
                capacity_tco_usd=0.003,
                gpu_seconds=0.05,
                energy_kwh=0.0002,
                elapsed_seconds=1,
                source_record_digest=_digest("r2-capacity-1"),
            ),
            NormalizedCapacityObservation(
                case_id="r2-private-case",
                concurrency=2,
                success=True,
                latency_ms=16,
                throughput_rps=14,
                runtime_cost_usd=0.003,
                capacity_tco_usd=0.004,
                gpu_seconds=0.08,
                energy_kwh=0.0003,
                elapsed_seconds=1,
                source_record_digest=_digest("r2-capacity-2"),
            ),
        ),
    )
    r2_roles.append("capacity_observations")
    return store.install(
        _suite_request(
            r2,
            adapter_id="r2-router",
            suite_id="composite-r2",
            case_id="r2-private-case",
            tracks=("routing", "model_pool", "joint", "capacity"),
            optional_roles=r2_roles,
        ),
        r2,
        source_root=r2.parent,
    ).id


def _install_composite(root: Path, store: NormalizedSuiteStore) -> tuple[str, ...]:
    return (
        _install_xroute_suite(root, store),
        _install_ace_suite(root, store),
        _install_r2_suite(root, store),
    )


def _manifest(
    run_id: str,
    suite_ids: tuple[str, ...],
    suite_store: NormalizedSuiteStore,
) -> RunManifest:
    revisions = {
        suite_id: suite_store.get_suite_manifest(suite_id).revision
        for suite_id in suite_ids
    }
    return RunManifest(
        manifest_digest="sha256:" + "0" * 64,
        run_id=run_id,
        mode="replay",
        target=EvaluationTarget(id="fixture", kind="builtin-fixture"),
        change_profile="schema_adapter",
        gate_contract_version="evaluation-release-gates.v1",
        suite_ids=suite_ids,
        suite_revisions=revisions,
        track_ids=TRACK_IDS,
        sample_limit=100,
        concurrency=1,
        seed=19,
        created_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value(
            {"kind": "normalized-replay-policy", "suite_revisions": revisions}
        ),
        config_digest=digest_value({"normalized_suite_test": True}),
        redaction_policy="strict-no-prompts",
    )


def test_installed_composite_executes_all_tracks_deterministically_without_leaks(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_ids = _install_composite(tmp_path / "bundles", suite_store)
    manifest = _manifest("normalized-composite", suite_ids, suite_store)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

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

    first_store = tmp_path / "evaluation-a"
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
    first = json.loads(executed.output)
    second = run_evaluation(
        manifest,
        LocalArtifactStore(tmp_path / "evaluation-b"),
        suite_store=suite_store,
    )

    second_payload = second.model_dump(mode="json", exclude_none=False)
    for payload in (first, second_payload):
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
    assert first == second_payload
    assert {track["track_id"] for track in first["tracks"]} == set(TRACK_IDS)
    assert all(track["status"] == "completed" for track in first["tracks"])
    assert first["run"]["evidence_level"] == "E0"
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

    lineage_path = first_store / "runs" / manifest.run_id / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    assert lineage["resolved_snapshot"]["policy"]["id"] != "fixture-policy"
    assert (
        lineage["resolved_snapshot"]["environment"]["platform"]
        == "normalized-suite-replay"
    )
    assert any(
        row["source_id"] == "secret-arm-a"
        for row in lineage["normalized_suite_aliases"]["arm_aliases"]
    )
    assert stat.S_IMODE(lineage_path.stat().st_mode) == 0o600
    assert (
        "HIDDEN EXPECTED ANSWER"
        in (first_store / "runs" / manifest.run_id / "grading-cases.jsonl").read_text()
    )


def test_declared_track_without_qualification_artifact_is_unavailable(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    bundle = tmp_path / "missing-safety"
    _base_bundle(bundle, "missing-private-case")
    request = _suite_request(
        bundle,
        adapter_id="acebench",
        suite_id="missing-safety-suite",
        case_id="missing-private-case",
        tracks=("safety",),
        optional_roles=(),
    )
    installed = suite_store.install(request, bundle, source_root=bundle.parent)
    manifest = _manifest("missing-safety-run", (installed.id,), suite_store).model_copy(
        update={"track_ids": ("safety",)}
    )

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
