from __future__ import annotations

from pathlib import Path

import pytest
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import (
    NormalizedFault,
    NormalizedPerturbation,
    NormalizedPreference,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_install_contract import (
    BenchmarkSuiteInstallRequest,
)
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError
from evaluation_normalized_suite_test_support import (
    _decision,
    _digest,
    _manifest,
    _qualification_cases,
    _suite_request,
    _trusted_source_verifier,
    _write_jsonl,
)

pytestmark = pytest.mark.usefixtures(_trusted_source_verifier.__name__)


def _perturbation(**updates: object) -> NormalizedPerturbation:
    return NormalizedPerturbation(
        pair_id="pair",
        source_case_id="source",
        perturbed_case_id="perturbed",
        relation="invariant",
        slice_ids=("routerarena:paraphrase",),
        native_pair_count=1,
        source_record_digest=_digest("pair"),
    ).model_copy(update=updates)


@pytest.mark.parametrize(
    ("rows", "schema_invalid"),
    (
        ((_perturbation(pair_id="self", perturbed_case_id="source"),), True),
        ((_perturbation(pair_id="missing-action", relation="expected_change"),), True),
        (
            (
                _perturbation(pair_id="duplicate-a", native_pair_count=2),
                _perturbation(
                    pair_id="duplicate-b",
                    relation="expected_change",
                    expected_action_id="secret-arm-b",
                    native_pair_count=2,
                    source_record_digest=_digest("duplicate-b"),
                ),
            ),
            False,
        ),
    ),
)
def test_imported_robustness_rows_are_schema_checked_but_never_qualified(
    tmp_path: Path,
    rows: tuple[NormalizedPerturbation, ...],
    schema_invalid: bool,
) -> None:
    bundle = tmp_path / "invalid-robustness"
    case_ids = ("source", "perturbed")
    _qualification_cases(bundle, case_ids, track_ids=("routing",))
    _write_jsonl(
        bundle / "grading/decisions.jsonl",
        (_decision(case_id) for case_id in case_ids),
    )
    _write_jsonl(bundle / "grading/perturbations.jsonl", rows)
    request = _suite_request(
        bundle,
        adapter_id="routerarena",
        suite_id="invalid-robustness",
        case_id="source",
        tracks=("routing",),
        optional_roles=("decisions", "perturbations"),
        case_count=2,
    )

    store = NormalizedSuiteStore(tmp_path / "store")
    if schema_invalid:
        with pytest.raises(SuiteStoreError, match="invalid normalized perturbations"):
            store.install(request, bundle, source_root=bundle.parent)
    else:
        manifest = store.install(request, bundle, source_root=bundle.parent)
        assert manifest.qualification_receipt.evidence_level == "E0"
        assert manifest.qualification_receipt.qualified_gate_ids == ()


def _agentic_qualification_request(
    bundle: Path, faults: tuple[NormalizedFault, ...]
) -> BenchmarkSuiteInstallRequest:
    case_id = "agent-case"
    _qualification_cases(bundle, (case_id,), track_ids=("agentic",))
    _write_jsonl(
        bundle / "grading/trajectories.jsonl",
        (
            NormalizedTrajectoryStep(
                trajectory_id="trajectory-agent-case",
                step_id="step-0",
                sequence=0,
                case_id=case_id,
                terminal=False,
                state_digest_after=_digest("agent-state"),
                source_record_digest=_digest("agent-step-0"),
            ),
            NormalizedTrajectoryStep(
                trajectory_id="trajectory-agent-case",
                step_id="step-1",
                sequence=1,
                case_id=case_id,
                state_digest_before=_digest("agent-state"),
                state_digest_after=_digest("agent-state"),
                terminal=True,
                terminal_success=True,
                source_record_digest=_digest("agent-step-1"),
            ),
        ),
    )
    _write_jsonl(bundle / "grading/faults.jsonl", faults)
    return _suite_request(
        bundle,
        adapter_id="continuity-bench",
        suite_id="agentic-faults",
        case_id=case_id,
        tracks=("agentic",),
        optional_roles=("trajectories", "faults"),
    )


def _fault(**updates: object) -> NormalizedFault:
    return NormalizedFault(
        id="fault-1",
        trajectory_id="trajectory-agent-case",
        sequence=0,
        kind="timeout",
        diagnostic_scope="provider_fallback_label_and_context_preservation",
        method_id="continuity.labeled-failover.v1",
        cohort_pair_id="cohort-1",
        conversation_id="conversation-1",
        system_role="treatment",
        concurrency=1,
        failure_turn=0,
        native_repetition_count=1,
        repeated_seed_evidence=False,
        native_pair_count=1,
        failover_labeled=True,
        context_preserved=True,
        experiment_manifest_digest=_digest("fault-plan"),
        baseline_record_digest=_digest("baseline"),
        treatment_record_digest=_digest("treatment"),
        baseline_terminal_success=False,
        treatment_terminal_success=True,
        baseline_latency_ms=100,
        treatment_latency_ms=120,
        source_record_digest=_digest("fault"),
    ).model_copy(update=updates)


def test_labeled_failover_is_diagnostic_and_cannot_qualify_g6(tmp_path: Path) -> None:
    bundle = tmp_path / "valid-fault"
    request = _agentic_qualification_request(bundle, (_fault(),))

    suite_store = NormalizedSuiteStore(tmp_path / "store")
    manifest = suite_store.install(request, bundle, source_root=bundle.parent)
    qualification = manifest.qualification_receipt.qualification
    assert qualification.status == "exploratory_import"
    assert manifest.qualification_receipt.qualified_gate_ids == ()
    assert manifest.qualification_receipt.evidence_level == "E0"
    run = _manifest(
        "method-only-g6", (manifest.id,), suite_store
    ).with_semantic_updates(track_ids=("agentic",), change_profile="agent_multimodal")
    report = run_evaluation(
        run, LocalArtifactStore(tmp_path / "evaluation"), suite_store=suite_store
    )
    assert (
        next(gate for gate in report.gates if gate.id == "G6").verdict == "unavailable"
    )


@pytest.mark.parametrize(
    ("faults", "schema_invalid"),
    (
        ((_fault(failover_labeled=False),), True),
        ((_fault(context_preserved=False),), True),
        ((_fault(sequence=1),), False),
        ((_fault(), _fault()), False),
    ),
)
def test_imported_continuity_rows_are_schema_checked_but_never_qualified(
    tmp_path: Path,
    faults: tuple[NormalizedFault, ...],
    schema_invalid: bool,
) -> None:
    bundle = tmp_path / "invalid-fault"
    request = _agentic_qualification_request(bundle, faults)

    store = NormalizedSuiteStore(tmp_path / "store")
    if schema_invalid:
        with pytest.raises(SuiteStoreError, match="invalid normalized faults"):
            store.install(request, bundle, source_root=bundle.parent)
    else:
        manifest = store.install(request, bundle, source_root=bundle.parent)
        assert manifest.qualification_receipt.evidence_level == "E0"
        assert manifest.qualification_receipt.qualified_gate_ids == ()


def _online_preference(
    case_id: str, index: int, **updates: object
) -> NormalizedPreference:
    return NormalizedPreference(
        case_id=case_id,
        left_action_id="secret-arm-a",
        right_action_id="secret-arm-b",
        preference="left",
        chosen_action_id="secret-arm-a",
        assignment_id=f"assignment-{index}",
        exposure_id=f"exposure-{index}",
        exposure_probability=0.5,
        behavior_propensity=0.5,
        participant_digest=_digest(f"participant-{index}"),
        source_record_digest=_digest(f"preference-{index}"),
    ).model_copy(update=updates)


def _preference_qualification_request(
    bundle: Path, preferences: tuple[NormalizedPreference, ...]
) -> BenchmarkSuiteInstallRequest:
    case_ids = ("preference-a", "preference-b")
    _qualification_cases(bundle, case_ids, track_ids=("preference",))
    _write_jsonl(bundle / "grading/preferences.jsonl", preferences)
    return _suite_request(
        bundle,
        adapter_id="xroutebench",
        suite_id="online-preference",
        case_id=case_ids[0],
        tracks=("preference",),
        optional_roles=("preferences",),
        case_count=2,
    )


def test_research_inventory_preference_cannot_forge_executable_g9(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "valid-preference"
    preferences = (
        _online_preference("preference-a", 1),
        _online_preference("preference-b", 2),
    )
    request = _preference_qualification_request(bundle, preferences)

    manifest = NormalizedSuiteStore(tmp_path / "store").install(
        request,
        bundle,
        source_root=bundle.parent,
    )
    assert manifest.qualification_receipt.evidence_level == "E0"
    assert manifest.qualification_receipt.qualified_gate_ids == ()


@pytest.mark.parametrize(
    "second_updates",
    (
        {"assignment_id": "assignment-1"},
        {"exposure_id": "exposure-1"},
        {"participant_digest": _digest("participant-1")},
        {"exposure_probability": None},
        {"behavior_propensity": None},
    ),
)
def test_imported_preference_identity_never_becomes_online_evidence(
    tmp_path: Path,
    second_updates: dict[str, object],
) -> None:
    bundle = tmp_path / "invalid-preference"
    preferences = (
        _online_preference("preference-a", 1),
        _online_preference("preference-b", 2, **second_updates),
    )
    request = _preference_qualification_request(bundle, preferences)

    manifest = NormalizedSuiteStore(tmp_path / "store").install(
        request, bundle, source_root=bundle.parent
    )
    assert manifest.qualification_receipt.evidence_level == "E0"
    assert manifest.qualification_receipt.qualified_gate_ids == ()
