from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import (
    DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.live_executor import LiveRawResult
from cli.evaluation.normalized_suite_inputs import load_selected_cases
from cli.evaluation.normalized_suite_live_robustness import (
    DECLARED_SHIFT_LIVE_METHOD_ID,
    attach_live_declared_shift_evidence,
)
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import NormalizedPerturbation
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError
from evaluation_normalized_suite_test_support import (
    _catalog,
    _digest,
    _live_manifest,
    _qualification_cases,
    _suite_request,
    _target_arms,
    _trusted_source_verifier,
    _write_jsonl,
)

pytestmark = pytest.mark.usefixtures(_trusted_source_verifier.__name__)


def _install_registered_pair(
    root: Path,
    store: NormalizedSuiteStore,
    *,
    registered: bool = True,
    native_pair_count: int = 1,
) -> str:
    case_ids = ("source", "perturbed")
    _qualification_cases(root, case_ids, track_ids=("routing",))
    _write_jsonl(
        root / "grading/perturbations.jsonl",
        (
            NormalizedPerturbation(
                pair_id="pair-1",
                source_case_id="source",
                perturbed_case_id="perturbed",
                relation="invariant",
                slice_ids=("declared:paraphrase",),
                native_pair_count=native_pair_count,
                source_record_digest=_digest("declared-pair-1"),
            ),
        ),
    )
    request = _suite_request(
        root,
        adapter_id="routerarena",
        suite_id="registered-declared-shift",
        case_id="source",
        tracks=("routing",),
        optional_roles=("perturbations",),
        case_count=2,
    )
    if registered:
        request = request.model_copy(
            update={"normalization_origin": "registered_parser_import"}
        )
    return store.install(request, root, source_root=root.parent).id


@pytest.mark.parametrize(
    ("target_model", "verdict"),
    (("provider-strong", "pass"), ("provider-fast", "fail")),
)
def test_registered_pinned_pairs_get_server_portable_live_g4_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_model: str,
    verdict: str,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = _install_registered_pair(tmp_path / "bundle", suite_store)
    source_catalog = _catalog(suite_store).get(suite_id)
    assert source_catalog.evidence_level == "E0"
    assert source_catalog.modes == ("replay", "live")
    source_method = next(
        method
        for method in source_catalog.methods
        if method.id == DECLARED_SHIFT_LIVE_METHOD_ID
    )
    assert source_method.status == "configured"
    assert source_method.qualified_gate_ids == ("G4",)
    assert source_method.evidence_source == "server_brokered_live"
    assert all(
        method.status == "configured" and not method.qualified_gate_ids
        for method in source_catalog.methods
        if method.evidence_source == "normalized_import"
    )
    manifest = _live_manifest(
        f"declared-shift-{verdict}", suite_id, suite_store, track_ids=("routing",)
    ).with_semantic_updates(sample_limit=2)

    def execute(visible: Any, **_kwargs: object) -> LiveRawResult:
        cases = tuple(visible.cases)
        assert len(cases) == 2
        # The invariant source is always strong. For the fail case, only one
        # member changes; which member is the target is discovered below by
        # its deterministic opaque order, so make the later case the target.
        records = []
        for index, case in enumerate(cases):
            selected = "provider-strong"
            if verdict == "fail" and index == 1:
                selected = target_model
            records.append(
                ExecutionRecord(
                    id=f"routing-{case.id}",
                    track_id="routing",
                    case_id=case.id,
                    attempt_id=f"attempt-{case.id}",
                    status="succeeded",
                    selected_arm_id=selected,
                    selection_status="selected",
                    success=True,
                    latency_ms=2.0,
                    broker_receipt=_digest(f"receipt-{case.id}"),
                )
            )
        return LiveRawResult(
            records=records,
            discovered_entrypoints=("entrypoint-a",),
            routing_traces=(),
            chat_results={},
            model_pool_results={},
            model_pool_arm_ids=(),
            joint_results={},
        )

    monkeypatch.setattr(
        "cli.evaluation.normalized_suite_live_executor.execute_live_raw", execute
    )
    store = LocalArtifactStore(tmp_path / "evaluation")
    report = run_evaluation(manifest, store, suite_store=suite_store)
    records = [
        ExecutionRecord.model_validate_json(row)
        for row in store.read_run_bytes(manifest.run_id, "records.jsonl").splitlines()
    ]
    methods = [row.robustness for row in records if row.robustness is not None]
    assert len(methods) == 1
    assert methods[0].method_id == DECLARED_SHIFT_LIVE_METHOD_ID
    assert methods[0].suite_id == suite_id
    assert all(
        row.evidence_kind == DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID
        and row.broker_receipt is not None
        for row in records
    )
    assert report.run.evidence_level == "E4"
    assert next(gate for gate in report.gates if gate.id == "G4").verdict == verdict


def test_missing_receipt_never_gets_live_g4_candidate(
    tmp_path: Path,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    bundle = tmp_path / "bundle"
    suite_id = _install_registered_pair(bundle, suite_store)
    manifest = suite_store.get_suite_manifest(suite_id)
    selected, _ = load_selected_cases(
        suite_store,
        (manifest,),
        2,
        19,
        ("routing",),
        "normalized-suite-live.v1",
    )
    records = [
        ExecutionRecord(
            id=f"routing-{case.visible.id}",
            track_id="routing",
            case_id=case.visible.id,
            attempt_id=f"attempt-{case.visible.id}",
            status="succeeded",
            selected_arm_id="arm-strong",
            success=True,
            latency_ms=1,
            broker_receipt=(None if index == 0 else _digest("receipt")),
        )
        for index, case in enumerate(selected)
    ]
    attached = attach_live_declared_shift_evidence(
        records=records,
        selected=selected,
        manifests=(manifest,),
        store=suite_store,
        arms=_target_arms(),
    )
    assert all(row.robustness is None for row in attached)
    assert all(row.evidence_kind is None for row in attached)


@pytest.mark.parametrize(("registered", "native_pair_count"), ((False, 1), (True, 2)))
def test_unverified_parser_or_native_count_drift_never_gets_live_g4_candidate(
    tmp_path: Path,
    registered: bool,
    native_pair_count: int,
) -> None:
    suite_store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = _install_registered_pair(
        tmp_path / "bundle",
        suite_store,
        registered=registered,
        native_pair_count=native_pair_count,
    )
    if registered:
        with pytest.raises(SuiteStoreError, match="native pair count drifted"):
            _catalog(suite_store).get(suite_id)
        return
    source_catalog = _catalog(suite_store).get(suite_id)
    assert source_catalog.evidence_level == "E0"
    assert source_catalog.modes == ("replay",)
    assert all(
        method.id != DECLARED_SHIFT_LIVE_METHOD_ID for method in source_catalog.methods
    )
    manifest = suite_store.get_suite_manifest(suite_id)
    selected, _ = load_selected_cases(
        suite_store,
        (manifest,),
        2,
        19,
        ("routing",),
        "normalized-suite-live.v1",
    )
    records = [
        ExecutionRecord(
            id=f"routing-{case.visible.id}",
            track_id="routing",
            case_id=case.visible.id,
            attempt_id=f"attempt-{case.visible.id}",
            status="succeeded",
            selected_arm_id="arm-strong",
            success=True,
            latency_ms=1,
            broker_receipt=_digest(f"receipt-{index}"),
        )
        for index, case in enumerate(selected)
    ]
    attached = attach_live_declared_shift_evidence(
        records=records,
        selected=selected,
        manifests=(manifest,),
        store=suite_store,
        arms=_target_arms(),
    )
    assert all(row.robustness is None for row in attached)
