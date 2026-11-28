"""The generic promotion-evidence replay rejects tampered records."""

import json

import pytest
from recipe_lab.contract import ContractError, ExperimentSpec
from recipe_lab.ledger import compute_record_id
from recipe_lab.live import measurements_from_rows, resolve_gates
from recipe_lab.promotion_evidence import verify_promotion_evidence
from tests.fixtures import VALID_SPEC


def _live_record(tmp_path, **overrides):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(json.dumps(VALID_SPEC), encoding="utf-8")
    spec = ExperimentSpec.from_yaml(str(spec_path))
    record = {
        "round": 1,
        "ts": "2026-01-01T00:00:00Z",
        "candidate": {"kind": "config", "ref": "config-rev:x", "provenance": {}},
        "tier": "live",
        "baseline_ref": "bench:baseline-id",
        "metrics": {spec.objective.metric_key: 0.805},
        "verdict": "INCONCLUSIVE",
        "dataset_hash": spec.workload.live_dataset_hash("dev"),
        "split": "dev",
        "gates": {},
        "bench_refs": [],
        "notes": "baseline=0.8",
        "verification": {
            "baseline": 0.8,
            "guard_measurements": [],
            "evidence_quality": {},
        },
        "paired": None,
    }
    record.update(overrides)
    guards, _ = resolve_gates(
        spec.gates, measurements_from_rows(record["verification"]["guard_measurements"])
    )
    record["gates"] = {guard.name: "pass" for guard in guards}
    record["record_id"] = compute_record_id(record)
    return spec, record


def test_inconclusive_record_passes_early_audit(tmp_path):
    spec, record = _live_record(tmp_path)
    # An INCONCLUSIVE record carries no promotion claim; the replay reaches
    # the verdict re-check and rejects it for lacking PASS support.
    with pytest.raises(ContractError, match="do not support PASS"):
        verify_promotion_evidence(spec, record)


def test_tampered_record_id_is_rejected(tmp_path):
    spec, record = _live_record(tmp_path)
    record["record_id"] = "rec-tampered000"
    with pytest.raises(ContractError, match="content differs from its record_id"):
        verify_promotion_evidence(spec, record)


def test_dataset_hash_drift_is_rejected(tmp_path):
    spec, record = _live_record(tmp_path)
    record["dataset_hash"] = "sha256:" + "f" * 64
    record["record_id"] = compute_record_id(record)
    with pytest.raises(ContractError, match="dataset_hash"):
        verify_promotion_evidence(spec, record)
